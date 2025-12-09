"""
Auto data collector: runs PingPong with simple heuristic agents and logs per-player frame features+action.
Generates rows with columns: player,frame,ball_x,ball_y,ball_vx,ball_vy,self_px,opp_px,ball_served,serving_is_self,blocker_x,action
"""
import os
import sys
import csv
import random
import argparse
# Ensure repo root is on sys.path so `src` package imports work
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.game import PingPong
from mlgame.utils.enum import get_ai_name
from ml.landing import simulate_landing_x

# Configuration
OUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_CSV = os.path.join(OUT_DIR, 'play_data_auto.csv')
TARGET_ROWS = 5000  # adjust as needed
RANDOM_ACTION_PROB = 0.10  # chance to take a random non-serve action
RANDOM_SERVE_FLIP_PROB = 0.15  # chance to flip serve direction
ANTICIPATE_JITTER = 4.0  # jitter added to 2P anticipation

os.makedirs(OUT_DIR, exist_ok=True)

fieldnames = [
    'player','frame','ball_x','ball_y','ball_vx','ball_vy','self_px','opp_px','ball_served','serving_is_self','blocker_x',
    # predicted landing x (simple linear projection)
    'pred_landing_x',
    # history features (previous 1)
    'prev1_ball_vx','prev1_ball_vy','prev1_self_px',
    # history features (previous 2)
    'prev2_ball_vx','prev2_ball_vy','prev2_self_px',
    'action'
]

# Simple heuristic policy for auto-play
def heuristic_action(side, scene, randomize=False):
    # side: '1P' or '2P'
    ball_x, ball_y = scene['ball']
    ball_vx, ball_vy = scene['ball_speed']
    p1x, p1y = scene['platform_1P']
    p2x, p2y = scene['platform_2P']
    self_px = p1x if side == '1P' else p2x

    # If ball not served and this side should serve, use different serve preferences
    if not scene['ball_served'] and scene['serving_side'] == side:
        # 1P prefers SERVE_TO_RIGHT, 2P prefers SERVE_TO_LEFT (asymmetric)
        serve = 'SERVE_TO_RIGHT' if side == '1P' else 'SERVE_TO_LEFT'
        # occasionally flip serve direction for diversity
        if randomize and random.random() < RANDOM_SERVE_FLIP_PROB:
            serve = 'SERVE_TO_LEFT' if serve == 'SERVE_TO_RIGHT' else 'SERVE_TO_RIGHT'
        return serve

    # Different heuristics for two sides to create asymmetric data:
    if side == '1P':
        # 1P: simple follow-the-ball
        if abs(self_px - ball_x) <= 5:
            return 'NONE'
        return 'MOVE_LEFT' if self_px > ball_x else 'MOVE_RIGHT'
    else:
        # 2P: anticipate ball by aiming to ball_x + k*ball_vx (predictive)
        anticipate_x = ball_x + 8 * ball_vx
        # add jitter to anticipation for variability when requested
        if randomize:
            anticipate_x += random.uniform(-ANTICIPATE_JITTER, ANTICIPATE_JITTER)
        if abs(self_px - anticipate_x) <= 6:
            return 'NONE'
        return 'MOVE_LEFT' if self_px > anticipate_x else 'MOVE_RIGHT'


def collect(target_rows=TARGET_ROWS, out_csv=OUT_CSV, difficulty='HARD', randomize=True):
    # Use HARD difficulty by default to include blocker & faster play
    game = PingPong(difficulty=difficulty, game_over_score=3)

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        from collections import deque
        # keep last 2 scene_info dicts for history features
        last_scenes = deque(maxlen=2)

        rows = 0
        # keep running games (reset between matches) until we collect enough rows
        while rows < target_rows:
            # Build scene and choose actions for both players
            scene_dict = game.get_data_from_game_to_player()
            ai1 = get_ai_name(0)
            ai2 = get_ai_name(1)
            scene = scene_dict[ai1]

            action1 = heuristic_action('1P', scene, randomize=randomize)
            action2 = heuristic_action('2P', scene, randomize=randomize)

            # with some probability, take a random non-serve action to increase diversity
            if randomize:
                if random.random() < RANDOM_ACTION_PROB:
                    action1 = random.choice(['NONE', 'MOVE_LEFT', 'MOVE_RIGHT'])
                if random.random() < RANDOM_ACTION_PROB:
                    action2 = random.choice(['NONE', 'MOVE_LEFT', 'MOVE_RIGHT'])

            commands = {ai1: action1, ai2: action2}

            # push current scene to history buffer before logging so prev features reflect past frames
            last_scenes.append(scene.copy())

            # Log both players as separate rows
            for side, action in (('1P', action1), ('2P', action2)):
                ball_x, ball_y = scene['ball']
                ball_vx, ball_vy = scene['ball_speed']
                p1x, p1y = scene['platform_1P']
                p2x, p2y = scene['platform_2P']
                blocker_x, _ = scene.get('blocker', (0, 0))

                # build history features: default to 0 if not available
                prev1_ball_vx = prev1_ball_vy = prev1_self_px = 0.0
                prev2_ball_vx = prev2_ball_vy = prev2_self_px = 0.0
                if len(last_scenes) >= 1:
                    s1 = last_scenes[-1]
                    pvx, pvy = s1['ball_speed']
                    p1x_, _ = s1['platform_1P']
                    p2x_, _ = s1['platform_2P']
                    prev1_ball_vx = pvx
                    prev1_ball_vy = pvy
                    prev1_self_px = p1x_ if side == '1P' else p2x_
                if len(last_scenes) >= 2:
                    s2 = last_scenes[-2]
                    pvx2, pvy2 = s2['ball_speed']
                    p1x2, _ = s2['platform_1P']
                    p2x2, _ = s2['platform_2P']
                    prev2_ball_vx = pvx2
                    prev2_ball_vy = pvy2
                    prev2_self_px = p1x2 if side == '1P' else p2x2

                # estimate platform horizontal velocities from history (current - previous)
                p1_vx = 0
                p2_vx = 0
                if len(last_scenes) >= 2:
                    prev_scene = last_scenes[-2]
                    prev_p1x, _ = prev_scene['platform_1P']
                    prev_p2x, _ = prev_scene['platform_2P']
                    p1_vx = p1x - prev_p1x
                    p2_vx = p2x - prev_p2x

                try:
                    from ml.landing import get_predicted_landing
                    pred_landing_x = get_predicted_landing(scene, max_steps=200, p1_vx=p1_vx, p2_vx=p2_vx)
                except Exception:
                    pred_landing_x = ball_x

                row = {
                    'player': side,
                    'frame': scene['frame'],
                    'ball_x': ball_x,
                    'ball_y': ball_y,
                    'ball_vx': ball_vx,
                    'ball_vy': ball_vy,
                    'self_px': p1x if side == '1P' else p2x,
                    'opp_px': p2x if side == '1P' else p1x,
                    'ball_served': int(scene['ball_served']),
                    'serving_is_self': int(scene['serving_side'] == side),
                    'blocker_x': blocker_x,
                    'pred_landing_x': pred_landing_x,
                    'prev1_ball_vx': prev1_ball_vx,
                    'prev1_ball_vy': prev1_ball_vy,
                    'prev1_self_px': prev1_self_px,
                    'prev2_ball_vx': prev2_ball_vx,
                    'prev2_ball_vy': prev2_ball_vy,
                    'prev2_self_px': prev2_self_px,
                    'action': action,
                }
                writer.writerow(row)
                rows += 1
                if rows >= target_rows:
                    break

            # periodic progress
            if rows % 500 == 0 and rows > 0:
                print(f'Collected {rows} rows...')

            # Advance game
            result = game.update(commands)
            # If the round ended, reset and continue collecting
            if result in ('RESET', 'QUIT'):
                game.reset()

    print(f'Finished collecting {rows} rows to {out_csv}')

if __name__ == '__main__':
    collect()
