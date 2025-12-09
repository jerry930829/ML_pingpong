"""
Collect targeted gameplay rows where landing prediction or ball speed indicate hard-to-catch scenarios.

Writes to `ml/data/play_data_targeted.csv`.

Usage: `python -m ml.collect_targeted --target_rows 10000`
"""
import os
import sys
import csv
import argparse
import random

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.game import PingPong
from mlgame.utils.enum import get_ai_name
from ml.landing import get_predicted_landing

OUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_CSV = os.path.join(OUT_DIR, 'play_data_targeted.csv')
os.makedirs(OUT_DIR, exist_ok=True)

FIELDNAMES = [
    'player','frame','ball_x','ball_y','ball_vx','ball_vy','self_px','opp_px','ball_served','serving_is_self','blocker_x','pred_landing_x',
    'prev1_ball_vx','prev1_ball_vy','prev1_self_px','prev2_ball_vx','prev2_ball_vy','prev2_self_px','action'
]


def collect(target_rows=10000, difficulty='HARD', randomize=True, speed_thresh=12.0, landing_dx_thresh=50.0):
    game = PingPong(difficulty=difficulty, game_over_score=3)

    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        from collections import deque
        last_scenes = deque(maxlen=2)

        rows = 0
        while rows < target_rows:
            scene_dict = game.get_data_from_game_to_player()
            ai1 = get_ai_name(0)
            ai2 = get_ai_name(1)
            scene = scene_dict[ai1]

            # simple heuristics for actions
            # let collector play heuristics but we only save frames matching filters
            def heuristic(side):
                ball_x, ball_y = scene['ball']
                p1x, p1y = scene['platform_1P']
                p2x, p2y = scene['platform_2P']
                self_px = p1x if side == '1P' else p2x
                if not scene['ball_served'] and scene['serving_side'] == side:
                    return 'SERVE_TO_RIGHT' if side == '1P' else 'SERVE_TO_LEFT'
                if side == '1P':
                    if abs(self_px - ball_x) <= 5:
                        return 'NONE'
                    return 'MOVE_LEFT' if self_px > ball_x else 'MOVE_RIGHT'
                else:
                    anticipate_x = ball_x + 8 * scene['ball_speed'][0]
                    if abs(self_px - anticipate_x) <= 6:
                        return 'NONE'
                    return 'MOVE_LEFT' if self_px > anticipate_x else 'MOVE_RIGHT'

            action1 = heuristic('1P')
            action2 = heuristic('2P')
            if randomize:
                if random.random() < 0.05:
                    action1 = random.choice(['NONE','MOVE_LEFT','MOVE_RIGHT'])
                if random.random() < 0.05:
                    action2 = random.choice(['NONE','MOVE_LEFT','MOVE_RIGHT'])

            last_scenes.append(scene.copy())

            # compute pred landing and filter
            p1x, p1y = scene['platform_1P']
            p2x, p2y = scene['platform_2P']
            p1_vx = p2_vx = 0
            if len(last_scenes) >= 2:
                prev = last_scenes[-2]
                p1_vx = p1x - prev['platform_1P'][0]
                p2_vx = p2x - prev['platform_2P'][0]

            try:
                res = get_predicted_landing(scene, p1_vx=p1_vx, p2_vx=p2_vx, return_steps=True)
                if isinstance(res, tuple):
                    pred_lx, steps = res
                else:
                    pred_lx = res
            except Exception:
                pred_lx = scene['ball'][0]

            ball_vx, ball_vy = scene['ball_speed']
            speed = (ball_vx**2 + ball_vy**2)**0.5

            # filter condition: either very high speed or landing far from platform center
            cond = False
            # check 1P side condition (we'll record both players rows but filter when 1P scenario occurs)
            if abs(pred_lx - p1x) >= landing_dx_thresh or speed >= speed_thresh:
                cond = True

            # Log both players but only when cond true
            for side, action in (('1P', action1), ('2P', action2)):
                p1x_cur, _ = scene['platform_1P']
                p2x_cur, _ = scene['platform_2P']
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

                row = {
                    'player': side,
                    'frame': scene['frame'],
                    'ball_x': scene['ball'][0],
                    'ball_y': scene['ball'][1],
                    'ball_vx': scene['ball_speed'][0],
                    'ball_vy': scene['ball_speed'][1],
                    'self_px': p1x_cur if side == '1P' else p2x_cur,
                    'opp_px': p2x_cur if side == '1P' else p1x_cur,
                    'ball_served': int(scene['ball_served']),
                    'serving_is_self': int(scene['serving_side'] == side),
                    'blocker_x': scene.get('blocker', (0,0))[0],
                    'pred_landing_x': pred_lx,
                    'prev1_ball_vx': prev1_ball_vx,
                    'prev1_ball_vy': prev1_ball_vy,
                    'prev1_self_px': prev1_self_px,
                    'prev2_ball_vx': prev2_ball_vx,
                    'prev2_ball_vy': prev2_ball_vy,
                    'prev2_self_px': prev2_self_px,
                    'action': action,
                }

                if cond:
                    writer.writerow(row)
                    rows += 1
                    if rows % 500 == 0:
                        print(f'Collected {rows} targeted rows...')
                    if rows >= target_rows:
                        break

            result = game.update({get_ai_name(0): action1, get_ai_name(1): action2})
            if result in ('RESET', 'QUIT'):
                game.reset()

    print(f'Finished collecting {rows} targeted rows to {OUT_CSV}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--target_rows', type=int, default=10000)
    p.add_argument('--difficulty', type=str, default='HARD')
    p.add_argument('--speed_thresh', type=float, default=12.0)
    p.add_argument('--landing_dx_thresh', type=float, default=50.0)
    args = p.parse_args()
    collect(target_rows=args.target_rows, difficulty=args.difficulty, speed_thresh=args.speed_thresh, landing_dx_thresh=args.landing_dx_thresh)
