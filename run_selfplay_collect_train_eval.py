"""
Self-play pipeline:
- Run Collector 1P vs Collector 2P for N games and collect data (ml/data/1p_data.csv, ml/data/2p_data.csv)
- Train Decision Tree models for 1P and 2P using existing training scripts
- Run an evaluation match between trained 1P and 2P models and report the winner

Usage:
    python run_selfplay_collect_train_eval.py --games 10 --difficulty HARD --game_over_score 3 --init_vel 7

Note: This script expects existing collector modules under `ml/` and training scripts
`ml/train_dt_1P.py` and `ml/train_dt_2P.py` to be present.
"""
import os
import sys
import argparse
import subprocess
import time

sys.path.insert(0, os.path.dirname(__file__))

import pygame
from mlgame.view.view import PygameView
from mlgame.game.generic import quit_or_esc
from mlgame.utils.enum import get_ai_name
from src.game import PingPong

# Import collectors and AI modules
from ml.ml_play_collector_1P import MLPlay as Collector1P
from ml.ml_play_collector_2P import MLPlay as Collector2P

# Paths to training scripts
TRAIN_1P = os.path.join("ml", "train_dt_1P.py")
TRAIN_2P = os.path.join("ml", "train_dt_2P.py")

# AI modules for evaluation
from ml.ml_play_dt_1P import MLPlay as AI_1P
from ml.ml_play_dt_2P import MLPlay as AI_2P


def build_scene_info(game, frame_count):
    blocker = None
    if hasattr(game, '_blocker') and game._blocker.rect.x < 1000:
        blocker = [game._blocker.rect.x, game._blocker.rect.y]

    status_obj = game.get_game_status()
    status_name = status_obj.name if hasattr(status_obj, "name") else str(status_obj)

    scene_info = {
        "frame": frame_count,
        "status": status_name,
        "ball": [game._ball.rect.x, game._ball.rect.y],
        "ball_speed": list(game._ball.speed),
        "ball_served": game._ball_served,
        "serving_side": "1P" if frame_count % 2 == 0 else "2P",
        "platform_1P": [game._platform_1P.rect.x, game._platform_1P.rect.y],
        "platform_2P": [game._platform_2P.rect.x, game._platform_2P.rect.y],
        "blocker": blocker,
    }
    return scene_info


def run_collection(games, difficulty, game_over_score, init_vel, fps=30, show_window=True):
    pygame.init()
    game = PingPong(difficulty=difficulty, game_over_score=game_over_score, init_vel=init_vel)

    scene_init_info_dict = game.get_scene_init_data()
    view = PygameView(scene_init_info_dict) if show_window else None

    collector1 = Collector1P(ai_name="1P")
    collector2 = Collector2P(ai_name="2P")

    frame_count = 0
    completed = 0
    clock = pygame.time.Clock()

    print(f"Starting collection: target {games} games")
    while game.is_running and completed < games and not quit_or_esc():
        clock.tick_busy_loop(fps)
        scene_info = build_scene_info(game, frame_count)

        # If not alive, still pass scene_info; collectors expect status == 'GAME_ALIVE'
        cmd1 = collector1.update(scene_info)
        cmd2 = collector2.update(scene_info)

        commands = {get_ai_name(0): cmd1, get_ai_name(1): cmd2}
        result = game.update(commands)

        if view:
            view.draw(game.get_scene_progress_data())

        frame_count += 1

        if result == "RESET":
            collector1.reset()
            collector2.reset()
            game.reset()
            if view:
                view.reset()
            completed += 1
            print(f"Completed collection games: {completed}/{games}")

    if view:
        pygame.quit()
    print("Collection finished")


def run_training():
    # Run both training scripts sequentially
    print("Training 1P model...")
    rc1 = subprocess.call([sys.executable, TRAIN_1P])
    print(f"Train 1P exit code: {rc1}")

    print("Training 2P model...")
    rc2 = subprocess.call([sys.executable, TRAIN_2P])
    print(f"Train 2P exit code: {rc2}")

    return rc1 == 0 and rc2 == 0


def run_evaluation(game_over_score, difficulty, init_vel, fps=30):
    pygame.init()
    game = PingPong(difficulty=difficulty, game_over_score=game_over_score, init_vel=init_vel)
    scene_init_info_dict = game.get_scene_init_data()
    view = PygameView(scene_init_info_dict)

    # Load AI models
    ai1 = AI_1P(ai_name="1P")
    ai2 = AI_2P(ai_name="2P")

    frame_count = 0
    clock = pygame.time.Clock()

    print("Starting evaluation match: 1P (AI) vs 2P (AI)")
    while game.is_running and not quit_or_esc():
        clock.tick_busy_loop(fps)
        scene_info = build_scene_info(game, frame_count)

        cmd1 = ai1.update(scene_info)
        cmd2 = ai2.update(scene_info)

        commands = {get_ai_name(0): cmd1, get_ai_name(1): cmd2}
        result = game.update(commands)

        view.draw(game.get_scene_progress_data())

        frame_count += 1

        if result == "RESET":
            # Determine winner from game (check score in game._score if available)
            try:
                score1 = game._score[0]
                score2 = game._score[1]
            except Exception:
                score1 = None
                score2 = None

            ai1.reset()
            ai2.reset()
            game.reset()
            view.reset()

            pygame.quit()

            print(f"Evaluation finished: final score 1P={score1} 2P={score2}")
            if score1 is not None and score2 is not None:
                if score1 > score2:
                    print("Winner: 1P (AI)")
                elif score2 > score1:
                    print("Winner: 2P (AI)")
                else:
                    print("Result: Draw")
            return score1, score2

    pygame.quit()
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Self-play data collection, training, and evaluation")
    parser.add_argument("--games", type=int, default=10, help="Number of self-play games to collect")
    parser.add_argument("--difficulty", default="HARD", choices=["NORMAL", "HARD"]) 
    parser.add_argument("--game_over_score", type=int, default=3)
    parser.add_argument("--init_vel", type=int, default=7)
    parser.add_argument("--no-train", action="store_true", help="Skip training step")
    args = parser.parse_args()

    # Run collection
    run_collection(args.games, args.difficulty, args.game_over_score, args.init_vel)

    # Train
    if not args.no_train:
        ok = run_training()
        if not ok:
            print("Training failed; aborting evaluation")
            return

    # Evaluate models against each other
    run_evaluation(args.game_over_score, args.difficulty, args.init_vel)


if __name__ == "__main__":
    main()
