"""
Hybrid data collection runner: Player 1 (collector heuristic) vs Player 2 (human).

This script allows a human player to control Player 2 while Player 1 uses a simple
heuristic strategy. During gameplay, Player 1's state->action pairs are logged to CSV
for training the Decision Tree model with more realistic gameplay data.

Usage:
    python run_collector_1P_vs_human_2P.py [--difficulty NORMAL|HARD] [--game_over_score N] [--init_vel N]

Example:
    python run_collector_1P_vs_human_2P.py --difficulty HARD --game_over_score 3 --init_vel 7

Keyboard Controls (2P / Player 2):
    - Q: Serve to left
    - E: Serve to right
    - A: Move left
    - D: Move right

1P (Player 1) uses a simple heuristic strategy and logs data to ml/data/1p_data.csv
"""
import sys
import argparse
import os
sys.path.insert(0, os.path.dirname(__file__))

import pygame
import csv
from datetime import datetime
from mlgame.view.view import PygameView
from mlgame.game.generic import quit_or_esc
from mlgame.utils.enum import get_ai_name
from src.game import PingPong


# Key-to-action mapping for 2P human control
KEY_TO_ACTION_2P = {
    pygame.K_a: "MOVE_LEFT",
    pygame.K_d: "MOVE_RIGHT",
    pygame.K_q: "SERVE_TO_LEFT",
    pygame.K_e: "SERVE_TO_RIGHT",
}


def get_2p_action_from_keyboard():
    """
    Read keyboard input and return 2P action.
    """
    keys = pygame.key.get_pressed()
    for key, action in KEY_TO_ACTION_2P.items():
        if keys[key]:
            return action
    return "NONE"


def build_scene_info(game, frame_count):
    """
    Build scene_info dict from game state to match MLGame format.
    """
    blocker = None
    if hasattr(game, '_blocker') and game._blocker.rect.x < 1000:
        blocker = [game._blocker.rect.x, game._blocker.rect.y]
    
    # Normalize status to the enum name (e.g., 'GAME_ALIVE') when possible
    status_obj = game.get_game_status()
    status_name = status_obj.name if hasattr(status_obj, "name") else str(status_obj)

    scene_info = {
        "frame": frame_count,
        "status": status_name,
        "ball": [game._ball.rect.x, game._ball.rect.y],
        "ball_speed": list(game._ball.speed),
        "ball_served": game._ball_served,
        "serving_side": "1P" if frame_count % 2 == 0 else "2P",  # Simplified
        "platform_1P": [game._platform_1P.rect.x, game._platform_1P.rect.y],
        "platform_2P": [game._platform_2P.rect.x, game._platform_2P.rect.y],
        "blocker": blocker,
    }
    return scene_info


class Collector1P:
    """
    Simple heuristic collector that logs state->action pairs to CSV.
    """
    def __init__(self, csv_path):
        self.ball_served = False
        self.side = "1P"
        self.csv_path = csv_path
        
        # Create header if file doesn't exist
        if not os.path.exists(csv_path):
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "frame",
                    "ball_x",
                    "ball_y",
                    "ball_vx",
                    "ball_vy",
                    "platform_x",
                    "platform_y",
                    "blocker_x",
                    "blocker_y",
                    "ball_served",
                    "serving_side",
                    "action",
                ])
    
    def _choose_action(self, scene_info):
        """
        Simple heuristic: serve to right, then move platform toward ball x.
        """
        if not self.ball_served:
            return "SERVE_TO_RIGHT"

        ball_x = scene_info.get("ball", [0, 0])[0]
        platform_x = scene_info.get("platform_1P", [0, 0])[0]

        # platform width is 40, consider center
        platform_center = platform_x + 20

        if abs(ball_x - platform_center) <= 5:
            return "NONE"
        elif ball_x < platform_center:
            return "MOVE_LEFT"
        else:
            return "MOVE_RIGHT"
    
    def update(self, scene_info):
        """
        Choose action and log state->action pair to CSV.
        """
        # Only log while game is alive
        if "GAME_ALIVE" not in str(scene_info.get("status")):
            # provide small debug hint if status mismatches
            # (helps identify why collector may return RESET)
            # print(f"[Collector1P] Ignoring frame {scene_info.get('frame')} status={scene_info.get('status')}")
            return "RESET"

        action = self._choose_action(scene_info)

        # log features and action
        ball = scene_info.get("ball", [0, 0])
        ball_speed = scene_info.get("ball_speed", [0, 0])
        platform = scene_info.get("platform_1P", [0, 0])
        blocker = scene_info.get("blocker") or [-1, -1]

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                scene_info.get("frame", 0),
                ball[0],
                ball[1],
                ball_speed[0],
                ball_speed[1],
                platform[0],
                platform[1],
                blocker[0] if blocker is not None else -1,
                blocker[1] if blocker is not None else -1,
                int(scene_info.get("ball_served", False)),
                scene_info.get("serving_side"),
                action,
            ])

        # once we serve, flip flag
        if not self.ball_served and action.startswith("SERVE"):
            self.ball_served = True

        return action
    
    def reset(self):
        """Reset state for new game."""
        self.ball_served = False


def main():
    parser = argparse.ArgumentParser(description="Data Collection: 1P Heuristic vs 2P Human")
    parser.add_argument("--difficulty", default="HARD", choices=["NORMAL", "HARD"],
                        help="Game difficulty (default: HARD)")
    parser.add_argument("--game_over_score", type=int, default=3,
                        help="Score to win (default: 3)")
    parser.add_argument("--init_vel", type=int, default=7,
                        help="Initial ball velocity (default: 7)")
    args = parser.parse_args()

    pygame.init()
    
    # Initialize game with specified parameters
    game = PingPong(
        difficulty=args.difficulty,
        game_over_score=args.game_over_score,
        init_vel=args.init_vel
    )
    
    # Initialize 1P collector
    data_path = os.path.join(os.path.dirname(__file__), "ml", "data", "1p_data.csv")
    collector_1p = Collector1P(data_path)
    
    scene_init_info_dict = game.get_scene_init_data()
    game_view = PygameView(scene_init_info_dict)
    
    frame_count = 0
    games_completed = 0
    total_frames_logged = 0
    FPS = 30
    clock = pygame.time.Clock()
    
    debug_frame_limit = 50  # Only log first 50 frames per game
    
    print("=" * 70)
    print("Data Collection: Player 1 (Heuristic) vs Player 2 (Human)")
    print("=" * 70)
    print(f"Difficulty: {args.difficulty}")
    print(f"Game Over Score: {args.game_over_score}")
    print(f"Initial Ball Velocity: {args.init_vel}")
    print(f"Data will be saved to: {data_path}")
    print("\nPlayer 2 Controls:")
    print("  Q: Serve to left")
    print("  E: Serve to right")
    print("  A: Move left")
    print("  D: Move right")
    print("\nPlayer 1 uses a simple heuristic strategy.")
    print("State->action pairs are logged to CSV for training.")
    print("Press ESC or close window to quit.")
    print("=" * 70)
    print()
    
    # Wait for initial state
    initial_wait = 0
    while game.is_running and not quit_or_esc() and initial_wait < 5:
        clock.tick_busy_loop(FPS)
        scene_info = build_scene_info(game, initial_wait)
        # Accept either 'GAME_ALIVE' or 'GameStatus.GAME_ALIVE' by checking substring
        if scene_info and "GAME_ALIVE" in str(scene_info.get("status")):
            break
        initial_wait += 1
    
    game_frames = 0
    
    while game.is_running and not quit_or_esc():
        clock.tick_busy_loop(FPS)
        
        # Build current game state from game objects
        scene_info = build_scene_info(game, frame_count)
        
        # Get 1P action from collector (with logging)
        command_1p = collector_1p.update(scene_info)
        
        # Get 2P action from keyboard
        command_2p = get_2p_action_from_keyboard()
        
        # Combine commands using get_ai_name convention for game.update()
        commands = {
            get_ai_name(0): command_1p,  # 0 = 1P
            get_ai_name(1): command_2p,  # 1 = 2P
        }
        
        # Debug output (first 50 frames per game)
        if game_frames < debug_frame_limit and "GAME_ALIVE" in str(scene_info.get("status")):
            print(f"[GAME {games_completed + 1}] frame {game_frames} -> 1P: {command_1p:20s} | 2P: {command_2p}")
            total_frames_logged += 1
        
        # Update game state
        result = game.update(commands)
        
        # Get updated scene data for rendering
        game_progress_data = game.get_scene_progress_data()
        game_view.draw(game_progress_data)
        
        frame_count += 1
        game_frames += 1
        
        # If game is over or needs reset, reset both collector and game
        if result == "RESET":
            print(f"[GAME {games_completed + 1}] Completed after {game_frames - 1} frames (logged {min(game_frames - 1, debug_frame_limit)} frames)")
            print()
            collector_1p.reset()
            game.reset()
            game_view.reset()
            games_completed += 1
            game_frames = 0
    
    pygame.quit()
    print("\n" + "=" * 70)
    print("Data collection ended.")
    print(f"Total games completed: {games_completed}")
    print(f"Total frames logged: {total_frames_logged}")
    print(f"Data saved to: {data_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
