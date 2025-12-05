"""
Hybrid game runner: Player 1 controlled by Decision Tree AI, Player 2 controlled by human.

Usage:
    python run_hybrid_1P_ai_2P_human.py [--difficulty NORMAL|HARD] [--game_over_score N] [--init_vel N]

Example:
    python run_hybrid_1P_ai_2P_human.py --difficulty HARD --game_over_score 3 --init_vel 7

Keyboard Controls (2P / Player 2):
    - Q: Serve to left
    - E: Serve to right
    - A: Move left
    - D: Move right

1P (Player 1) is controlled by the Decision Tree model automatically.
"""
import sys
import argparse
import os
sys.path.insert(0, os.path.dirname(__file__))

import pygame
from mlgame.view.view import PygameView
from mlgame.game.generic import quit_or_esc
from mlgame.utils.enum import get_ai_name
from src.game import PingPong

# Import the 1P AI agent
from ml.ml_play_dt_1P import MLPlay as AI_1P

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
    # Normalize status to enum name when available (e.g., 'GAME_ALIVE')
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


def main():
    parser = argparse.ArgumentParser(description="Hybrid Pingpong: 1P AI vs 2P Human")
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
    
    # Initialize 1P AI agent
    ai_1p = AI_1P(ai_name="1P")
    
    scene_init_info_dict = game.get_scene_init_data()
    game_view = PygameView(scene_init_info_dict)
    
    frame_count = 0
    FPS = 30
    clock = pygame.time.Clock()
    
    debug_frame_limit = 100  # Only log first 100 frames
    
    print("=" * 60)
    print("Hybrid Pingpong: Player 1 (AI) vs Player 2 (Human)")
    print("=" * 60)
    print(f"Difficulty: {args.difficulty}")
    print(f"Game Over Score: {args.game_over_score}")
    print(f"Initial Ball Velocity: {args.init_vel}")
    print("\nPlayer 2 Controls:")
    print("  Q: Serve to left")
    print("  E: Serve to right")
    print("  A: Move left")
    print("  D: Move right")
    print("\nPlayer 1 is controlled by Decision Tree AI.")
    print("Press ESC or close window to quit.")
    print("=" * 60)
    
    # Wait for initial state - the first frame may have incomplete scene_info
    initial_wait = 0
    while game.is_running and not quit_or_esc() and initial_wait < 5:
        clock.tick_busy_loop(FPS)
        scene_info = build_scene_info(game, initial_wait)
        # Accept either 'GAME_ALIVE' or 'GameStatus.GAME_ALIVE' by checking substring
        if scene_info and "GAME_ALIVE" in str(scene_info.get("status")):
            break
        initial_wait += 1
    
    print(f"[HYBRID] Game started after {initial_wait} frames")
    
    while game.is_running and not quit_or_esc():
        clock.tick_busy_loop(FPS)
        
        # Build current game state from game objects
        scene_info = build_scene_info(game, frame_count)
        
        # Get 1P action from AI model
        command_1p = ai_1p.update(scene_info)
        
        # Get 2P action from keyboard
        command_2p = get_2p_action_from_keyboard()
        
        # Combine commands using get_ai_name convention for game.update()
        commands = {
            get_ai_name(0): command_1p,  # 0 = 1P
            get_ai_name(1): command_2p,  # 1 = 2P
        }
        
        # Only print debug frames while game is alive (accept different status formats)
        if frame_count < debug_frame_limit and "GAME_ALIVE" in str(scene_info.get("status")):
            print(f"[HYBRID] frame {scene_info.get('frame')} status={scene_info.get('status')} -> 1P: {command_1p}, 2P: {command_2p}")
        
        # Update game state
        result = game.update(commands)
        
        # Get updated scene data for rendering
        game_progress_data = game.get_scene_progress_data()
        game_view.draw(game_progress_data)
        
        frame_count += 1
        
        # If game is over or needs reset, reset both AI and game
        if result == "RESET":
            ai_1p.reset()
            game.reset()
            game_view.reset()
    
    pygame.quit()
    print("\nGame ended. Thanks for playing!")


if __name__ == "__main__":
    main()
