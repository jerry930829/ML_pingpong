"""
Run the game with decision-tree agents for 1P and 2P and open a pygame window for observation.
"""
import os
import sys
import pygame

# Ensure repo root on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    # prefer RandomForest agents if available
    from ml.ml_play_rf_1p import MLPlay as MLPlay1
    from ml.ml_play_rf_2p import MLPlay as MLPlay2
except Exception:
    # fallback to decision-tree agents
    from ml.ml_play_dt_1p import MLPlay as MLPlay1
    from ml.ml_play_dt_2p import MLPlay as MLPlay2
from src.game import PingPong
from mlgame.view.view import PygameView
from mlgame.game.generic import quit_or_esc
from mlgame.utils.enum import get_ai_name

FPS = 60


def run():
    pygame.init()
    game = PingPong(difficulty='HARD', game_over_score=8,init_vel=7)
    scene_init_info_dict = game.get_scene_init_data()
    game_view = PygameView(scene_init_info_dict)

    # instantiate agents
    agent1 = MLPlay1('1P')
    agent2 = MLPlay2('2P')

    frame_count = 0
    clock = pygame.time.Clock()

    try:
        while game.is_running and not quit_or_esc():
            clock.tick(FPS)
            # get scene
            scene_dict = game.get_data_from_game_to_player()
            ai1 = get_ai_name(0)
            ai2 = get_ai_name(1)
            scene = scene_dict[ai1]

            # agents decide
            cmd1 = agent1.update(scene)
            cmd2 = agent2.update(scene)
            commands = {ai1: cmd1, ai2: cmd2}

            result = game.update(commands)
            game_progress_data = game.get_scene_progress_data()
            game_view.draw(game_progress_data)
            frame_count += 1

            if result == 'RESET':
                game.reset()
                game_view.reset()
    finally:
        pygame.quit()


if __name__ == '__main__':
    run()
