"""
Run multiple headless evaluation matches between trained 1P and 2P models.
Usage:
    python run_eval_matches.py --matches 20 --difficulty HARD --game_over_score 3 --init_vel 7

This script runs matches without opening the Pygame window (no rendering) and
prints a summary of wins and scores.
"""
import argparse
import sys
import os
import pygame
from src.game import PingPong
from ml.ml_play_dt_1P import MLPlay as AI_1P
from ml.ml_play_dt_2P import MLPlay as AI_2P
from mlgame.utils.enum import get_ai_name
from mlgame.game.generic import quit_or_esc


def build_scene_info(game, frame_count):
    blocker = None
    if hasattr(game, '_blocker') and game._blocker.rect.x < 1000:
        blocker = [game._blocker.rect.x, game._blocker.rect.y]

    status_obj = game.get_game_status()
    status_name = status_obj.name if hasattr(status_obj, 'name') else str(status_obj)

    return {
        'frame': frame_count,
        'status': status_name,
        'ball': [game._ball.rect.x, game._ball.rect.y],
        'ball_speed': list(game._ball.speed),
        'ball_served': game._ball_served,
        'serving_side': '1P' if frame_count % 2 == 0 else '2P',
        'platform_1P': [game._platform_1P.rect.x, game._platform_1P.rect.y],
        'platform_2P': [game._platform_2P.rect.x, game._platform_2P.rect.y],
        'blocker': blocker,
    }


def run_one_match(difficulty, game_over_score, init_vel, fps=60):
    pygame.init()
    game = PingPong(difficulty=difficulty, game_over_score=game_over_score, init_vel=init_vel)

    ai1 = AI_1P(ai_name='1P')
    ai2 = AI_2P(ai_name='2P')

    clock = pygame.time.Clock()
    frame = 0
    # run until RESET occurs (one full game)
    while game.is_running and not quit_or_esc():
        clock.tick_busy_loop(fps)
        scene = build_scene_info(game, frame)
        cmd1 = ai1.update(scene)
        cmd2 = ai2.update(scene)
        commands = {get_ai_name(0): cmd1, get_ai_name(1): cmd2}
        res = game.update(commands)
        frame += 1
        if res == 'RESET':
            try:
                score1 = game._score[0]
                score2 = game._score[1]
            except Exception:
                score1 = None
                score2 = None
            ai1.reset()
            ai2.reset()
            game.reset()
            pygame.quit()
            return score1, score2
    pygame.quit()
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matches', type=int, default=20)
    parser.add_argument('--difficulty', default='HARD', choices=['NORMAL', 'HARD'])
    parser.add_argument('--game_over_score', type=int, default=3)
    parser.add_argument('--init_vel', type=int, default=7)
    args = parser.parse_args()

    wins = {'1P': 0, '2P': 0, 'draw': 0}
    scores = []

    for i in range(args.matches):
        print(f'Running match {i+1}/{args.matches}...')
        s1, s2 = run_one_match(args.difficulty, args.game_over_score, args.init_vel)
        print(f'  Result: 1P={s1} 2P={s2}')
        scores.append((s1, s2))
        if s1 is None or s2 is None:
            print('  Warning: could not read scores for this match')
            continue
        if s1 > s2:
            wins['1P'] += 1
        elif s2 > s1:
            wins['2P'] += 1
        else:
            wins['draw'] += 1

    print('\nSummary:')
    print(f"  Matches run: {args.matches}")
    print(f"  1P wins: {wins['1P']}")
    print(f"  2P wins: {wins['2P']}")
    print(f"  Draws : {wins['draw']}")
    print('  Scores:')
    for idx, (a, b) in enumerate(scores, 1):
        print(f'    {idx}: 1P={a} 2P={b}')


if __name__ == '__main__':
    main()
