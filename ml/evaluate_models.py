"""
Evaluate trained dt models by running several matches using ml/ml_play_dt_1p.py and ml/ml_play_dt_2p.py
"""
import os
import sys
# Ensure repo root on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.game import PingPong
#from ml.ml_play_dt_1p import MLPlay as MLPlay1
#from ml.ml_play_dt_2p import MLPlay as MLPlay2
from mlgame.utils.enum import get_ai_name

NUM_MATCHES = 10

wins = { '1P': 0, '2P': 0, 'draw': 0 }

for m in range(NUM_MATCHES):
    game = PingPong(difficulty='HARD', game_over_score=1)
    import os
    import sys
    import csv
    import argparse
    # Ensure repo root on path
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    from src.game import PingPong
    from mlgame.utils.enum import get_ai_name

    # Prefer RandomForest agents if available, otherwise fallback to DT agents
    try:
        from ml.ml_play_rf_1p import MLPlay as MLPlay1
        from ml.ml_play_rf_2p import MLPlay as MLPlay2
    except Exception:
        from ml.ml_play_dt_1p import MLPlay as MLPlay1
        from ml.ml_play_dt_2p import MLPlay as MLPlay2


    def evaluate(num_matches=10, difficulty='HARD', game_over_score=1, out_csv=None):
        wins = {'1P': 0, '2P': 0, 'draw': 0}

        rows = []

        for m in range(num_matches):
            game = PingPong(difficulty=difficulty, game_over_score=game_over_score)
            # instantiate agents
            agent1 = MLPlay1('1P')
            agent2 = MLPlay2('2P')

            # Safety: ensure classifiers used for inference do not use heavy parallelism
            for a in (agent1, agent2):
                try:
                    clf = getattr(a, 'clf', None)
                    if clf is not None and hasattr(clf, 'n_jobs'):
                        clf.n_jobs = 1
                except Exception:
                    pass

            # safeguard: limit frames per match to avoid infinite runs
            max_frames = getattr(game, '_game_over_score', None)
            # default cap (frames) if not provided externally
            MATCH_FRAME_CAP = 20000
            while game.is_running:
                scene_dict = game.get_data_from_game_to_player()
                ai1 = get_ai_name(0)
                ai2 = get_ai_name(1)
                scene = scene_dict[ai1]

                cmd1 = agent1.update(scene)
                cmd2 = agent2.update(scene)
                commands = {ai1: cmd1, ai2: cmd2}

                result = game.update(commands)

                # safety: force end if too many frames
                if getattr(game, '_frame_count', 0) > MATCH_FRAME_CAP:
                    print(f'Match {m+1}: frame cap reached, treating as draw')
                    # treat as draw
                    score1, score2 = game._score[0], game._score[1]
                    winner = 'draw'
                    frames = getattr(game, '_frame_count', None)
                    ball_speed_val = getattr(getattr(game, '_ball', None), 'speed', None)
                    rows.append({
                        'match': m + 1,
                        'winner': winner,
                        'score_1p': score1,
                        'score_2p': score2,
                        'frames': frames,
                        'ball_speed': ball_speed_val,
                    })
                    wins['draw'] += 1
                    break

                if result in ('RESET', 'QUIT'):
                    # match ended; read last scores from game
                    score1, score2 = game._score[0], game._score[1]
                    if score1 > score2:
                        wins['1P'] += 1
                        winner = '1P'
                    elif score1 < score2:
                        wins['2P'] += 1
                        winner = '2P'
                    else:
                        wins['draw'] += 1
                        winner = 'draw'

                    # capture some match stats
                    frames = getattr(game, '_frame_count', None)
                    ball_speed = getattr(game, '_ball', None)
                    ball_speed_val = ball_speed.speed if ball_speed is not None else None

                    print(f'Match {m+1}: {winner} wins (score {score1}-{score2})')

                    rows.append({
                        'match': m + 1,
                        'winner': winner,
                        'score_1p': score1,
                        'score_2p': score2,
                        'frames': frames,
                        'ball_speed': ball_speed_val,
                    })

                    break

        print('Evaluation finished. wins:', wins)

        if out_csv:
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            with open(out_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['match', 'winner', 'score_1p', 'score_2p', 'frames', 'ball_speed'])
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            print('Saved evaluation CSV to', out_csv)


    if __name__ == '__main__':
        p = argparse.ArgumentParser()
        p.add_argument('--matches', type=int, default=10)
        p.add_argument('--difficulty', type=str, default='HARD')
        p.add_argument('--game_over_score', type=int, default=1)
        p.add_argument('--out', type=str, default=None, help='Path to output CSV for per-match results')
        args = p.parse_args()

        evaluate(num_matches=args.matches, difficulty=args.difficulty, game_over_score=args.game_over_score, out_csv=args.out)
