"""
Analyze failed matches saved by `ml/diagnose_1p.py`.

Outputs:
- ml/data/failure_patterns_per_match.csv
- ml/data/failure_patterns_summary.json

Usage: python -m ml.analyze_failures
"""
import os
import glob
import json
import math
from collections import Counter
import pandas as pd

ROOT = os.path.dirname(__file__)
FAILED_DIR = os.path.join(ROOT, 'data', 'failed_matches')
OUT_CSV = os.path.join(ROOT, 'data', 'failure_patterns_per_match.csv')
OUT_SUM = os.path.join(ROOT, 'data', 'failure_patterns_summary.json')

from ml.landing import simulate_landing_x


def load_failed_matches():
    files = sorted(glob.glob(os.path.join(FAILED_DIR, 'match_*.jsonl')))
    matches = []
    for f in files:
        frames = []
        with open(f, 'r') as fh:
            for line in fh:
                frames.append(json.loads(line))
        matches.append({'file': os.path.basename(f), 'frames': frames})
    return matches


def analyze_match(frames, last_n=30, fps=60):
    L = len(frames)
    start = max(0, L - last_n)
    tail = frames[start:]

    final = tail[-1]
    ball_x, ball_y = final['ball']
    ball_vx, ball_vy = final['ball_speed']
    self_px = final['platform_1P'][0]
    opp_px = final['platform_2P'][0]

    # estimate platform velocity from earlier frame if available
    p1_vx = 0
    if L >= 2:
        prev = frames[-2]
        p1_vx = final['platform_1P'][0] - prev['platform_1P'][0]

    # build minimal scene for landing sim
    scene = {
        'frame': final['frame'],
        'ball': final['ball'],
        'ball_speed': final['ball_speed'],
        'ball_served': True,
        'serving_side': '1P',
        'platform_1P': final['platform_1P'],
        'platform_2P': final['platform_2P'],
        'blocker': final.get('blocker', (0,0))
    }

    try:
        res = simulate_landing_x(scene, p1_vx=int(round(p1_vx)), return_steps=True)
        if isinstance(res, tuple):
            landing_x, steps = res
            time_to_land = steps / float(fps)
        else:
            landing_x = float(res)
            time_to_land = None
    except Exception:
        landing_x = float(ball_x)
        time_to_land = None

    landing_dx = landing_x - self_px
    ball_speed_abs = math.hypot(ball_vx, ball_vy)

    # agent1 command sequence in tail
    cmds = [f.get('agent1_cmd') for f in tail]
    final_cmd = cmds[-1]

    # detect if agent moved in last 5 frames (i.e., issued MOVE_LEFT or MOVE_RIGHT)
    moved_last5 = any(c in ('MOVE_LEFT', 'MOVE_RIGHT') for c in cmds[-5:])

    # detect oscillation (direction flips) in last 8 frames
    moves = [c for c in cmds if c in ('MOVE_LEFT', 'MOVE_RIGHT')]
    flips = 0
    for i in range(1, len(moves)):
        if moves[i] != moves[i-1]:
            flips += 1

    return {
        'frames': L,
        'landing_dx': landing_dx,
        'time_to_land': time_to_land,
        'ball_speed': ball_speed_abs,
        'final_cmd': final_cmd,
        'moved_last5': moved_last5,
        'move_flips': flips,
        'cmd_seq_tail': cmds
    }


def aggregate(matches_analysis):
    # bins for landing_dx
    bins = ['<-50','-50~-20','-20~-5','-5~5','5~20','20~50','>50']
    def bin_dx(dx):
        if dx < -50: return '<-50'
        if dx < -20: return '-50~-20'
        if dx < -5: return '-20~-5'
        if dx <= 5: return '-5~5'
        if dx <= 20: return '5~20'
        if dx <= 50: return '20~50'
        return '>50'

    dx_counts = Counter()
    speed_counts = Counter()
    final_cmd_counts = Counter()
    moved_last5_count = 0
    flips_vals = []
    total = len(matches_analysis)

    for m in matches_analysis:
        dx = m['landing_dx']
        dx_counts[bin_dx(dx)] += 1
        spb = int(m['ball_speed'] // 5 * 5)
        speed_counts[str(spb)] += 1
        final_cmd_counts[m['final_cmd']] += 1
        if m['moved_last5']:
            moved_last5_count += 1
        flips_vals.append(m['move_flips'])

    summary = {
        'total_failed_matches': total,
        'landing_dx_bins': dict(dx_counts),
        'ball_speed_buckets': dict(speed_counts),
        'final_cmd_counts': dict(final_cmd_counts),
        'moved_last5_fraction': moved_last5_count / total if total else 0,
        'avg_move_flips': sum(flips_vals)/len(flips_vals) if flips_vals else 0
    }
    return summary


def main():
    matches = load_failed_matches()
    analyses = []
    for m in matches:
        a = analyze_match(m['frames'])
        row = {'match_file': m['file']}
        row.update(a)
        analyses.append(row)

    df = pd.DataFrame(analyses)
    if not df.empty:
        df.to_csv(OUT_CSV, index=False)
        summary = aggregate(analyses)
        with open(OUT_SUM, 'w') as fh:
            json.dump(summary, fh, indent=2)
        print('Wrote per-match CSV to', OUT_CSV)
        print('Wrote summary JSON to', OUT_SUM)
    else:
        print('No failed matches found in', FAILED_DIR)


if __name__ == '__main__':
    main()
