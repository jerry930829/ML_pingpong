"""
Unified ML Pipeline for PingPong 1P & 2P training.

This script provides a simplified workflow to go from scratch to trained models.

Usage:
  python -m ml.train_pipeline --stage collect_auto    # Collect 20k base data
  python -m ml.train_pipeline --stage collect_targeted # Collect 10k hard samples
  python -m ml.train_pipeline --stage augment          # Add derived features
  python -m ml.train_pipeline --stage train_1p         # Train 1P model
  python -m ml.train_pipeline --stage train_2p         # Train 2P model
  python -m ml.train_pipeline --stage evaluate         # Evaluate both models
  python -m ml.train_pipeline --stage all              # Run all stages
"""
import os
import sys
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def stage_collect_auto(target_rows=20000):
    """Collect baseline data with heuristics (easy scenarios)."""
    print("\n" + "="*60)
    print("STAGE 1: Collect Baseline Data (20k rows)")
    print("="*60)
    from ml.collect_data_auto import collect as collect_auto
    collect_auto(target_rows=target_rows)
    print("✓ Baseline data saved to ml/data/play_data_auto.csv")


def stage_collect_targeted(target_rows=20000):
    """Collect hard scenarios (high speed, large landing error)."""
    print("\n" + "="*60)
    print("STAGE 2: Collect Targeted Hard Data (10k rows)")
    print("="*60)
    from ml.collect_targeted import collect as collect_targeted
    collect_targeted(
        target_rows=target_rows,
        difficulty='HARD',
        speed_thresh=12.0,
        landing_dx_thresh=50.0
    )
    print("✓ Targeted data saved to ml/data/play_data_targeted.csv")


def stage_augment():
    """Add derived features (landing_dx, ball_speed_abs, etc)."""
    print("\n" + "="*60)
    print("STAGE 3: Augment Features")
    print("="*60)
    from ml.augment_features import augment
    augment(
        input_csv='ml/data/play_data_auto.csv',
        output_csv='ml/data/play_data_auto_aug.csv'
    )
    print("✓ Augmented data saved to ml/data/play_data_auto_aug.csv")


def stage_train_1p(multiplier=3):
    """Train 1P RandomForest with augmented + targeted data (multiplier=3)."""
    print("\n" + "="*60)
    print("STAGE 4: Train 1P Model (RandomForest)")
    print("="*60)
    print("Config: n_estimators=100, max_depth=12, n_jobs=1")
    print(f"        balance=True, class_weight=balanced, multiplier={multiplier}")
    from ml.retrain_1p_targeted import train_and_save
    train_and_save(
        multiplier=multiplier,
        n_estimators=100,
        max_depth=12,
        balance=True,
        class_weight=True
    )
    print("✓ 1P model saved to ml/models/rf_1p.joblib")


def stage_train_2p():
    """Train 2P RandomForest with baseline data only."""
    print("\n" + "="*60)
    print("STAGE 5: Train 2P Model (RandomForest)")
    print("="*60)
    print("Config: n_estimators=100, max_depth=12, n_jobs=1")
    print("        balance=True, class_weight=balanced")
    from ml.train_random_forest import train_rf_2p
    train_rf_2p(
        input_csv='ml/data/play_data_auto_aug.csv',
        output_model='ml/models/rf_2p.joblib',
        n_estimators=100,
        max_depth=12,
        balance=True,
        class_weight=True
    )
    print("✓ 2P model saved to ml/models/rf_2p.joblib")


def stage_evaluate(num_matches=20):
    """Evaluate both 1P and 2P models over num_matches games."""
    print("\n" + "="*60)
    print(f"STAGE 6: Evaluate Models ({num_matches} matches)")
    print("="*60)
    from ml.evaluate_models import evaluate
    evaluate(num_matches=num_matches, difficulty='HARD', game_over_score=1)
    print(f"✓ Evaluation results saved to ml/data/eval_results_*.csv")


def stage_run_visual():
    """Run Pygame visual demo (selfplay)."""
    print("\n" + "="*60)
    print("STAGE 7: Visual Demo (Pygame)")
    print("="*60)
    print("Launching visual selfplay... (press ESC or close window to exit)")
    os.system('python -m ml.run_selfplay')


def main():
    parser = argparse.ArgumentParser(description='Unified ML Pipeline for PingPong')
    parser.add_argument(
        '--stage',
        type=str,
        default='all',
        choices=['collect_auto', 'collect_targeted', 'augment', 'train_1p', 'train_2p', 'evaluate', 'visual', 'all'],
        help='Which pipeline stage to run'
    )
    parser.add_argument('--auto_rows', type=int, default=20000, help='Baseline collection rows')
    parser.add_argument('--target_rows', type=int, default=10000, help='Targeted collection rows')
    parser.add_argument('--multiplier', type=int, default=3, help='1P training data multiplier')
    parser.add_argument('--eval_matches', type=int, default=20, help='Number of evaluation matches')
    
    args = parser.parse_args()
    stage = args.stage

    try:
        if stage == 'collect_auto' or stage == 'all':
            stage_collect_auto(target_rows=args.auto_rows)
        
        if stage == 'collect_targeted' or stage == 'all':
            stage_collect_targeted(target_rows=args.target_rows)
        
        if stage == 'augment' or stage == 'all':
            stage_augment()
        
        if stage == 'train_1p' or stage == 'all':
            stage_train_1p(multiplier=args.multiplier)
        
        if stage == 'train_2p' or stage == 'all':
            stage_train_2p()
        
        if stage == 'evaluate' or stage == 'all':
            stage_evaluate(num_matches=args.eval_matches)
        
        if stage == 'visual':
            stage_run_visual()

        print("\n" + "="*60)
        print("✓ Pipeline completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
