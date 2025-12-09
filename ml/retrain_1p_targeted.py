"""
Retrain 1P RandomForest using augmented + targeted datasets.

This script loads `ml/data/play_data_auto_aug.csv` and `ml/data/play_data_targeted.csv`,
concatenates them (with optional multiplier for targeted rows), trains RF, and saves model.

Usage:
  python -m ml.retrain_1p_targeted --multiplier 3 --n_estimators 300 --max_depth 16 --balance --class_weight
"""
import os
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
import joblib

ROOT = os.path.dirname(__file__)
DATA_AUG = os.path.join(ROOT, 'data', 'play_data_auto_aug.csv')
DATA_TARGET = os.path.join(ROOT, 'data', 'play_data_targeted.csv')
MODELS_DIR = os.path.join(ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

ACTION_ORDER = ['NONE', 'MOVE_LEFT', 'MOVE_RIGHT', 'SERVE_TO_LEFT', 'SERVE_TO_RIGHT']
ACTION_TO_INT = {a: i for i, a in enumerate(ACTION_ORDER)}

# Candidate features (keep intersection with available columns)
FEATURE_CANDIDATES = ['ball_x','ball_y','ball_vx','ball_vy','self_px','opp_px','ball_served','serving_is_self','blocker_x','pred_landing_x',
                      'prev1_ball_vx','prev1_ball_vy','prev1_self_px','prev2_ball_vx','prev2_ball_vy','prev2_self_px',
                      'landing_dx','time_to_land','ball_speed_abs']


def load_and_merge(multiplier=3):
    if not os.path.exists(DATA_AUG):
        raise RuntimeError(f'Augmented data not found: {DATA_AUG}')
    df_aug = pd.read_csv(DATA_AUG)
    df_aug = df_aug[df_aug['player'] == '1P']

    if not os.path.exists(DATA_TARGET):
        print('Targeted data not found; using augmented only')
        df_combined = df_aug
    else:
        df_t = pd.read_csv(DATA_TARGET)
        # targeted file may include both players; filter 1P
        df_t = df_t[df_t['player'] == '1P']
        if multiplier > 1:
            df_t = pd.concat([df_t] * multiplier, ignore_index=True)
        df_combined = pd.concat([df_aug, df_t], ignore_index=True)

    # choose features present
    features = [f for f in FEATURE_CANDIDATES if f in df_combined.columns]
    if not features:
        raise RuntimeError('No candidate features present in combined dataset')

    X = df_combined[features]
    y = df_combined['action'].map(ACTION_TO_INT)
    return X, y, features


def train_and_save(multiplier=3, n_estimators=100, max_depth=12, balance=False, class_weight=False):
    X, y, features = load_and_merge(multiplier=multiplier)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if balance:
        train_df = X_train.copy()
        train_df['y'] = y_train.values
        counts = train_df['y'].value_counts()
        max_count = counts.max()
        resampled = []
        for cls, cnt in counts.items():
            cls_df = train_df[train_df['y'] == cls]
            if cnt < max_count:
                cls_up = resample(cls_df, replace=True, n_samples=max_count, random_state=42)
            else:
                cls_up = cls_df
            resampled.append(cls_up)
        train_bal = pd.concat(resampled)
        y_train = train_bal['y']
        X_train = train_bal.drop(columns=['y'])

    cw = 'balanced' if class_weight else None
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=1, class_weight=cw)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print('---- 1P targeted RandomForest retrain ----')
    print('acc', accuracy_score(y_test, y_pred))
    present_labels = sorted(list(set(y_test.unique()) | set(y_train.unique())))
    target_names = [ACTION_ORDER[i] for i in present_labels]
    print(classification_report(y_test, y_pred, labels=present_labels, target_names=target_names))

    model_path = os.path.join(MODELS_DIR, 'rf_1p.joblib')
    joblib.dump({'clf': clf, 'action_to_int': ACTION_TO_INT, 'features': features}, model_path)
    print('Saved 1P model to', model_path)
    return model_path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--multiplier', type=int, default=3, help='Repeat targeted rows this many times before concatenation')
    p.add_argument('--n_estimators', type=int, default=100)
    p.add_argument('--max_depth', type=int, default=12)
    p.add_argument('--balance', action='store_true')
    p.add_argument('--class_weight', action='store_true')
    args = p.parse_args()

    train_and_save(multiplier=args.multiplier, n_estimators=args.n_estimators, max_depth=args.max_depth, balance=args.balance, class_weight=args.class_weight)
