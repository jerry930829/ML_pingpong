"""
Retrain only 1P RandomForest using augmented CSV (expects `play_data_auto_aug.csv`).
Saves model to `ml/models/rf_1p.joblib`.

Usage: `python -m ml.retrain_1p --balance --class_weight`
"""
import os
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
import joblib

ROOT = os.path.dirname(__file__)
DATA_AUG = os.path.join(ROOT, 'data', 'play_data_auto_aug.csv')
MODELS_DIR = os.path.join(ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

ACTION_ORDER = ['NONE', 'MOVE_LEFT', 'MOVE_RIGHT', 'SERVE_TO_LEFT', 'SERVE_TO_RIGHT']
ACTION_TO_INT = {a: i for i, a in enumerate(ACTION_ORDER)}

# include our new derived features if present
FEATURES = ['ball_x','ball_y','ball_vx','ball_vy','self_px','opp_px','ball_served','serving_is_self','blocker_x','pred_landing_x',
            'prev1_ball_vx','prev1_ball_vy','prev1_self_px','prev2_ball_vx','prev2_ball_vy','prev2_self_px',
            'landing_dx','time_to_land','ball_speed_abs']


def load_data(path=DATA_AUG):
    df = pd.read_csv(path)
    df = df[df['player'] == '1P']
    if df.empty:
        raise RuntimeError('No 1P rows in augmented CSV')
    available = [f for f in FEATURES if f in df.columns]
    X = df[available]
    y = df['action'].map(ACTION_TO_INT)
    return X, y


def train(balance=False, class_weight=False, n_estimators=100, max_depth=12):
    X, y = load_data()
    # split
    from sklearn.model_selection import train_test_split
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
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1, class_weight=cw)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print('1P RandomForest retrain')
    print('acc', accuracy_score(y_test, y_pred))
    present_labels = sorted(list(set(y_test.unique()) | set(y_train.unique())))
    target_names = [ACTION_ORDER[i] for i in present_labels]
    print(classification_report(y_test, y_pred, labels=present_labels, target_names=target_names))

    model_path = os.path.join(MODELS_DIR, 'rf_1p.joblib')
    joblib.dump({'clf': clf, 'action_to_int': ACTION_TO_INT, 'features': list(X_train.columns)}, model_path)
    print('Saved 1P model to', model_path)
    return model_path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--balance', action='store_true')
    p.add_argument('--class_weight', action='store_true')
    p.add_argument('--n_estimators', type=int, default=100)
    p.add_argument('--max_depth', type=int, default=12)
    args = p.parse_args()
    train(balance=args.balance, class_weight=args.class_weight, n_estimators=args.n_estimators, max_depth=args.max_depth)
