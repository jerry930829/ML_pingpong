"""
Train decision tree models for 1P and 2P from collected CSV data.
Saves models to ml/models/dt_1p.joblib and dt_2p.joblib
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.utils import resample
import argparse

DATA_CSV = os.path.join(os.path.dirname(__file__), 'data', 'play_data_auto.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

ACTION_ORDER = ['NONE', 'MOVE_LEFT', 'MOVE_RIGHT', 'SERVE_TO_LEFT', 'SERVE_TO_RIGHT']
ACTION_TO_INT = {a: i for i, a in enumerate(ACTION_ORDER)}

# Feature order used for training — save with model so agents can construct DataFrame consistently
# Base features
BASE_FEATURES = ['ball_x','ball_y','ball_vx','ball_vy','self_px','opp_px','ball_served','serving_is_self','blocker_x','pred_landing_x']
# Add previous-frame features (prev1, prev2)
HISTORY_FEATURES = ['prev1_ball_vx','prev1_ball_vy','prev1_self_px',
                    'prev2_ball_vx','prev2_ball_vy','prev2_self_px']

# Combined target features. Training will pick whichever subset exists in CSV (for backward compatibility).
FEATURES = BASE_FEATURES + HISTORY_FEATURES


def _load_and_split(player_side='1P'):
    df = pd.read_csv(DATA_CSV)
    df = df[df['player'] == player_side]
    if df.empty:
        raise RuntimeError(f'No data for {player_side} in {DATA_CSV}')

    available_features = [f for f in FEATURES if f in df.columns]
    if not set(BASE_FEATURES).issubset(set(available_features)):
        raise RuntimeError(f'Missing required base features in {DATA_CSV}')

    X = df[available_features]
    y = df['action'].map(ACTION_TO_INT)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_for(player_side='1P', balance=False, class_weight=False):
    X_train, X_test, y_train, y_test = _load_and_split(player_side)

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
    clf = DecisionTreeClassifier(max_depth=8, random_state=42, class_weight=cw)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f'---- {player_side} model ----')
    print('acc', accuracy_score(y_test, y_pred))
    present_labels = sorted(list(set(y_test.unique()) | set(y_train.unique())))
    target_names = [ACTION_ORDER[i] for i in present_labels]
    print(classification_report(y_test, y_pred, labels=present_labels, target_names=target_names))

    model_path = os.path.join(MODELS_DIR, f'dt_{player_side.lower()}.joblib')
    joblib.dump({'clf': clf, 'action_to_int': ACTION_TO_INT, 'features': list(X_train.columns)}, model_path)
    print('Saved model to', model_path)

    return model_path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--balance', action='store_true', help='Apply simple oversampling to balance classes')
    p.add_argument('--class_weight', action='store_true', help="Use class_weight='balanced' in classifier")
    args = p.parse_args()

    train_for('1P', balance=args.balance, class_weight=args.class_weight)
    train_for('2P', balance=args.balance, class_weight=args.class_weight)
