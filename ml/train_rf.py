"""
訓練 1P / 2P 隨機森林模型的腳本。

預期輸入：
    - data/data_1p.csv
    - data/data_2p.csv

CSV 至少需要包含：
    - frame, status
    - ball_x, ball_y, ball_vx, ball_vy
    - platform_x
    - blocker_x, blocker_y (若非 HARD 模式可填 0)
    - action  (字串，對應 ML 指令，例如 MOVE_LEFT, MOVE_RIGHT, NONE, SERVE_TO_LEFT, SERVE_TO_RIGHT)

你可以依照專案需求調整欄位名稱，並在 FEATURE_COLUMNS 中同步修改。
"""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from rf_utils import (
    predict_landing_x_for_1p,
    predict_landing_x_for_2p,
    PLAYFIELD_WIDTH,
    PLAYFIELD_HEIGHT,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


FEATURE_COLUMNS_BASE = [
    "ball_x",
    "ball_y",
    "ball_vx",
    "ball_vy",
    "platform_x",
    "blocker_x",
    "blocker_y",
    "pred_landing_x",
]


def _prepare_dataframe(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """
    將原始資料轉成訓練用特徵：
        - 計算 pred_landing_x（依 side=1P/2P）
        - 僅保留 FEATURE_COLUMNS_BASE + action
    """
    # 若欄位名與假設不同，可在這裡做 rename
    rename_map = {}
    for col in ["ball_x", "ball_y", "ball_vx", "ball_vy", "platform_x", "blocker_x", "blocker_y", "action"]:
        if col not in df.columns:
            # 若不存在，嘗試由其他欄位推測或給預設值
            if col == "blocker_x" or col == "blocker_y":
                df[col] = 0
            else:
                # 對於關鍵欄位，直接報錯以提醒使用者調整資料
                raise KeyError(f"Column '{col}' not found in dataset. Please adjust your CSV or the training script.")

    # 計算預測落點
    if side == "1P":
        df["pred_landing_x"] = [
            predict_landing_x_for_1p(
                (bx, by),
                (vx, vy),
            )
            for bx, by, vx, vy in zip(df["ball_x"], df["ball_y"], df["ball_vx"], df["ball_vy"])
        ]
    else:
        df["pred_landing_x"] = [
            predict_landing_x_for_2p(
                (bx, by),
                (vx, vy),
            )
            for bx, by, vx, vy in zip(df["ball_x"], df["ball_y"], df["ball_vx"], df["ball_vy"])
        ]

    # clip 落點到場地範圍內，避免極端值
    df["pred_landing_x"] = df["pred_landing_x"].clip(0, PLAYFIELD_WIDTH)

    return df[FEATURE_COLUMNS_BASE + ["action"]].copy()


def train_for_side(side: str, csv_path: Path, model_output_path: Path):
    print(f"=== Training RF model for {side} ===")
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df_raw = pd.read_csv(csv_path)
    df = _prepare_dataframe(df_raw, side)

    X = df[FEATURE_COLUMNS_BASE].to_numpy(dtype=np.float32)
    y = df["action"].astype(str).to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=0,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print(f"=== {side} validation report ===")
    print(classification_report(y_val, y_pred))

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_output_path)
    print(f"Saved {side} model to: {model_output_path}")


def main():
    data_1p = DATA_DIR / "data_1p.csv"
    data_2p = DATA_DIR / "data_2p.csv"

    model_1p = MODEL_DIR / "rf_model_1p.pkl"
    model_2p = MODEL_DIR / "rf_model_2p.pkl"

    train_for_side("1P", data_1p, model_1p)
    train_for_side("2P", data_2p, model_2p)


if __name__ == "__main__":
    main()


