"""
隨機森林 RF 版本的 1P AI。

使用訓練好的 RF 模型，根據當前場景資訊預測下一個動作。
"""
from pathlib import Path
import sys
import os
from typing import Tuple, List

import joblib
import numpy as np


# 確保可以匯入同一目錄下的 rf_utils
ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.append(str(ML_DIR))

#from rf_utils import build_feature_vector_1p, load_model
# 遊戲場地尺寸（去掉左右背景後的實際可玩區域）
PLAYFIELD_WIDTH = 200
PLAYFIELD_HEIGHT = 500

# 板子高度（來自 README）
PLATFORM_1P_Y = 420
PLATFORM_2P_Y = 70

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        """
        Constructor

        @param ai_name A string "1P" or "2P" indicates that the `MLPlay` is used by
               which side. 這個腳本預期被 1P 使用。
        """
        self.side = ai_name
        # 控制是否已經發球，避免重複 SERVE 指令
        self.ball_served = False

        # 模型預設路徑：專案根目錄下的 models/rf_model_1p.pkl
        root = Path(__file__).resolve().parents[1]
        model_path = root / "models" / "rf_model_1p.pkl"
        self.model = load_model(str(model_path))

    def update(self, scene_info, *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        status = scene_info.get("status")
        if status != "GAME_ALIVE":
            # 回合結束，重置內部狀態
            self.ball_served = False
            return "RESET"

        # 使用 RF 來決定發球方向，以減少推論開銷
        if not self.ball_served:
            features_vec = build_feature_vector_1p(scene_info)
            features = features_vec.reshape(1, -1)
            action = self.model.predict(features)[0]

            if action in ("SERVE_TO_LEFT", "SERVE_TO_RIGHT"):
                self.ball_served = True
                return action
            # 若模型沒叫發球，預設發向右
            self.ball_served = True
            return "SERVE_TO_RIGHT"

        features_vec = build_feature_vector_1p(scene_info)
        pred_x = float(features_vec[-1])

        platform_x, _ = scene_info["platform_1P"]
        platform_center = platform_x + 20  # 板子寬 40
        margin = 2.0

        if pred_x > platform_center + margin:
            return "MOVE_RIGHT"
        elif pred_x < platform_center - margin:
            return "MOVE_LEFT"
        else:
            return "NONE"

    def reset(self):
        """
        Reset the status
        """
        self.ball_served = False



def _simulate_landing_x(
    ball_x: float,
    ball_y: float,
    vx: float,
    vy: float,
    target_y: float,
    dt: float = 1.0,
) -> float:
    """
    粗略模擬球在僅與上下左右邊界反彈的情況下，於 target_y 高度的落點 x。

    說明：
    - 忽略板子與障礙物造成的切球效果，作為 RF 的輔助特徵。
    - 若 vy == 0，則無法預測與 target_y 交會，直接回傳當前 x。
    """
    x, y = float(ball_x), float(ball_y)
    vx, vy = float(vx), float(vy)

    if vy == 0:
        return x

    # 安全迴圈上限，避免極端情況無限迴圈
    for _ in range(2000):
        # 若已經跨過目標高度，則線性內插出交點
        if (vy > 0 and y >= target_y) or (vy < 0 and y <= target_y):
            if vy == 0:
                return x
            t = (target_y - y) / vy
            return x + vx * t

        # 單步更新
        x += vx * dt
        y += vy * dt

        # 左右牆反彈
        if x < 0:
            x = -x
            vx *= -1
        elif x > PLAYFIELD_WIDTH:
            x = 2 * PLAYFIELD_WIDTH - x
            vx *= -1

        # 上下邊界反彈
        if y < 0:
            y = -y
            vy *= -1
        elif y > PLAYFIELD_HEIGHT:
            y = 2 * PLAYFIELD_HEIGHT - y
            vy *= -1

    # 若超過迴圈仍未抵達，回傳目前 x 當作近似
    return x


def predict_landing_x_for_1p(ball_pos: Tuple[float, float], ball_speed: Tuple[float, float]) -> float:
    """
    預測球在 1P 板子高度 (y=420) 的落點 x。
    """
    return _simulate_landing_x(ball_pos[0], ball_pos[1], ball_speed[0], ball_speed[1], PLATFORM_1P_Y)


def predict_landing_x_for_2p(ball_pos: Tuple[float, float], ball_speed: Tuple[float, float]) -> float:
    """
    預測球在 2P 板子高度 (y=70) 的落點 x。
    """
    return _simulate_landing_x(ball_pos[0], ball_pos[1], ball_speed[0], ball_speed[1], PLATFORM_2P_Y)


def build_feature_vector_1p(scene_info: dict) -> np.ndarray:
    """
    根據 scene_info 建立 1P 使用的特徵向量。
    """
    ball_x, ball_y = scene_info["ball"]
    ball_vx, ball_vy = scene_info["ball_speed"]
    platform_x, _ = scene_info["platform_1P"]
    blocker_x, blocker_y = scene_info.get("blocker", (0, 0)) or (0, 0)

    pred_x = predict_landing_x_for_1p((ball_x, ball_y), (ball_vx, ball_vy))

    features: List[float] = [
        ball_x,
        ball_y,
        ball_vx,
        ball_vy,
        platform_x,
        blocker_x,
        blocker_y,
        pred_x,
    ]
    return np.asarray(features, dtype=np.float32)


def build_feature_vector_2p(scene_info: dict) -> np.ndarray:
    """
    根據 scene_info 建立 2P 使用的特徵向量。
    """
    ball_x, ball_y = scene_info["ball"]
    ball_vx, ball_vy = scene_info["ball_speed"]
    platform_x, _ = scene_info["platform_2P"]
    blocker_x, blocker_y = scene_info.get("blocker", (0, 0)) or (0, 0)

    pred_x = predict_landing_x_for_2p((ball_x, ball_y), (ball_vx, ball_vy))

    features: List[float] = [
        ball_x,
        ball_y,
        ball_vx,
        ball_vy,
        platform_x,
        blocker_x,
        blocker_y,
        pred_x,
    ]
    return np.asarray(features, dtype=np.float32)


def load_model(model_path: str):
    """
    載入 RF 模型的輔助函式，會檢查檔案是否存在。
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"RF model file not found: {model_path}")
    return joblib.load(model_path)





