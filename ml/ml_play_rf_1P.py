"""
隨機森林 RF 版本的 1P AI。

使用訓練好的 RF 模型，根據當前場景資訊預測下一個動作。
"""
from pathlib import Path
import sys

# 確保可以匯入同一目錄下的 rf_utils
ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.append(str(ML_DIR))

from rf_utils import build_feature_vector_1p, load_model


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


