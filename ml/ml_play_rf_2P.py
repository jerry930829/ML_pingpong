"""
隨機森林 RF 版本的 2P AI。

使用訓練好的 RF 模型，根據當前場景資訊預測下一個動作。
"""
from pathlib import Path
import sys

# 確保可以匯入同一目錄下的 rf_utils
ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.append(str(ML_DIR))

from rf_utils import build_feature_vector_2p, load_model


class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        """
        Constructor

        @param ai_name A string "1P" or "2P" indicates that the `MLPlay` is used by
               which side. 這個腳本預期被 2P 使用。
        """
        self.side = ai_name
        self.ball_served = False

        root = Path(__file__).resolve().parents[1]
        model_path = root / "models" / "rf_model_2p.pkl"
        self.model = load_model(str(model_path))

    def update(self, scene_info, *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        status = scene_info.get("status")
        if status != "GAME_ALIVE":
            self.ball_served = False
            return "RESET"

        # 尚未發球時，才使用 RF 來決定發球方向
        if not self.ball_served:
            features_vec = build_feature_vector_2p(scene_info)
            features = features_vec.reshape(1, -1)
            action = self.model.predict(features)[0]

            if action in ("SERVE_TO_LEFT", "SERVE_TO_RIGHT"):
                self.ball_served = True
                return action
            # 預設 2P 往左發球以示區分
            self.ball_served = True
            return "SERVE_TO_LEFT"

        # 球在場上時，完全依照物理落點預測移動板子（與 recorder 策略一致）
        features_vec = build_feature_vector_2p(scene_info)
        pred_x = float(features_vec[-1])

        platform_x, _ = scene_info["platform_2P"]
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


