"""
規則型（非亂數）自動對戰 + 資料記錄的 MLPlay。

用途：
- 以簡單的物理落點預測規則操控 1P / 2P，產生比亂數更合理的訓練資料。
- 每個 frame 記錄一筆資料到 CSV，供 Random Forest 訓練使用。

輸出檔案：
- data/data_1p.csv
- data/data_2p.csv

執行範例：
    python -m mlgame -f 60 -i ./ml/ml_play_rule_recorder.py -i ./ml/ml_play_rule_recorder.py ./ --difficulty HARD --game_over_score 8 --init_vel 7
"""
import csv
from pathlib import Path
import sys

# 確保能從同一資料夾匯入 rf_utils
ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.append(str(ML_DIR))

from rf_utils import (
    predict_landing_x_for_1p,
    predict_landing_x_for_2p,
)


class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        """
        Constructor

        @param ai_name A string "1P" or "2P" indicates that the `MLPlay` is used by
               which side.
        """
        self.side = ai_name
        self.ball_served = False

        # 資料輸出路徑：data/data_1p.csv 或 data/data_2p.csv
        root = Path(__file__).resolve().parents[1]
        data_dir = root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        if self.side == "1P":
            self.csv_path = data_dir / "data_1p.csv"
        else:
            self.csv_path = data_dir / "data_2p.csv"

        self._ensure_header()

    def _ensure_header(self):
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "frame",
                        "side",
                        "ball_x",
                        "ball_y",
                        "ball_vx",
                        "ball_vy",
                        "platform_x",
                        "blocker_x",
                        "blocker_y",
                        "action",
                    ]
                )

    def _log_row(self, scene_info: dict, action: str):
        ball_x, ball_y = scene_info["ball"]
        ball_vx, ball_vy = scene_info["ball_speed"]

        if self.side == "1P":
            platform_x, _ = scene_info["platform_1P"]
        else:
            platform_x, _ = scene_info["platform_2P"]

        blocker = scene_info.get("blocker", (0, 0)) or (0, 0)
        blocker_x, blocker_y = blocker

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    scene_info.get("frame", -1),
                    self.side,
                    ball_x,
                    ball_y,
                    ball_vx,
                    ball_vy,
                    platform_x,
                    blocker_x,
                    blocker_y,
                    action,
                ]
            )

    def _rule_based_action(self, scene_info: dict) -> str:
        """
        使用落點預測的簡單規則：
        - 尚未發球：1P 固定向右發球，2P 固定向左發球。
        - 球在場上：根據預測落點 pred_x 追球。
        """
        if not scene_info.get("ball_served", False):
            if self.side == "1P":
                self.ball_served = True
                return "SERVE_TO_RIGHT"
            else:
                self.ball_served = True
                return "SERVE_TO_LEFT"

        ball_pos = tuple(scene_info["ball"])
        ball_speed = tuple(scene_info["ball_speed"])

        if self.side == "1P":
            pred_x = predict_landing_x_for_1p(ball_pos, ball_speed)
            platform_x, _ = scene_info["platform_1P"]
        else:
            pred_x = predict_landing_x_for_2p(ball_pos, ball_speed)
            platform_x, _ = scene_info["platform_2P"]

        platform_center = platform_x + 20  # 板子寬 40
        margin = 2.0

        if pred_x > platform_center + margin:
            return "MOVE_RIGHT"
        elif pred_x < platform_center - margin:
            return "MOVE_LEFT"
        else:
            return "NONE"

    def update(self, scene_info, *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        if scene_info["status"] != "GAME_ALIVE":
            self.ball_served = False
            return "RESET"

        action = self._rule_based_action(scene_info)
        self._log_row(scene_info, action)
        return action

    def reset(self):
        """
        Reset the status
        """
        self.ball_served = False





