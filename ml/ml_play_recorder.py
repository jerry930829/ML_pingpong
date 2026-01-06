"""
使用鍵盤操作進行遊戲，同時將資料記錄成 CSV，提供 RF 訓練用。

說明：
- 這個 MLPlay 腳本可同時給 1P 與 2P 使用（MLGame 會分別以 ai_name="1P"/"2P" 啟動）。
- 每一個 frame 會記錄：
    frame, side, ball_x, ball_y, ball_vx, ball_vy, platform_x, blocker_x, blocker_y, action
- 1P 鍵位：上/下發球，左右移動
- 2P 鍵位：Q/E 發球，A/D 移動

執行範例（資料收集）：
    python -m mlgame -f 60 -i ./ml/ml_play_recorder.py -i ./ml/ml_play_recorder.py ./ --difficulty HARD --game_over_score 3 --init_vel 7
"""
import csv
import os
from pathlib import Path

import pygame


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

        # 準備 CSV 標頭（若檔案不存在或為空檔）
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

    def update(self, scene_info, keyboard=[], *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        if scene_info["status"] != "GAME_ALIVE":
            # 回合結束不記錄動作，只重置狀態
            self.ball_served = False
            return "RESET"

        command = "NONE"

        if self.side == "1P":
            # Red 紅色 下方
            if pygame.K_UP in keyboard:
                command = "SERVE_TO_LEFT"
                self.ball_served = True
            elif pygame.K_DOWN in keyboard:
                command = "SERVE_TO_RIGHT"
                self.ball_served = True
            elif pygame.K_LEFT in keyboard:
                command = "MOVE_LEFT"
            elif pygame.K_RIGHT in keyboard:
                command = "MOVE_RIGHT"

        elif self.side == "2P":
            # Blue 藍色 上方
            if pygame.K_q in keyboard:
                command = "SERVE_TO_LEFT"
                self.ball_served = True
            elif pygame.K_e in keyboard:
                command = "SERVE_TO_RIGHT"
                self.ball_served = True
            elif pygame.K_a in keyboard:
                command = "MOVE_LEFT"
            elif pygame.K_d in keyboard:
                command = "MOVE_RIGHT"

        # 記錄當前 frame 的資料與動作
        self._log_row(scene_info, command)

        return command

    def reset(self):
        """
        Reset the status
        """
        print("reset " + self.side)
        self.ball_served = False





