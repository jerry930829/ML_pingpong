"""
離線（無 GUI）規則型資料產生腳本。

說明：
- 直接在程式中建立 PingPong 遊戲實例，不透過 MLGame / pygame 視窗。
- 使用「物理解算落點」的簡單規則操控 1P / 2P，比亂數更合理，適合作為 RF 訓練資料。

輸出檔案：
- data/data_1p.csv
- data/data_2p.csv

使用方式（在專案根目錄）：
    python -m ml.generate_random_data_offline

可在程式內調整：
- NUM_EPISODES        ：總模擬回合數
- MAX_FRAMES_PER_GAME ：單局最多 frame 數（防止無限遊戲）
"""
import csv
from pathlib import Path
import sys

from mlgame.utils.enum import get_ai_name

from src.game import PingPong

# 匯入 rf_utils 的落點預測
ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.append(str(ML_DIR))

from rf_utils import predict_landing_x_for_1p, predict_landing_x_for_2p  # noqa: E402

# ===== 可調整參數 =====
NUM_EPISODES = 200  # 要模擬的遊戲回合數
MAX_FRAMES_PER_GAME = 3000  # 單局最多 frame 數，避免極端情況


def _ensure_header(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
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


def _log_row(path: Path, side: str, scene_info: dict, action: str):
    ball_x, ball_y = scene_info["ball"]
    ball_vx, ball_vy = scene_info["ball_speed"]

    if side == "1P":
        platform_x, _ = scene_info["platform_1P"]
    else:
        platform_x, _ = scene_info["platform_2P"]

    blocker = scene_info.get("blocker", (0, 0)) or (0, 0)
    blocker_x, blocker_y = blocker

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                scene_info.get("frame", -1),
                side,
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

def _rule_based_action(side: str, scene_info: dict) -> str:
    """
    與 ml_play_rule_recorder 相同邏輯的規則型策略：
    - 尚未發球：1P 固定向右發球，2P 固定向左發球。
    - 球在場上：根據預測落點 pred_x 追球。
    """
    if not scene_info.get("ball_served", False):
        if side == "1P":
            return "SERVE_TO_RIGHT"
        else:
            return "SERVE_TO_LEFT"

    ball_pos = tuple(scene_info["ball"])
    ball_speed = tuple(scene_info["ball_speed"])

    if side == "1P":
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


def run_offline_random_generation():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_1p = data_dir / "data_1p.csv"
    csv_2p = data_dir / "data_2p.csv"
    _ensure_header(csv_1p)
    _ensure_header(csv_2p)

    # 建立遊戲（與你指定的設定一致：HARD, init_vel=7, user_num=2）
    game = PingPong(difficulty="HARD", game_over_score=8, user_num=2, init_vel=7)

    ai_1p_name = get_ai_name(0)
    ai_2p_name = get_ai_name(1)

    episode = 0
    while episode < NUM_EPISODES:
        frames_in_game = 0

        # 每一局從當前狀態開始，直到 RESET / QUIT / frame 超出上限
        while game.is_running and frames_in_game < MAX_FRAMES_PER_GAME:
            # 取得給 AI 的場景資訊（兩邊相同內容）
            data_to_players = game.get_data_from_game_to_player()
            scene_info = data_to_players[ai_1p_name]

            # 規則型決策雙方動作
            action_1p = _rule_based_action("1P", scene_info)
            action_2p = _rule_based_action("2P", scene_info)

            # 紀錄資料
            _log_row(csv_1p, "1P", scene_info, action_1p)
            _log_row(csv_2p, "2P", scene_info, action_2p)

            # 組合成遊戲需要的指令格式
            commands = {
                ai_1p_name: action_1p,
                ai_2p_name: action_2p,
            }

            result = game.update(commands)
            frames_in_game += 1

            if result == "RESET":
                game.reset()
                break
            if result == "QUIT":
                break

        episode += 1
        print(f"Episode {episode}/{NUM_EPISODES} finished, frames={frames_in_game}")

        if not game.is_running:
            break

    print("Offline random data generation finished.")


if __name__ == "__main__":
    run_offline_random_generation()
