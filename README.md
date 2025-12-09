# PingPong ML 系統架構

## 高層流程圖

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML 訓練流程（train_pipeline.py）              │
└─────────────────────────────────────────────────────────────────┘
                           ↓
        ┌──────────────────┬──────────────────┐
        ↓                  ↓                  ↓
   ╔════════════╗   ╔════════════╗    ╔════════════╗
   ║ 數據收集    ║   ║ 特徵工程    ║    ║ 模型訓練    ║
   ╚════════════╝   ╚════════════╝    ╚════════════╝
   - collect_auto  - augment_features - train_rf_1p
   - collect_       (landing_dx,       - train_rf_2p
     targeted       ball_speed_abs)    
   - 30k 行數據    - 19 特徵         - 100 樹深度 12
                                       
        ↓                               ↓
   ┌──────────────────┐          ┌──────────────────┐
   │ ml/data/         │          │ ml/models/       │
   │ ├─ play_data_    │          │ ├─ rf_1p.joblib  │
   │ │  auto.csv      │          │ └─ rf_2p.joblib  │
   │ ├─ play_data_    │          └──────────────────┘
   │ │  targeted.csv  │                    ↓
   │ └─ play_data_    │          ╔════════════════════╗
   │    auto_aug.csv  │          ║ 推理（Runtime）    ║
   └──────────────────┘          ╚════════════════════╝
                                 - ml_play_rf_1p.py
                                 - ml_play_rf_2p.py
                                   (landing_dx=15px)
                                        ↓
                                 ┌──────────────────┐
                                 │ 遊戲引擎           │
                                 │ (src/game.py)    │
                                 └──────────────────┘
                                        ↓
                                 ╔════════════════════╗
                                 ║ 遊戲玩法           ║
                                 ║ (evaluate或visual) ║
                                 ╚════════════════════╝
```

---

## 模塊詳解

### 1. 遊戲引擎層 (`src/`)

| 文件 | 功能 |
|------|------|
| `game.py` | PingPong 遊戲主類<br>- 初始化球、平台<br>- update() 每幀更新物理 |
| `game_object.py` | Ball, Platform 類<br>- Ball: 碰撞、速度更新、切球效果<br>- Platform: 位置、輸入響應 |
| `utils.py` | 工具函數 |

**關鍵參數**: 
- 初始球速: `init_vel=7`
- 難度: EASY (無切球) / HARD (有切球)
- 得分目標: `game_over_score`

---

### 2. 著陸預測層 (`ml/landing.py`)

**目的**: 預測球會在何處著陸（沒有平台干預）

**流程**:
```python
get_predicted_landing(scene_info, p1_vx, p2_vx, return_steps=False)
├─ simulate_landing_x(ball, platform, max_steps=100)
│  ├─ 逐幀計算球軌跡
│  ├─ 檢測邊界碰撞 (rect.colliderect)
│  ├─ 檢測切球 (slice_ball 邏輯)
│  └─ 返回最終著陸 X 座標
└─ 快取結果（同幀重複查詢）
```

**優化**:
- `max_steps=100` (從 200 降低 → 速度 +50%)
- `n_jobs=1` (禁用多進程開銷)
- 幀級快取 (避免重複計算)

---

### 3. 數據收集層

#### 3.1 基礎收集 (`collect_data_auto.py`)
- **輸出**: `play_data_auto.csv` (20k 行)
- **方法**: 啟發式 agent vs 啟發式 agent
- **特點**: 簡單場景，易平衡
- **過濾**: 無特殊過濾

#### 3.2 難例收集 (`collect_targeted.py`)
- **輸出**: `play_data_targeted.csv` (10k 行)
- **方法**: 啟發式 agent vs 啟發式 agent，但僅記錄難例
- **過濾**: `球速 ≥12 OR |著陸誤差| ≥50px`
- **用途**: 提升 1P 難例能力

#### 3.3 特徵增強 (`augment_features.py`)
- **輸入**: `play_data_auto.csv`
- **輸出**: `play_data_auto_aug.csv`
- **新增特徵**:
  - `landing_dx`: 著陸點與平台中心距離
  - `ball_speed_abs`: 球速大小
  - `time_to_land`: 預計著陸時間（如可用）
- **目的**: 幫助模型識別難例特徵

---

### 4. 訓練層

#### 4.1 隨機森林訓練

**基礎訓練** (`train_random_forest.py`)
```python
RandomForestClassifier(
    n_estimators=100,      # 樹數量（精簡版）
    max_depth=12,          # 樹深度
    random_state=42,
    n_jobs=1,              # 單線程推理
    class_weight='balanced'
)
```

**1P 定向訓練** (`retrain_1p_targeted.py`)
- 融合: `play_data_auto_aug.csv` + `play_data_targeted.csv × multiplier`
- `multiplier=3`: 難例重複 3 次，提升權重
- 輸出: `ml/models/rf_1p.joblib`

**決策樹訓練** (同樣配置，較少用)

---

### 5. 推理層（Agent）

#### 5.1 1P Agent (`ml_play_rf_1p.py`)

```
輸入: scene_info（球、平台、狀態）
  ↓
特徵提取: 構造 19D 特徵向量
  ├─ ball_x, ball_y, ball_vx, ball_vy
  ├─ self_px, opp_px, blocker_x
  ├─ pred_landing_x (著陸預測)
  ├─ prev1/prev2 歷史特徵
  └─ landing_dx = pred_landing_x - self_px
  ↓
模型推理: rf_1p.predict(features)
  ↓
後處理 & 安全檢查:
  ├─ landing_dx 容限: if |landing_dx| ≤ 15px → NONE
  ├─ 方向平滑: 避免 <4 幀內頻繁轉向
  └─ 安全覆蓋: 球快速接近且著陸在左時強制 MOVE_LEFT
  ↓
輸出: 行動 (NONE, MOVE_LEFT, MOVE_RIGHT, SERVE_*)
```

**關鍵參數**:
- `landing_dx 容限 = 20px` (應對切球)
- `方向平滑窗口 = 4 幀`
- `安全覆蓋閾值 = self_px - 8`

#### 5.2 2P Agent (`ml_play_rf_2p.py`)
- 相同架構，稍少後處理
- 用 `play_data_auto_aug` 訓練（無難例增強）

---

### 6. 評估層

#### 6.1 無頭評估 (`evaluate_models.py`)
- 執行 N 場遊戲（無 UI）
- 統計勝負、得分、幀數
- 輸出 CSV: `eval_results_*.csv`

#### 6.2 分析工具 (`analyze_failures.py`)
- 分析失敗日誌，提取著陸誤差分布
- 找出常見失誤模式

#### 6.3 視覺演示 (`run_selfplay.py`)
- Pygame 窗口顯示 1P vs 2P
- 目標 FPS: 60
- 遊戲結束時統計勝負

---

## 數據流

### 訓練時數據流

```
play_data_auto.csv (20k)
    ↓ augment_features
play_data_auto_aug.csv (20k, 19 features)
    ↓ + play_data_targeted.csv × 3
Combined Data (50k)
    ↓ train_test_split (80-20)
Train: 40k → RandomForest → rf_1p.joblib
Test: 10k → Evaluate → accuracy_report
```

### 推理時數據流

```
scene_info (球、平台、狀態)
    ↓ extract_features
19D 向量
    ↓ rf_1p.predict()
預測動作 ID (0-4)
    ↓ post_process
最終動作 (MOVE_LEFT etc)
    ↓ game.update()
新 scene_info
```

---

## 性能指標

| 指標 | 值 | 備註 |
|------|-----|------|
| **著陸預測** | 6.6ms/call | max_steps=100, n_jobs=1 |
| **模型推理** | 6.6ms/predict | RF 100×12 |
| **幀耗時預算** | 16.67ms@60fps | 已滿足 |
| **訓練速度** | ~30sec/模型 | n_est=100, depth=12 |
| **1P 準確度** | ~91.6% | 測試集 (20% split) |

---

## 常見調整點

| 調整項 | 檔案 | 行數 | 效果 |
|--------|------|------|------|
| 著陸容限 | `ml_play_rf_1p.py` | 131 | 影響反應敏銳度 |
| 模型大小 | `retrain_1p_targeted.py` | 67 | n_est & max_depth |
| 著陸精度 | `ml/landing.py` | 20 | `max_steps` |
| 方向平滑 | `ml_play_rf_1p.py` | 144 | 震盪頻率 |

# PingPong ML 訓練流程 - 簡化版指南

## 建立訓練環境

```bash
# 建立並啟動訓練環境（若尚未建立）
conda env create -f .\envs\pingpong-train.yml
conda activate pingpong-train
```

---

## 分步驟執行 (推薦用於調試)

### 1. 收集基礎數據 (10k 行)
```bash
python ml/collect_data_auto.py --target_rows 10000 
```
- 輸出: `ml/data/play_data_auto.csv`
- 內容: 簡單場景，作為訓練基礎

### 2. 收集難例數據 (10k 行)
```bash
python ml/collect_targeted.py --target_rows 10000 
```
- 輸出: `ml/data/play_data_targeted.csv`
- 過濾條件: 球速 ≥12 或著陸誤差 ≥50px

### 3. 增加特徵工程
```bash
python ml/evaluate_models.py
```
- 輸出: `ml/data/play_data_auto_aug.csv`
- 新增特徵: landing_dx, ball_speed_abs, time_to_land

### 4. 訓練 1P 模型 
```bash
python ml/retrain_1p_targeted.py
```
- 模型: `ml/models/rf_1p.joblib`
- 配置: n_estimators=100, max_depth=12, n_jobs=1
- 資料融合: 基礎數據 + 難例×3倍

### 5. 訓練 2P 模型
```bash
python ml/train_random_forest.py
```
- 模型: `ml/models/rf_2p.joblib`
- 配置: 同上
- 資料: 只用基礎數據


### 6. 視覺演示 
```bash
python ml/run_selfplay.py
```
- 啟動 Pygame 演示窗口，觀看 1P vs 2P 實際遊玩

---

## 系統配置速查表

| 組件 | 參數 | 說明 |
|------|------|------|
| **著陸預測** | max_steps=100 | 預測計算迭代，已優化速度 |
| **1P Agent** | landing_dx容限=15px | 著陸點容限，應對切球邊界 |
| **RF 模型** | n_estimators=100<br>max_depth=12<br>n_jobs=1 | 輕量模型 (6.6ms/推理) |
| **數據乘數** | multiplier=3 | 1P 訓練時難例重複次數 |

---

## 調試工作流

### 1p輸得很慘：
1. 檢查著陸容限: `ml/ml_play_rf_1p.py` 第 130-135 行，`landing_dx <= 20`
2. 嘗試增加難例比例: `--multiplier 5` 或重新收集更多難例

### 如果 FPS 還是低：
1. 檢查模型推理: `python ml/benchmark_model.py`（應 <6.6ms）
2. 檢查著陸計算: `ml/landing.py` 第 20 行，`max_steps=100`

### 如果想看詳細診斷：
```bash
python -m ml.analyze_failures
```

---

## 關鍵文件清單

| 檔案 | 用途 |
|------|------|
| `ml/landing.py` | 著陸預測引擎（已優化） |
| `ml/ml_play_rf_1p.py` | 1P Agent（landing_dx=20） |
| `ml/ml_play_rf_2p.py` | 2P Agent |
| `ml/models/rf_1p.joblib` | 1P 模型 (n_est=100, depth=12) |
| `ml/models/rf_2p.joblib` | 2P 模型 (n_est=100, depth=12) |

---

## 典型執行時間

| 步驟 | 時間 |
|------|------|
| collect_auto (20k) | ~10 min |
| collect_targeted (10k) | ~5 min |
| augment | <1 sec |
| train_1p | ~30 sec |
| train_2p | ~30 sec |
| evaluate (20 matches) | ~2 min |
| **Total (all)** | **~20 min** |

---

## 常見問題

**Q: 要重新訓練嗎？**  
A: 如果要改模型配置（n_estimators, max_depth）或重新收集數據，需要重跑 train_1p / train_2p。

**Q: 數據可以重用嗎？**  
A: 是的。如果只改 Agent 邏輯（landing_dx容限等），不需重新訓練，直接修改 `ml_play_rf_1p.py` 重啟遊戲。


