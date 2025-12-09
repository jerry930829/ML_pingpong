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


