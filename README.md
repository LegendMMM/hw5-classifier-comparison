# HW5 分類器效能比較與模型選擇

| 檔案 | 內容 |
|---|---|
| `part1_practice.py` | Part 1：ScikitLearn 操作記錄單的練習（每個 classifier 兩組以上設定、confusion_matrix、classification_report、F1、5-fold CV） |
| `hw5.py` | Part 2–5：Breast Cancer baseline、Standardization 比較、5-fold stratified CV hyperparameter selection、Digits 比較 |

## 執行

```bash
pip install -r requirements.txt
python part1_practice.py   # 輸出 part1_results.json
python hw5.py              # 輸出 results.json 與 figures/*.png
```

實驗規則：`random_state=42`、80%/20% stratified split、StandardScaler 只在 training set（或每個 training fold 內，用 Pipeline）fit，test set 只在選定設定後評估一次。
