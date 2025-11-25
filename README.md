# HW3 Sentiment Anaysis README
這份 README 是用來重現我這次 HW3 模型的，依照下面步驟操作即能重現我的結果。

---

## 1. 檔案放的位置

我在 Google Drive 裡用以下結構：

```
MyDrive/HW3/
 ├── dataset/
 │    └── dataset.csv
 └── solo_train_deberta_base.py
```

`dataset.csv` 有兩欄：

| text | label |
|------|--------|
| 句子內容 | 0/1/2 或 Negative/Neutral/Positive |

---

## 2. 在 Colab 重現

### (1) 掛載 Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### (2) 執行訓練腳本
```python
!python /content/drive/MyDrive/HW3/solo_train_deberta_base.py
```

所有訓練流程（切資料、tokenize、AMP、scheduler…）都內建在腳本裡，不需要額外設定。

---

## 3. 執行完後會輸出的資料（請助教確認）

程式跑完後會生成：

```
saved_deberta_base_2025_BEST/
 ├── checkpoint/
 │    ├── config.json
 │    ├── pytorch_model.safetensors
 │    ├── tokenizer.json
 │    ├── tokenizer_config.json
 ├── loss_curve.png
 ├── val_accuracy_curve.png
 ├── confusion_matrix.png
 ├── confusion_matrix.csv
 ├── classification_report.txt
 └── summary.json
```

重現是否成功可以看：

- checkpoint 是否存在（尤其是 `.safetensors`）
- summary.json 是否存在  
- loss / accuracy 曲線是否生成  
- confusion matrix 是否生成  
- validation accuracy 是否跟我報告的差不多（差 0.01 左右很正常）

---

## 4. 訓練超參數

| 參數 | 值 |
|------|------|
| Model | microsoft/deberta-v3-base |
| Seed | 2025 |
| Epochs | 6 |
| Batch size | 32 |
| Learning rate | 2e-5 |
| Train/Val split | 9:1 |
| Precision | AMP (fp16) |

`summary.json` 也會記錄這些資訊。

---

## 5. 重現完成

按照上面流程跑，就能重現我這次的訓練結果。

