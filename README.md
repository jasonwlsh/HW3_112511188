# HW3: Sentiment Analysis — Reproducibility Guide (Colab)

請助教參考這份README來重現我的訓練結果~

---

## 1. 檔案結構 (File Structure)

請在Colab 變更執行階段類型中選擇T4 GPU

並在Google Drive內下載好以下結構：

(解壓縮我的檔案後直接匯入Google Drive即可)
```
MyDrive/HW3_112511188/
├── dataset/
│   ├── dataset.csv
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── main.py
├── saved_models/   # 一開始沒有，執行後會生成
├── 112511188.pdf
└── requirements.txt
```
---
## 2. 環境設定 (Environment)

本作業是在colab上執行的，雖然colab已經下載好大部分模組，但為確保版本一致

請先在colab開啟terminal並安裝所有requirements：

```bash
cd drive/MyDrive/HW3_112511188
pip install -r requirements.txt
```

使用的主要工具：
- Python 3  
- PyTorch + CUDA（colab 預設環境）  
- Transformers  
- scikit-learn  
- pandas / numpy  


---

## 3. 在 Colab 上執行 

### **(1) 掛載 Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```

### **(2) 執行訓練**
```python
python /content/drive/MyDrive/HW3_112511188/main.py
```

程式會自動：
- 讀入 dataset
- 切分 train/val
- 執行訓練（使用 AMP）
- 儲存最佳 checkpoint


---

## 4. 訓練超參數 (Hyperparameters)

| 參數 | 值 |
|------|------|
| Model | microsoft/deberta-v3-base |
| Seed | 2025 |
| Batch Size | 32 |
| Epochs | 6 |
| Learning Rate | 2e-5 |
| Max Length | 128 |
| Optimizer | AdamW + Linear Warmup/Decay |
| Train/Val | 9:1 |
| Precision | AMP (fp16) |

所有參數會寫進 `summary.json`，可以去裡面查看。

---

## 5. 輸出結果 (Outputs)

執行後會產生：

```
saved_models/
 ├── checkpoint/
 ├── loss_curve.png
 ├── val_accuracy_curve.png
 ├── confusion_matrix.png
 ├── confusion_matrix.csv
 ├── classification_report.txt
 └── summary.json
```

內容包含：
- 最佳模型 checkpoint（模型權重檔 `model.safetensors` 應在此資料夾內）
- 訓練與驗證曲線
- 混淆矩陣
- Precision / Recall / F1 報告
- 超參數與最佳 Val Acc

---

## 6. 模型架構簡述

- Backbone：DeBERTa-v3-Base  
- Head：Linear → GELU → Dropout → Linear  
- Loss：CrossEntropyLoss  
（架構整合於 `main.py`）

---

依照上述步驟在 **colab 上執行** 就能完整重現我的結果。
