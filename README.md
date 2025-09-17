# Indian Birds Classification

This project focuses on classifying **25 Indian bird species** using both **classical machine learning with hand-crafted features** and **deep learning approaches (CNNs)**.  
Dataset link (Kaggle): [Indian Birds Dataset](https://www.kaggle.com/datasets/ichhadhari/indian-birds/data)

The notebook: **`bird-classification.ipynb`**

---

## Project Overview
We build a complete end-to-end pipeline:

1. **Feature Extraction → Classical ML**  
   - Extract image features such as **Color Histograms**, **SIFT**, **HOG**, and **Gabor filters**.  
   - Train models: **SVM**, **Random Forest**, **Naive Bayes**, **MLP**.  
   - Each feature set is evaluated separately and compared.

2. **Dimensionality Reduction & Feature Selection**  
   - Apply **PCA** for dimensionality reduction.  
   - Use one **feature elimination** method (e.g., `SelectKBest`, `VarianceThreshold`, `mutual_info_classif`).  
   - Re-train ML models and compare results against raw features.

3. **Fine-Tuning Pretrained CNNs**  
   - Fine-tune at least three backbones (e.g., **ResNet50**, **EfficientNet-B0**, **MobileNetV3**) from **timm** or `torchvision`.  
   - Log loss/accuracy curves, apply **early stopping**, and save best checkpoints.

4. **Training From Scratch**  
   - Implement a **custom CNN** from scratch.  
   - Train the same pretrained models with **random weights** and compare them to fine-tuning results.

**Evaluation metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix.  
**Validation strategy:** 80-10-10 split (train/test/val). If the dataset provides 1,200 train + 300 val per class, split the 300 validation into 150 test + 150 val to maintain the ratio.

---

## Dataset
- **25 species**, **1,500 images per species**, ~**37,500 images total**.  
- Each image has ~1 MP resolution.  


Download via Kaggle CLI:
```bash
kaggle datasets download -d ichhadhari/indian-birds -p data/
unzip data/indian-birds.zip -d data/indian-birds
```

## Installation
```bash
# Python 3.9+ recommended
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Run
jupyter lab   # or: jupyter notebook
Open 'bird-classification.ipynb' and run cells in order

