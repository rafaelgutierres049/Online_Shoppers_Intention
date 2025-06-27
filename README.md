# Online Shoppers Purchase Prediction

This project uses Logistic Regression with PyTorch to predict the purchase intention of users on an e-commerce website, based on behavioral data extracted from browsing sessions.

---

## Objective

Develop a binary classification model to identify whether a visitor will make a purchase or not, based on metrics such as:
- Browsing time
- Number of pages visited
- Bounce rates
- Visitor type
- Session period, among others

---

## Dataset

- **Name**: [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)  
- **Font**: UCI Machine Learning Repository  
- **Size**: 12.330 registers | 18 features  
- **Target**: `Revenue` → 1 (Made a purchase), 0 (Did not made a purchase)

---

## Techniques Used

- Logistic Regression implemented with PyTorch
- Feature Engineering
- One-Hot Encoding for categorical variables
- Standardization using StandardScaler
- Cross-Validation (5-Fold)
- Learning rate tuning
- L2 Regularization (weight_decay) and handling class imbalance with pos_weight

---

## Model Evaluation

### Average results from 5-Fold Cross-Validation:

- **F1-Score**: `0.5205 ± 0.0306`  
- **Recall**: `0.7765 ± 0.0227`  
- **Precision**: `0.3917 ± 0.0289`  
- **Accuracy**: `0.7821 ± 0.0208`

The model achieves a **good balance between recall and precision**, which is especially important in cases with an important minority class (buyers)
---

## Project and Structure

data/
    online_shoppers_intention.csv
models/
    model.pth
plots/
    feature_importance.png
    precision_recall.png
    roc_curve.png
saved_model/
    logistic_regression_model.pth
src/
    cross_validation.py
    data_preparation.py
    evaluate.py
    features_importance.py
    logistic_model.py
    train.py
    tune_lr.py
main.py
README.md
requirements.txt

---

## How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/rafaelgutierres049/Online_Shoppers_Intention.git
cd online-shoppers-purchase-prediction
```

---

2. Install required packages:

```bash
pip install -r requirements.txt
```

---

3. Run the complete pipeline:

```bash
python main.py
```

---

## Project Outputs

Binary classification (purchase or not)

Terminal output report:

- Accuracy, Precision, Recall, F1-Score

Plots saved to /plots folder:

- ROC Curve with AUC

- Confusion Matrix (also displayed with matplotlib)

---

## Used Technologies

Python 3.10+

PyTorch

scikit-learn

pandas

matplotlib

---

## License

This project is licensed under the MIT License.

---

## Autor

Developed by Rafael Ponte Gutierres.
