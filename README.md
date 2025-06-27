# Online Shoppers Purchase Prediction

Este projeto utiliza **Regressão Logística com PyTorch** para prever a **intenção de compra de usuários em um site de e-commerce**, com base em dados comportamentais extraídos de sessões de navegação.

---

## Objetivo

Desenvolver um modelo de classificação binária que identifique se um visitante realizará ou não uma compra, com base em métricas como:
- Tempo de navegação
- Número de páginas visitadas
- Taxas de rejeição
- Tipo de visitante
- Período da sessão, entre outros

---

## Dataset

- **Nome**: [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)  
- **Fonte**: UCI Machine Learning Repository  
- **Tamanho**: 12.330 registros | 18 features  
- **Target**: `Revenue` → 1 (Compra realizada), 0 (Não realizou compra)

---

## Técnicas Utilizadas

- Regressão Logística com PyTorch
- Engenharia de atributos
- One-Hot Encoding para variáveis categóricas
- Padronização com `StandardScaler`
- **Validação Cruzada (5-Fold)**
- Ajuste de `learning rate`
- Regularização **L2 (weight_decay)** e ajuste de **classe desbalanceada (pos_weight)**

---

## Avaliação do Modelo

### Resultados médios com Validação Cruzada (5 Folds):

- **F1-Score**: `0.5205 ± 0.0306`  
- **Recall**: `0.7765 ± 0.0227`  
- **Precision**: `0.3917 ± 0.0289`  
- **Accuracy**: `0.7821 ± 0.0208`

O modelo apresenta **bom equilíbrio entre recall e precisão**, especialmente relevante em casos com classe minoritária importante (compradores).

---

## Estrutura do Projeto

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

## Como Rodar o Projeto

1. Clone o repositório:

```bash
git clone https://github.com/rafaelgutierres049/Online_Shoppers_Intention.git
cd online-shoppers-purchase-prediction
```

---

2. Instale os pacotes necessários:

```bash
pip install -r requirements.txt
```

---

3. Execute o pipeline completo:

```bash
python main.py
```

---

## Saídas do Projeto

Classificação binária (compra ou não)

Relatório exibido no terminal:

    Acurácia, Precision, Recall, F1-Score

Gráficos salvos em /plots:

    Curva ROC com AUC

    Matriz de confusão (exibida via matplotlib)

---

## Tecnologias Utilizadas

Python 3.10+

PyTorch

scikit-learn

pandas

matplotlib

---

## Licença 

Este projeto está licenciado sob a MIT License.

---

## Autor

Desenvolvido por Rafael Ponte Gutierres.

