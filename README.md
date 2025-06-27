# Online Shoppers Purchase Prediction

Este projeto utiliza **Regressão Logística com PyTorch** para prever a **intenção de compra de usuários em um site de e-commerce**, com base em dados comportamentais extraídos de sessões de navegação.

---

## Objetivo

Desenvolver um modelo de **classificação binária** que identifique se um visitante realizará ou não uma compra, com base em métricas como:
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

- Regresão Logística com PyTorch
- Engenharia de Atributos
- One-Hot Encoding para variáveis categóricas
- Padronização (`StandardScaler`)
- Avaliação com:
  - Acurácia
  - Classification Report
  - Matriz de Confusão
  - Curva ROC e AUC

---

## Estrutura do Projeto

online-shoppers-purchase-prediction/
├── data/
│ └── online_shoppers_intention.csv
├── models/
│ └── model.pth
├── plots/
│ └── roc_curve.png
├── src/
│ ├── data_preparation.py
│ ├── logistic_model.py
│ ├── train.py
│ ├── evaluate.py
│ └── config.py (opcional)
├── main.py
├── requirements.txt
└── README.md

---

## Como Rodar o Projeto

1. Clone o repositório:

```bash
git clone https://github.com/rafaelgutierres049/online-shoppers-purchase-prediction.git
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

## Resultados esperados

Acurácia média esperada: ~85% (pode variar por random state ou ajustes)

Gráficos salvos na pasta plots/:

    -Curva ROC com AUC

    -Matriz de confusão exibida em tela

Relatório de Classificação exibido no terminal

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

