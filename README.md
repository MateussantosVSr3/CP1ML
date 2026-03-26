# Checkpoint 01 — Machine Learning: Classificação e Regressão

- Mateus dos Santos da Silva RM558436
- André Giovane de Maria RM556384
- Nickolas Moreno Cardoso RM557132

## 📌 Sobre o Projeto
Este projeto foi desenvolvido como parte do Checkpoint 01 da disciplina de Machine Learning do curso de Engenharia de Software da FIAP. O objetivo principal é aplicar o pipeline completo de Machine Learning, desde a coleta e preparação dos dados até o treinamento e avaliação de modelos preditivos.

## 📊 Dataset Escolhido
**Medical Cost Personal Datasets** (Seguro de Saúde)
- **Fonte original:** Kaggle
- **Descrição:** O dataset contém dados demográficos e características corporais de pacientes (como idade, IMC, quantidade de filhos, se é fumante e região) e o valor em dólares cobrado pelo seguro de saúde.

## 🤖 Modelos Desenvolvidos

O projeto contempla duas abordagens para resolver diferentes necessidades de negócio:

1. **Modelo de Regressão (Regressão Linear):** - **Objetivo:** Prever o valor contínuo do custo do seguro de saúde (`charges`).
   - **Métrica de Avaliação:** MAE (Erro Médio Absoluto).

2. **Modelo de Classificação (KNN - K-Nearest Neighbors):** - **Objetivo:** Prever e classificar se um paciente é fumante ou não (`smoker_yes`) com base no seu perfil financeiro e corporal.
   - **Métrica de Avaliação:** Acurácia.

## ⚙️ Tecnologias e Bibliotecas
- Python 3
- Pandas (para leitura e manipulação dos dados)
- Scikit-Learn (para divisão de treino/teste, algoritmos de regressão/classificação e métricas)
