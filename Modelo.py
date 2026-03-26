import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score

# Carregando os dados do dataset Medical Cost Personal (Kaggle)
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
df = pd.read_csv(url)

# Convertendo variáveis categóricas (texto) para numéricas (0 e 1)
df_numerico = pd.get_dummies(df, drop_first=True)

# MODELO 1: REGRESSÃO LINEAR (Prever o valor do seguro)
X_regressao = df_numerico.drop('charges', axis=1) 
y_regressao = df_numerico['charges'] 

# Divisão de 80% para treino e 20% para teste
X_treino_r, X_teste_r, y_treino_r, y_teste_r = train_test_split(X_regressao, y_regressao, test_size=0.2)

modelo_regressao = LinearRegression()
modelo_regressao.fit(X_treino_r, y_treino_r)
previsao_r = modelo_regressao.predict(X_teste_r)

erro = mean_absolute_error(y_teste_r, previsao_r)
print(f"Regressão: O modelo erra o valor do seguro em média por: ${erro:.2f}")

# MODELO 2: CLASSIFICAÇÃO COM REGRESSÃO LOGÍSTICA (Prever se é fumante)
X_classificacao = df_numerico.drop('smoker_yes', axis=1) 
y_classificacao = df_numerico['smoker_yes']

X_treino_c, X_teste_c, y_treino_c, y_teste_c = train_test_split(X_classificacao, y_classificacao, test_size=0.2)

modelo_classificacao = LogisticRegression(max_iter=1000)
modelo_classificacao.fit(X_treino_c, y_treino_c)
previsao_c = modelo_classificacao.predict(X_teste_c)

acuracia = accuracy_score(y_teste_c, previsao_c)
print(f"Classificação: O modelo acertou {acuracia * 100:.2f}% das vezes se o paciente é fumante.")