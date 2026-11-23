import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    make_scorer, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

# ------------------------------------------------------------
# 1. Amostra: mesmos arquivos do grupo, sem refazer o split
# ------------------------------------------------------------
treino_df = pd.read_csv(r'C:\Users\User\Projeto_CienciadeDados\dataset_treino.csv')
teste_df  = pd.read_csv(r'C:\Users\User\Projeto_CienciadeDados\dataset_test.csv')

X_train = treino_df.drop(columns=["Result"])
y_train = treino_df["Result"]

X_test = teste_df.drop(columns=["Result"])   # atenção: teste_df, não treino_df
y_test = teste_df["Result"]

# ------------------------------------------------------------
# 2. Escolha dos parâmetros (só no treino, validação cruzada)
#    Critério definido ANTES de rodar: maior F1 médio da classe phishing
# ------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_phishing = make_scorer(f1_score, pos_label=-1)

# Variando UM parâmetro por vez (num_leaves), o resto fixo
configs = {
    "num_leaves=15": 15,
    "num_leaves=31": 31,
    "num_leaves=63": 63,
}

resultados_cv = {}
for nome, n_leaves in configs.items():
    modelo = lgb.LGBMClassifier(
        num_leaves=n_leaves,
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring=f1_phishing)
    resultados_cv[nome] = (scores.mean(), scores.std())
    print(f"{nome}: F1 médio = {scores.mean():.4f} ± {scores.std():.4f}")

melhor = max(resultados_cv, key=lambda k: resultados_cv[k][0])
print(f"\nConfiguração escolhida: {melhor}")
print(f"F1 médio ± desvio (CV): {resultados_cv[melhor][0]:.4f} ± {resultados_cv[melhor][1]:.4f}")

# ------------------------------------------------------------
# 3. Avaliação no teste (UMA única vez)
# ------------------------------------------------------------
modelo_final = lgb.LGBMClassifier(
    num_leaves=configs[melhor],
    n_estimators=200,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
modelo_final.fit(X_train, y_train)
y_pred = modelo_final.predict(X_test)

# ------------------------------------------------------------
# 4. Resultados no formato do relatório
# ------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
VP, FN = cm[0, 0], cm[0, 1]
FP, VN = cm[1, 0], cm[1, 1]
total = VP + FN + FP + VN

print("\nMatriz de confusão (positiva = phishing, -1)")
print(f"Real: phishing  | VP = {VP} | FN = {FN}")
print(f"Real: legítimo  | FP = {FP} | VN = {VN}")

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=-1)
rec  = recall_score(y_test, y_pred, pos_label=-1)
f1   = f1_score(y_test, y_pred, pos_label=-1)

print(f"\nAcurácia: ({VP} + {VN}) / {total} = {acc:.4f}")
print(f"Precisão: {VP} / ({VP} + {FP}) = {prec:.4f}")
print(f"Recall  : {VP} / ({VP} + {FN}) = {rec:.4f}")
print(f"F1      : 2 · ({prec:.4f} · {rec:.4f}) / ({prec:.4f} + {rec:.4f}) = {f1:.4f}")