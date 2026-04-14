import pandas as pd#libreria para cargar y leer datos csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/diabetes.csv")

df.columns = ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "BMI", "diabetes_pedigree_function", "age", "outcome"]

print(df.head())
print(df.info())

print(df.describe()) # basic statistics

print(df["outcome"].value_counts()) # distribution of objective variable

# visualize diabetes distribution
sns.countplot(x="outcome", data=df)
plt.title("Diabetes Distribution")
plt.show()

# look for suspects zeros
print((df == 0).sum())

# data cleaning (replace zeros by NaN)
cols_with_zero = ["glucose", "blood_pressure", "skin_thickness", "insulin", "BMI"]
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
print(df.isnull().sum()) # watch nule values


# imputation of NaN values by robust median
df.fillna(df.median(), inplace=True)

# comprobar que se reemplazaron todos los Nan por las medianas de cada columna
print(df.isnull().sum())

# Aqui comienza la prediccion del modelo predictivo medico
X = df.drop("outcome", axis=1) # variables independientes
y = df["outcome"] # variable objetivo

# como la regresion logistica es sensible a escala y las variables no estan normalizadas, entonces
# antes mejor normalizar siguiendo (score z) la distribucion normal [0, 1], asi son mas fiables los resultados
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns) # keep pandas dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) # 80% entrenamiento, 20% prueba

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# make predicitions
y_pred = model.predict(X_test)

# evaluation
print("Precisión:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# visualizar que factores influyen en la diabetes
feature_importance = pd.DataFrame({"feature": X.columns, "importance": model.coef_[0]})
feature_importance = feature_importance.sort_values(by="importance", ascending=False)
print(feature_importance)

# visualización
sns.barplot(x="importance", y="feature", data=feature_importance)
plt.title("Importance of features (linear regression)")
plt.show()