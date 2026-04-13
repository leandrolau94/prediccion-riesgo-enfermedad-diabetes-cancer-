import pandas as pd#libreria para cargar y leer datos csv
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