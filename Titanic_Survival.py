#Importando as Bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#Análise exploratória
df = pd.read_csv('Titanic-Dataset.csv')
print(df.columns)
print(df.head())
print(df.isnull().sum())

#Criando novas variáveis
df['FamilySize'] = df['SibSp'] + df['Parch']
def categorize_age(age):
    if pd.isnull(age):
        return 'Unknown'
    elif age < 10:
        return '0s'
    elif age < 20:
        return '10s'
    elif age < 30:
        return '20s'
    elif age < 40:
        return '30s'
    elif age < 50:
        return '40s'
    elif age < 60:
        return '50s'
    elif age < 70:
        return '60s'
    else:
        return '70+'

df['AgeGroup'] = df['Age'].apply(categorize_age)
order = ['Unknown', '0s', '10s', '20s', '30s', '40s', '50s', '60s', '70+']

#Visualizações
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Survival Count by Class')
plt.show()

sns.countplot(data=df, x='Sex', hue='Survived')
plt.title('Survival Count by Sex')
plt.show()

sns.countplot(data=df, x='AgeGroup', hue='Survived', order=order)
plt.title('Survival Count by Age Group')
plt.show()

sns.countplot(data=df, x='FamilySize', hue='Survived')
plt.title('Survival Count by Family Size')
plt.show()

#Tratando dados nulos
df['Age'] = df['Age'].fillna(df['Age'].median())
print(df.isnull().sum())

#Transformando string em int
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

#Separando Colunas que serão utilizadas pelos modelos
X = df[['Pclass', 'Age', 'Sex', 'FamilySize', 'Fare']]
y = df['Survived']

#Heatmap para mostrar a correlação entre as variáveis
plt.title('Correlations')
sns.heatmap(df[['Survived', 'Pclass', 'Age', 'FamilySize', 'Sex', 'Fare']].corr(), annot=True, cmap='coolwarm')
plt.show()

#Reescalando as variáveis para evitar influência desproporcional pelos valores
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Separando o dataset entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,test_size=0.2, random_state=9)

#Aplicando o modelo de Regressão lógica
model = LogisticRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)

#Aplicando o modelo de Random Forest
rf_model = RandomForestClassifier(random_state=9)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

#Testando a acurácia da Regressão lógica
accuracy = accuracy_score(y_test, predictions)
print(f'\nAccuracy: {accuracy * 100:.2f}%')

#Testando a acurácia da Random Forest
rf_acc = accuracy_score(y_test, rf_preds)
print(f'\nRandom Forest Accuracy: {rf_acc * 100:.2f}%')

#Matrix de confusão
print("\n:Confusion Matrix")
print(confusion_matrix(y_test, predictions))

#Relatório de classificação
print("\nClassification Report:")
print(classification_report(y_test, predictions))
