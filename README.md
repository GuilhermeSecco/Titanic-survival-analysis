# Titanic Survival Prediction 🚢

Este projeto analisa os dados reais do naufrágio do Titanic para prever quais passageiros teriam maior chance de sobreviver, utilizando modelos de machine learning.

## 🔍 Etapas do Projeto

1. **Análise Exploratória de Dados (EDA)**:
   - Gráficos por classe, sexo, idade e tamanho da família.
   - Heatmap de correlação entre variáveis.

2. **Engenharia de Atributos**:
   - 'FamilySize' = 'SibSp' + 'Parch'
   - Agrupamento de idade por faixas ('AgeGroup')

3. **Tratamento de Dados**:
   - Preenchimento de valores ausentes com a mediana.
   - Codificação de variáveis categóricas.
   - Padronização com 'StandardScaler'.

4. **Modelagem e Avaliação**:
   - Modelos: Regressão Logística e Random Forest
   - Métricas: Acurácia, Matriz de Confusão e Relatório de Classificação

## 📊 Resultados

- Acurácia da Regressão Logística: **75.98%**
- Acurácia da Random Forest: **78.21%**
- Modelo balanceado, mas com recall mais baixo para a classe "sobreviveu".

## 🧠 Tecnologias Utilizadas

- Python (pandas, numpy, seaborn, matplotlib)
- Scikit-learn

---
