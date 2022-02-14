import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("D:/Programming/Udemy - 2022 Python for ML & DS/DATA/penguins_size.csv")

number_of_species = df['species'].unique()
number_of_null = df.isnull().sum()

df_new = df.dropna()  
df_new.at[336,'sex'] = 'FEMALE'
# a = sns.pairplot(df,hue= 'species')
# print(a)

X = pd.get_dummies(df_new.drop('species',axis=1),drop_first=True)
y = df_new['species']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


from sklearn.tree import DecisionTreeClassifier, plot_tree
model = DecisionTreeClassifier()
model = model.fit(X_train,y_train)

from sklearn.metrics import confusion_matrix, classification_report
def report_model(model):
    model_preds = model.predict(X_test)
    print(classification_report(y_test,model_preds))
    print('\n')
    plt.figure(figsize= (12,8), dpi= 200)
    plot_tree(model,feature_names= X.columns,filled=True);
