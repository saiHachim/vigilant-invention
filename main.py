
#Sujet : Détection de fraudes liées à la carte de crédit (Masterclass)

#Importer les bibliothèques nécessaires et charger les données
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns


# # Step2 récupérer les données

#Telechargement des données:
creditmasterclass = pd.read_csv('/Users/hachim/Documents/Masterclass.csv')
print(creditmasterclass.head())

#1.Chargement des données :

# Séparer les données en features et en target

X = creditmasterclass.drop(['Class'], axis=1)
y = creditmasterclass['Class']

# Afficher les données ( les ligne et les colonnes)

print("Les features X sont :\n", X)
print("La target y est :\n", y)


# # Step3 l'analyse exploratoire des données


#Afficher le creditmasterclass
creditmasterclass.info()


creditmasterclass.head()

creditmasterclass


# Tracer l'histogramme des classes
sns.countplot(x='Class', data=creditmasterclass)
plt.show()



# # Step4 préparer les données en train et tester

# # classification : 

# Train test split



from sklearn.model_selection import train_test_split

X = creditmasterclass.drop(['Class'], axis=1)
y = creditmasterclass['Class']

# Diviser les données en ensembles de formation et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Choix du model régression logistique pour la classification

# In[10]:


# Appliquer la régression logistique pour la classification ( avec un nombre maximum d'itérations de 500)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=500)


# Entraîner le modèle sur les données :
model.fit(X_train, y_train)

print(model.coef_)
print(model.intercept_)


# # Evaluation :

# In[12]:


# Import des modules nécessaires
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import RocCurveDisplay



# Prédiction sur les données de test
y_predict = model.predict(X_test)


print(y_predict)


# La liste contient principalement des prédictions de la classe "0", sauf pour quelques éléments qui ont été prédits comme étant de la classe "1".


# Calcul de la précision de test du modèle
test_accuracy = accuracy_score(y_test, y_predict) * 100
print('Précision de test:', test_accuracy)



# Calcul de la matrice de confusion du modèle
confusion_mat = confusion_matrix(y_test, y_predict)
print('Matrice de confusion :\n', confusion_mat)


# Extraction des vrais négatifs (TN), des faux positifs (FP), des faux négatifs (FN) et des vrais positifs (TP) de la matrice de confusion

tn, fp, fn, tp = confusion_mat.ravel()
print('TN :', tn, 'FP :', fp, 'FN :', fn, 'TP :', tp)


# Affichage du rapport de classification pour le modèle, y compris la précision, le rappel, le score F1 et le support
class_report = classification_report(y_test, y_predict, digits=6)
print(class_report)


# In[22]:


#Tracé de la courbe ROC à l'aide de RocCurveDisplay.

roc_display = RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.show()


# On cherche la meilleur valeur de seuil 


y_test_predict_probs = model.predict_proba(X_test)[:,1]

y_test_predict_probs 



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_test = le.fit_transform(y_test)


from sklearn.metrics import roc_curve 

fpr,tpr,thresholds = roc_curve(y_test, y_test_predict_probs )


from numpy import argmax
best = tpr - fpr

ix = argmax(best)
best_threshold = thresholds [ix]
print ('Meilleur seuil = ', best_threshold)#(thresholds)







