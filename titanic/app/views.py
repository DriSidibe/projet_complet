from django.shortcuts import render, HttpResponse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import pandas as pd
import numpy as np
import math

def home(request):
    survie = None
    accuracy = ""
    if request.method == "POST":
        donnees = {
        'pclass':[float(request.POST.get('pclass'))],
        'sex' :[float(request.POST.get('sexe'))],
        'age':[float(request.POST.get('age'))],
        'sibsp':[float(request.POST.get('sibsp'))]
                }
        indexes=['0']
        df = pd.DataFrame(donnees,index=indexes)
        model, accuracy = train_titanic_model(f"{os.getcwd()}/app/static/Data/train.csv")
        result = model.predict(np.array(df))
        survie = result
        accuracy = str(int(accuracy*100))
    return render(request, 'home.html', {"survie" : survie, "accuracy" : accuracy})

def train_titanic_model(data_file):
    # Charger les données du Titanic à partir d'un fichier CSV
    df = pd.read_csv(data_file)

    i = 0
    index_tab = []
    for a in df.age:
        if math.isnan(a):
            index_tab.append(i)
        i += 1
    df = df.drop(index_tab)

    # Prétraitement des données
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})

    # Sélection des caractéristiques (features) et de la cible (target)
    X = df[['pclass', 'sex', 'age', 'sibsp']]
    y = df['survived']

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialiser et entraîner un modèle de classification (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prédire les étiquettes sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Calculer l'exactitude (accuracy) du modèle
    accuracy = accuracy_score(y_test, y_pred)

    # Retourner le modèle entraîné et son exactitude
    return model, accuracy