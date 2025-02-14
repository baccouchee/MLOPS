# -*- coding: utf-8 -*-
"""Modeling & Evaluation.ipynb"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle

# Charger les datasets
train_path = "data/train_clean.csv"  # Remplace par ton chemin réel
test_path = "data/test_clean.csv"    # Remplace par ton chemin réel

train_clean = pd.read_csv(train_path)
test_clean = pd.read_csv(test_path)

# 1️⃣ Afficher les 5 premières lignes
print("\n🔹 Aperçu des 5 premières lignes du dataset d'entraînement (TR) :")
print(train_clean.head())

print("\n🔹 Aperçu des 5 premières lignes du dataset de test (TS) :")
print(test_clean.head())

# 2️⃣ Obtenir les dimensions des datasets
print("\n📏 Dimensions des datasets :")
print(f"Train: {train_clean.shape}, Test: {test_clean.shape}")

# 3️⃣ Identifier les types de variables
print("\n🔍 Types de variables dans le dataset d'entraînement (TR) :")
print(train_clean.dtypes)

print("\n🔍 Types de variables dans le dataset de test (TS) :")
print(test_clean.dtypes)

print("\n✅ Exploration terminée avec succès !")

# Sélectionner uniquement les colonnes numériques
numeric_columns = train_clean.select_dtypes(include=['float64', 'int64']).columns

# Calculer la matrice de corrélation sur les colonnes numériques
corr_matrix = train_clean[numeric_columns].corr()

# Identification des paires de variables avec une corrélation supérieure à 0.9
high_corr_var = [(i, j) for i in corr_matrix.columns for j in corr_matrix.columns if i != j and corr_matrix.loc[i, j] > 0.9]
print("Paires de variables avec une corrélation > 0.9 :")
print(high_corr_var)

# Affichage de la matrice de corrélation
print("\nMatrice de corrélation :")
print(corr_matrix)

# Sélection des variables catégorielles
cat_columns = train_clean.select_dtypes(include=['object']).columns

# Test du khi-deux pour chaque paire de variables catégorielles
chi2_p_values = {}
for col1 in cat_columns:
    for col2 in cat_columns:
        if col1 != col2:
            contingency_table = pd.crosstab(train_clean[col1], train_clean[col2])
            if contingency_table.size > 0:  # Ensure the contingency table is not empty
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                if p < 0.05:
                    chi2_p_values[(col1, col2)] = p
            else:
                print(f"Contingency table for {col1} and {col2} is empty, skipping chi-square test.")

# Affichage des p-values
print("\nP-values des tests du khi-deux pour les paires de variables catégorielles :")
for pair, p_value in chi2_p_values.items():
    print(f"{pair}: p-value = {p_value}")

# Gestion des valeurs manquantes pour les colonnes numériques
imputer = SimpleImputer(strategy='mean')
train_imputed = imputer.fit_transform(train_clean[numeric_columns])

# Standardisation des données
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_imputed)

# Application de la PCA
pca = PCA()
train_pca = pca.fit_transform(train_scaled)

# Variance expliquée par chaque composante principale
explained_variance = pca.explained_variance_ratio_

# Affichage de la variance expliquée cumulée
cumulative_variance = explained_variance.cumsum()
print("\nVariance expliquée cumulée par les composantes principales :")
print(cumulative_variance)

# Affichage de l'histogramme de la variable cible
plt.figure(figsize=(10, 6))
sns.histplot(train_clean['saleprice'], kde=True, color='blue', bins=30)
plt.title('Distribution de la variable cible : SalePrice')
plt.xlabel('saleprice')
plt.ylabel('Fréquence')
plt.show()

# Encodage one-hot pour les variables catégorielles
train_clean = pd.get_dummies(train_clean, columns=cat_columns, drop_first=True)

# Définir 3 groupes : "Bas", "Moyen", "Élevé"
train_clean["Price_Category"] = pd.qcut(train_clean["saleprice"], q=3, labels=["Bas", "Moyen", "Élevé"])

# Séparer X et y
X = train_clean.drop(columns=["id", "saleprice", "Price_Category"])
y = train_clean["Price_Category"]

# Appliquer le suréchantillonnage
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Afficher la nouvelle distribution après suréchantillonnage
plt.figure(figsize=(8, 5))
sns.countplot(x=y_resampled)
plt.title("Distribution des catégories de SalePrice après suréchantillonnage")
plt.xlabel("Catégorie de prix")
plt.ylabel("Nombre d'exemples")
plt.show()

# Séparer les données en train et test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialiser le modèle RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)

# Entraîner le modèle
rf_model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = rf_model.predict(X_test)

# Évaluer les performances du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy du modèle Random Forest (Après suréchantillonnage) : {accuracy:.2f}")

# Afficher un rapport de classification pour évaluer les performances sur chaque catégorie
print("Classification Report :")
print(classification_report(y_test, y_pred))

# Matrice de confusion après suréchantillonnage
cm_after = confusion_matrix(y_test, y_pred)

# Affichage de la matrice de confusion après suréchantillonnage
plt.figure(figsize=(8, 6))
sns.heatmap(cm_after, annot=True, fmt="d", cmap="Blues", xticklabels=["Bas", "Moyen", "Élevé"], yticklabels=["Bas", "Moyen", "Élevé"])
plt.title("Matrice de confusion - Après suréchantillonnage")
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.show()

# Comparaison avant et après suréchantillonnage
# Séparer les données en train et test (80% train, 20% test) avant suréchantillonnage
X_train_before, X_test_before, y_train_before, y_test_before = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le modèle RandomForestClassifier
rf_model_before = RandomForestClassifier(random_state=42)

# Entraîner le modèle sur les données avant le suréchantillonnage
rf_model_before.fit(X_train_before, y_train_before)

# Faire des prédictions sur l'ensemble de test
y_pred_before = rf_model_before.predict(X_test_before)

# Évaluer les performances du modèle avant suréchantillonnage
accuracy_before = accuracy_score(y_test_before, y_pred_before)
print(f"Accuracy du modèle Random Forest (avant suréchantillonnage) : {accuracy_before:.2f}")

# Afficher un rapport de classification pour évaluer les performances sur chaque catégorie
print("Classification Report (avant suréchantillonnage) :")
print(classification_report(y_test_before, y_pred_before))

# Matrice de confusion avant suréchantillonnage
cm_before = confusion_matrix(y_test_before, y_pred_before)

# Affichage de la matrice de confusion avant suréchantillonnage
plt.figure(figsize=(8, 6))
sns.heatmap(cm_before, annot=True, fmt="d", cmap="Blues", xticklabels=["Bas", "Moyen", "Élevé"], yticklabels=["Bas", "Moyen", "Élevé"])
plt.title("Matrice de confusion - Avant suréchantillonnage")
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.show()

# Save the trained model
model_path = 'model/random_forest_model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(rf_model, file)

# Evaluate the model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)