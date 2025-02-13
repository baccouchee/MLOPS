# -*- coding: utf-8 -*-
import os
import pandas as pd
# Ensure the data directory exists
os.makedirs('data', exist_ok=True)


    
# Print the permissions of the data directory
print("\n🔍 Permissions of the 'data' directory:")
os.system('ls -ld data')

# Correct paths to the data files
train_path = "data/train.csv"
test_path = "data/test.csv"

# Print the current working directory
print("\n🔍 Current working directory:")
os.system('pwd')

# Print the contents of the data directory
print("\n🔍 Contents of the 'data' directory:")
os.system('ls -l data')

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 1️⃣ Afficher les 5 premières lignes
print("\n🔹 Aperçu des 5 premières lignes du dataset d'entraînement (TR) :")
print(train_df.head())

print("\n🔹 Aperçu des 5 premières lignes du dataset de test (TS) :")
print(test_df.head())

# 2️⃣ Obtenir les dimensions des datasets
print("\n📏 Dimensions des datasets :")
print(f"Train: {train_df.shape}, Test: {test_df.shape}")



# 3️⃣ Identifier les types de variables
print("\n🔍 Types de variables dans le dataset d'entraînement (TR) :")
print(train_df.dtypes)

print("\n🔍 Types de variables dans le dataset de test (TS) :")
print(test_df.dtypes)

print("\n✅ Exploration terminée avec succès !")

import pandas as pd
import numpy as np


# 🔹 Spécifier les valeurs manquantes courantes (ex: "NA", "None", "null")
train_df = pd.read_csv(train_path, na_values=["", " ", "NA", "None", "null"])
test_df = pd.read_csv(test_path, na_values=["", " ", "NA", "None", "null"])

# 🔹 1. Identifier les valeurs manquantes et calculer leur pourcentage
def missing_values_info(df, name):
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100

    # Afficher uniquement les colonnes ayant des valeurs manquantes
    missing_df = pd.DataFrame({
        'Colonnes': df.columns,
        'Valeurs Manquantes': missing_count,
        'Pourcentage (%)': missing_percent
    })

    missing_df = missing_df[missing_df['Valeurs Manquantes'] > 0].sort_values(by='Pourcentage (%)', ascending=False)

    print(f"\n📌 Valeurs manquantes dans {name} :")
    if missing_df.empty:
        print("✅ Aucune valeur manquante détectée !")
    else:
        print(missing_df)

# Appliquer la fonction aux datasets
missing_values_info(train_df, "Train")
missing_values_info(test_df, "Test")

# Vérification manuelle : Afficher des exemples de valeurs
print("\n🔍 Exemples de valeurs uniques dans quelques colonnes :")
for col in train_df.columns[:5]:  # Vérifier les 5 premières colonnes
    print(f"\n🔹 {col}: {train_df[col].unique()[:10]}")

# 🔹 2. Afficher les statistiques descriptives (moyenne, médiane, écart-type)
print("\n📊 Statistiques descriptives du dataset Train :")
print(train_df.describe())

print("\n📊 Statistiques descriptives du dataset Test :")
print(test_df.describe())

print("\n🔍 Colonnes disponibles dans le dataset :")
print(train_df.columns.tolist())

train_df.columns = train_df.columns.str.strip()  # Supprime les espaces autour des noms de colonnes
train_df.columns = train_df.columns.str.lower()  # Convertit en minuscules pour éviter les erreurs

# Afficher les valeurs uniques de saleprice
unique_saleprice = train_df['saleprice'].unique()
print(unique_saleprice)

print("\n🔍 Colonnes disponibles dans le dataset :")
print(train_df.columns.tolist())

# 🔹 Spécifier les valeurs manquantes courantes (ex: "NA", "None", "null")
train_df_imputed = pd.read_csv("data/train.csv", na_values=["", " ", "NA", "None", "null"])
test_df = pd.read_csv("data/test.csv", na_values=["", " ", "NA", "None", "null"])

# 🔹 1. Identifier les valeurs manquantes et calculer leur pourcentage
def missing_values_info(df, name):
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100

    # Afficher uniquement les colonnes ayant des valeurs manquantes
    missing_df = pd.DataFrame({
        'Colonnes': df.columns,
        'Valeurs Manquantes': missing_count,
        'Pourcentage (%)': missing_percent
    })

    missing_df = missing_df[missing_df['Valeurs Manquantes'] > 0].sort_values(by='Pourcentage (%)', ascending=False)

    print(f"\n📌 Valeurs manquantes dans {name} :")
    if missing_df.empty:
        print("✅ Aucune valeur manquante détectée !")
    else:
        print(missing_df)

# Appliquer la fonction aux datasets
missing_values_info(train_df, "Train")
missing_values_info(test_df, "Test")

# Vérification des colonnes avec des NaN
train_nan_columns = train_df.columns[train_df.isnull().any()]
test_nan_columns = test_df.columns[test_df.isnull().any()]

print("\n📌 Colonnes avec des valeurs manquantes dans train_df :")
print(train_nan_columns)

print("\n📌 Colonnes avec des valeurs manquantes dans test_df :")
print(test_nan_columns)

# Uniformiser les noms de colonnes (minuscule et sans espaces)
train_df.columns = train_df.columns.str.lower().str.replace(" ", "")
test_df.columns = test_df.columns.str.lower().str.replace(" ", "")

# Vérifier les colonnes
print(train_df.columns)
print(test_df.columns)

train_df.columns = train_df.columns.str.lower().str.replace(' ', '')
test_df.columns = test_df.columns.str.lower().str.replace(' ', '')

print("\n🔍 Colonnes disponibles dans le dataset :")
print(train_df.columns.tolist())

# Séparation de la variable cible
X_train = train_df.drop(columns=['saleprice'])
y_train = train_df['saleprice']
X_test = test_df.copy()  # Le jeu de test ne contient pas 'SalePrice'

# Sélectionner les colonnes numériques
X_train_numeric = X_train.select_dtypes(include=['float64', 'int64'])
X_test_numeric = X_test.select_dtypes(include=['float64', 'int64'])

from sklearn.impute import KNNImputer

# Initialiser l'imputeur KNN
knn_imputer = KNNImputer(n_neighbors=5)

# Appliquer l'imputation sur les variables explicatives
X_train_imputed = pd.DataFrame(knn_imputer.fit_transform(X_train_numeric))
X_train_imputed.columns = X_train_numeric.columns

X_test_imputed = pd.DataFrame(knn_imputer.transform(X_test_numeric))
X_test_imputed.columns = X_test_numeric.columns

# Imputation des variables catégorielles
for col in X_train.select_dtypes(include=['object']).columns:
    mode_value = X_train[col].mode()[0]
    X_train_imputed[col] = X_train[col].fillna(mode_value)
    X_test_imputed[col] = X_test[col].fillna(mode_value)

# Réintégrer la variable cible
train_df_imputed = X_train_imputed.copy()
train_df_imputed['saleprice'] = y_train

# Appliquer la correction des valeurs manquantes pour les colonnes restantes
for col in train_df_imputed.columns:
    if col in X_test_imputed.columns:  # Vérifier que la colonne existe dans test_df
        if train_df_imputed[col].dtype == 'object':  # Catégorielle
            mode_value = train_df_imputed[col].mode()[0]
            train_df_imputed[col].fillna(mode_value, inplace=True)
            X_test_imputed[col].fillna(mode_value, inplace=True)
        else:  # Numérique
            median_value = train_df_imputed[col].median()
            train_df_imputed[col].fillna(median_value, inplace=True)
            X_test_imputed[col].fillna(median_value, inplace=True)

# Vérification après imputation
print(f"\n📝 Après imputation, train_df a {train_df.isnull().sum().sum()} NaN et test_df_imputed a {test_df.isnull().sum().sum()} NaN.")

# Pour le DataFrame d'entraînement
print(train_df_imputed.isnull().sum())

# Pour le DataFrame de test
print(X_test_imputed.isnull().sum())

print("\n🔍 Colonnes disponibles dans le dataset :")
print(train_df_imputed.columns.tolist())
# Delete existing files if they exist
if os.path.exists('data/train_clean.csv'):
    os.remove('data/train_clean.csv')
if os.path.exists('data/test_clean.csv'):
    os.remove('data/test_clean.csv')
# Save cleaned data
print("\n🔍 Saving cleaned training data to 'data/train_clean.csv'")
train_df.to_csv('data/train_clean.csv', index=False)
print("\n🔍 Saving cleaned test data to 'data/test_clean.csv'")
test_df.to_csv('data/test_clean.csv', index=False)