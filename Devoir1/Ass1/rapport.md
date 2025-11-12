# BLALET SALIMA
<img src="IMG_20251018_170628.jpg" style="height:464px;margin-right:432px"/>

# fetch dataset

student_performance = fetch_ucirepo(id=320)

# data (as pandas dataframes)

X = student_performance.data.features
y = student_performance.data.targets

# metadata

print(student_performance.metadata)

# variable information

print(student_performance.variables)

Voici une version améliorée du prompt pour obtenir un maximum d'informations détaillées sur la problématique du code, incluant la population étudiée et des statistiques descriptives :

```python
# Installation et importation de la bibliothèque ucimlrepo pour accéder aux datasets UCI
!pip install ucimlrepo
from ucimlrepo import fetch_ucirepo

# Chargement du dataset Student Performance (ID 320)
student_performance = fetch_ucirepo(id=320)

# Extraction des données
X = student_performance.data.features  # variables explicatives
y = student_performance.data.targets   # variables cibles

# Affichage des métadonnées du dataset, incluant population, contexte, et description
print("=== Métadonnées ===")
print(student_performance.metadata)

# Affichage des informations sur les variables (types, descriptions)
print("\n=== Description des Variables ===")
print(student_performance.variables)

# Statistiques descriptives des variables explicatives quantitatives
print("\n=== Statistiques descriptives des variables explicatives ===")
print(X.describe())

# Statistiques descriptives des variables cibles
print("\n=== Statistiques descriptives des variables cibles ===")
print(y.describe())

# Information sur la population étudiée (nombre d'exemples, caractéristiques démographiques principales)
print(f"\nNombre d'échantillons : {X.shape[^0]}, Nombre de variables : {X.shape[^1]}")
```



