# NOM:BLALET Prénom:SALIMA
# Filière:contrôle,audit et conseil

# Rapport descriptif du code de l’analyse Machine Learning sur la qualité des vins

## Introduction

Ce notebook vise à modéliser la qualité de vins portugais à partir de variables physico-chimiques. Pour cela, le jeu de données "Wine Quality" de l’UCI Machine Learning Repository, détaillant différentes mesures de vins rouges et blancs, sert de base à une analyse et à une prédiction de la qualité selon plusieurs algorithmes de Machine Learning.[1]

## Bibliothèques utilisées

Les principales bibliothèques utilisées sont :
- **pandas** et **numpy** pour la manipulation des données.
- **matplotlib** et **seaborn** pour la visualisation.
- **sklearn** (scikit-learn) pour l’implémentation des algorithmes de Machine Learning (séparation train/test, KNeighborsClassifier, accuracy_score, etc.).
- **ucimlrepo** pour la récupération structurée du jeu de données depuis l’UCI ML Repository.[1]

## Chargement et exploration du jeu de données

Le dataset est chargé depuis une source en ligne, puis converti en DataFrame pandas. Les informations principales (dimensions, types de variables, présence de valeurs manquantes) sont affichées ainsi qu’un aperçu des cinq premières lignes. La variable cible "quality" (qualité du vin) est étudiée par comptage des occurrences pour chaque note.[1]

- 4898 échantillons, 11 variables physico-chimiques + 1 qualitative (qualité).
- Pas de valeurs manquantes détectées.
- Répartition déséquilibrée des classes qualité.

## Prétraitement et visualisation

Une analyse de corrélation est conduite : création d’une matrice de corrélation affichée en heatmap pour visualiser les relations entre variables et cible. Cette étape permet d’identifier les variables les plus pertinentes pour prédire la qualité du vin. Les données sont séparées en variables explicatives \( X \) et cible \( Y \), puis le jeu de données est découpé en sous-ensembles d’entraînement, de validation et de test de manière stratifiée.[1]

## Pipeline d’apprentissage : K-Nearest Neighbors (KNN)

- Utilisation de l’algorithme des k plus proches voisins (**KNeighborsClassifier**) pour une tâche de classification supervisée.
- Optimisation de l’hyperparamètre principal \( k \) (nombre de voisins) via une validation croisée sur la validation set.
- Évaluation des performances par le calcul de l’erreur et de l’accuracy à chaque valeur de \( k \), représentation graphique de l’erreur selon \( k \).[1]

### Exemple de pipeline :
```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Séparation train/validation/test
Xa, Xt, Ya, Yt = train_test_split(X, Y, test_size=1/3, stratify=Y)
Xa, Xv, Ya, Yv = train_test_split(Xa, Ya, test_size=0.5, stratify=Ya)

# Entraînement et évaluation
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(Xa, Ya)
Ypredv = clf.predict(Xv)
erreur_v = 1 - accuracy_score(Yv, Ypredv)
```

## Résultats principaux

- La matrice de corrélation montre des liens forts entre certaines variables et la qualité, notamment l’alcool.
- Les performances du modèle KNN varient en fonction de \( k \), une valeur optimale est recherchée en minimisant l’erreur sur la validation.
- Rapport des principaux scores d’accuracy obtenus, discussion succincte.[1]

## Conclusion

Le notebook démontre, étape par étape, la préparation, l’approche et l’évaluation d’un modèle de classification supervisée adapté à la prédiction de la qualité des vins, avec un accent sur la rigueur de la validation par découpe des données et par utilisation de métriques pertinentes de performance. Les graphiques et analyses intermédiaires justifient les choix méthodologiques et offrent une compréhension globale de l’efficacité du pipeline retenu.[1]

***



[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/128828236/638ad1f5-b2cf-4ea9-9cbd-3da9bbbfc19a/ML.ipynb)
