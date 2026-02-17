# 🚗 Analyse Avancée du Marché Automobile Européen

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Projet d'analyse de données et de machine learning appliqué au secteur automobile, avec une base de données de **50 000 véhicules** et des modèles prédictifs avancés.

## 📋 Table des Matières

- [Vue d'ensemble](#-vue-densemble)
- [Caractéristiques](#-caractéristiques)
- [Dataset](#-dataset)
- [Technologies](#-technologies)
- [Installation](#-installation)
- [Structure du Projet](#-structure-du-projet)
- [Utilisation](#-utilisation)
- [Résultats Clés](#-résultats-clés)
- [Méthodologie](#-méthodologie)
- [Visualisations](#-visualisations)
- [Contributeurs](#-contributeurs)

## 🎯 Vue d'ensemble

Ce projet analyse en profondeur le marché automobile européen (2015-2024) à travers :
- **Analyse exploratoire** des tendances du marché
- **Modèles prédictifs** (prix, type de carburant)
- **Segmentation de marché** par clustering
- **Visualisations interactives** avec Plotly

### Objectifs

1. Comprendre l'évolution de l'électrification du parc automobile
2. Prédire le prix des véhicules avec précision
3. Identifier les segments de marché distincts
4. Analyser l'impact environnemental (émissions CO2)

## ✨ Caractéristiques

### Analyses Réalisées

- ✅ **Analyse temporelle** : Évolution de la répartition des carburants (2015-2024)
- ✅ **Analyse de prix** : Segmentation premium vs généraliste
- ✅ **Émissions CO2** : Réduction des émissions et conformité réglementaire
- ✅ **Analyse géographique** : Comparaison entre 8 pays européens
- ✅ **Dépréciation** : Courbes de perte de valeur
- ✅ **Tests statistiques** : ANOVA, tests t, corrélations

### Modèles Machine Learning

- 🤖 **Régression** : Prédiction de prix (5 algorithmes comparés)
  - Linear Regression
  - Ridge Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost

- 🎯 **Classification** : Prédiction du type de carburant
  - Random Forest Classifier
  - Accuracy > 85%

- 📊 **Clustering** : Segmentation en 5 clusters de marché
  - K-Means
  - Méthode du coude pour optimisation

## 📊 Dataset

### Caractéristiques du Dataset

- **Taille** : 50 000 véhicules
- **Période** : 2015-2024
- **Géographie** : 8 pays européens
- **Variables** : 17 variables originales + features engineerées

### Variables Principales

| Variable | Type | Description |
|----------|------|-------------|
| `marque` | Catégorielle | 15 marques européennes |
| `modele` | Catégorielle | 100+ modèles différents |
| `annee` | Numérique | Année de fabrication (2015-2024) |
| `carburant` | Catégorielle | Essence, Diesel, Hybride, Électrique, Hybride rechargeable |
| `puissance_cv` | Numérique | Puissance en chevaux (65-400 CV) |
| `prix_euro` | Numérique | Prix en euros |
| `co2_g_km` | Numérique | Émissions CO2 en g/km |
| `kilometrage` | Numérique | Kilométrage du véhicule |
| `categorie` | Catégorielle | Citadine, Compacte, Berline, SUV, Break, Monospace |

### Features Engineerées

```python
# Exemples de features créées
- km_par_an : Kilométrage annuel moyen
- efficience : Ratio puissance/consommation
- prix_par_cv : Prix par cheval
- is_premium : Flag marque premium
- is_electric_or_hybrid : Flag véhicule électrifié
```

## 🛠️ Technologies

### Librairies Python

**Data Science**
- `pandas` - Manipulation de données
- `numpy` - Calculs numériques
- `scipy` - Statistiques avancées

**Visualisation**
- `matplotlib` - Graphiques statiques
- `seaborn` - Visualisations statistiques
- `plotly` - Graphiques interactifs

**Machine Learning**
- `scikit-learn` - Modèles ML classiques
- `xgboost` - Gradient boosting optimisé

**Environnement**
- `jupyter` - Notebooks interactifs
- `python 3.10+`

## 🚀 Installation

### Prérequis

- Python 3.10 ou supérieur
- pip ou conda

### Étapes d'installation

1. **Cloner le repository**
```bash
git clone https://github.com/votre-username/automobile-market-analysis.git
cd automobile-market-analysis
```

2. **Créer un environnement virtuel** (recommandé)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Lancer Jupyter Notebook**
```bash
jupyter notebook
```

5. **Ouvrir les notebooks** dans l'ordre :
   - `01_data_generation.ipynb`
   - `02_exploratory_analysis.ipynb`
   - `03_machine_learning.ipynb`

## 📁 Structure du Projet

```
automobile-market-analysis/
│
├── data/
│   ├── raw/                          # Données brutes
│   │   ├── vehicles_dataset.csv
│   │   └── vehicles_dataset.parquet
│   └── processed/                    # Données traitées
│       ├── vehicles_analyzed.csv
│       ├── vehicles_with_clusters.csv
│       └── country_analysis.csv
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_generation.ipynb     # Génération du dataset
│   ├── 02_exploratory_analysis.ipynb # Analyse exploratoire
│   └── 03_machine_learning.ipynb    # Modèles ML
│
├── models/                           # Modèles sauvegardés
│   ├── price_prediction_model.pkl
│   ├── fuel_type_classifier.pkl
│   └── market_segmentation_kmeans.pkl
│
├── src/                              # Code source Python
│   ├── data_processing.py
│   └── visualization.py
│
├── requirements.txt                  # Dépendances
├── README.md                         # Ce fichier
└── LICENSE                           # Licence MIT
```

## 💻 Utilisation

### Génération des Données

```python
# Dans le notebook 01_data_generation.ipynb
python
# Générer 50 000 véhicules avec des caractéristiques réalistes
df_vehicles = generate_vehicle_data(N_VEHICLES=50000)
```

### Analyse Exploratoire

```python
# Dans le notebook 02_exploratory_analysis.ipynb

# Analyser l'évolution de l'électrification
fuel_evolution = df.groupby(['annee', 'carburant']).size()

# Analyser les corrélations
corr_matrix = df[numerical_vars].corr()

# Tests statistiques
from scipy import stats
t_stat, p_value = stats.ttest_ind(premium_prices, generaliste_prices)
```

### Machine Learning

```python
# Dans le notebook 03_machine_learning.ipynb

# Prédiction de prix
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(X_scaled)
```

## 📈 Résultats Clés

### Tendances du Marché

- 📊 **Électrification** : +1200% de véhicules électriques entre 2015 et 2024
- 💰 **Prix moyen Premium** : 45 000€ vs 22 000€ pour généraliste
- 🌱 **Réduction CO2** : -28% d'émissions moyennes en 9 ans
- 🚗 **Catégorie dominante** : SUV (35% du marché en 2024)

### Performance des Modèles

#### Régression (Prédiction de Prix)
| Modèle | R² Score | RMSE | MAE |
|--------|----------|------|-----|
| XGBoost | 0.92 | 3 200€ | 2 100€ |
| Random Forest | 0.91 | 3 450€ | 2 300€ |
| Gradient Boosting | 0.90 | 3 600€ | 2 450€ |

#### Classification (Type de Carburant)
- **Accuracy** : 87.5%
- **F1-Score moyen** : 0.85
- **Meilleure prédiction** : Véhicules électriques (95% de précision)

#### Clustering (Segmentation)
Identification de 5 segments distincts :
1. **Premium Performance** (10%) - Véhicules haut de gamme puissants
2. **Eco/Électrique** (18%) - Véhicules à faibles émissions
3. **Entrée de Gamme** (25%) - Véhicules économiques
4. **Citadines Économiques** (22%) - Petits véhicules urbains
5. **Milieu de Gamme** (25%) - Véhicules polyvalents

## 🔬 Méthodologie

### 1. Génération et Préparation des Données

- Génération de 50 000 véhicules avec distributions réalistes
- Cohérence des données (prix, consommation, émissions)
- Gestion des valeurs manquantes : 0%
- Outliers : Détection et traitement par IQR

### 2. Feature Engineering

```python
# Création de features dérivées
df['km_par_an'] = df['kilometrage'] / df['age_vehicule']
df['efficience'] = df['puissance_cv'] / df['consommation']
df['prix_par_cv'] = df['prix_euro'] / df['puissance_cv']
```

### 3. Validation des Modèles

- **Split** : 80% train / 20% test
- **Validation croisée** : 5-fold CV
- **Métriques** : R², RMSE, MAE pour régression ; Accuracy, F1 pour classification
- **Feature Importance** : Analyse SHAP values

### 4. Optimisation

- Grid Search pour hyperparamètres
- Sélection des features par importance
- Ensemble methods pour robustesse

## 📸 Visualisations

Le projet inclut de nombreuses visualisations interactives :

### Évolution Temporelle
- Courbes d'évolution de la part de marché par carburant
- Tendances des émissions CO2
- Prix moyens par année

### Distributions
- Box plots des prix par marque
- Histogrammes des puissances
- Heatmaps de corrélation

### Machine Learning
- Scatter plots prédictions vs réalité
- Matrices de confusion
- Feature importance
- Visualisations 3D des clusters

### Interactivité
Toutes les visualisations Plotly sont **interactives** :
- Zoom, pan, rotation 3D
- Hover pour détails
- Filtrage dynamique
- Export en image

## 🤝 Contributeurs

- **Votre Nom** - *Développeur Principal* - [GitHub](https://github.com/votre-username)

## 📝 License

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- Données inspirées du marché automobile européen réel
- Méthodologie basée sur les best practices de Data Science
- Visualisations inspirées par les dashboards professionnels

## 📧 Contact

Pour toute question ou suggestion :
- **Email** : benouattara3@gmail.com
- **Portfolio** : https://benouattara.github.io

---

⭐ **Si ce projet vous plaît, n'hésitez pas à lui donner une étoile !** ⭐

Fait avec ❤️ et Python 🐍
