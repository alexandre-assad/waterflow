# Waterflow

> Un projet de classification de la qualité de l'eau avec une approche MLOps basée sur MLflow.

## Contexte

L'accès à l'eau potable est un enjeu sanitaire et sociétal majeur. Ce projet vise à développer un outil de prédiction de la potabilité de l'eau à partir de 9 mesures physico-chimiques. Il s'inscrit dans une démarche MLOps, pour industrialiser et fiabiliser le déploiement de modèles de machine learning.

## Objectifs

- Développer un modèle de classification binaire prédictif.
- Suivre et gérer le cycle de vie du modèle avec MLflow.
- Créer une API Flask de prédiction en temps réel.
- Mettre en place des tests logiciels pour fiabiliser l’application.

## Stack technique

- Python 3.12 (géré avec [uv](https://github.com/astral-sh/uv))
- Scikit-learn / XGBoost
- MLflow
- Flask
- pytest

---

## Lancer le projet

### Pré-requis

```bash
pip install uv
uv python install 3.12
```

### Lancement

```bash
uv run mlflow ui                   # Lancer l'interface MLflow
uv run ./main.py                  # Entrée principale du projet
```

### Cas d’usage spécifiques

- **Suivi d’expérience MLflow** :
  ```bash
  uv run ./waterflow/experiment.py
  ```
- **Application Flask (interface de prédiction)** :
  ```bash
  uv run ./waterflow/app.py
  ```
- **Tests** :
  ```bash
  uv run pytest -v
  ```
- **Définir les modèles en production dans le registre MLflow** :
  ```bash
  uv run ./waterflow/ops/set_production_models.py
  ```

---

## Structure du projet

```
waterflow/
├── app.py                      # Application Flask
├── experiment.py              # Tracking d’expérience MLflow
├── ops/
│   └── set_production_models.py
├── tests/                     # Tests unitaires et fonctionnels
│   ├── test_app.py
│   └── test_model.py
├── data/                      # Données brutes et nettoyées
├── notebooks/                 # Analyse exploratoire, modélisation
└── ...
```

---

## Données

Les données utilisées proviennent d’un dataset public contenant 3276 observations de la qualité de l’eau selon 9 caractéristiques physico-chimiques :

- ph
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic_carbon
- Trihalomethanes
- Turbidity

---

## Démarche MLOps

Ce projet suit une logique MLOps complète, incluant :

1. **Exploration et préparation des données**
2. **Suivi des expériences** avec MLflow Tracking
3. **Versioning et enregistrement des modèles** avec le Model Registry
4. **Déploiement dans une API Flask**
5. **Mise en production des modèles**
6. **Tests logiciels** : unitaires, fonctionnels, non-régression


---

## Veille technologique

Une revue complète sur les concepts de MLOps et MLflow est disponible dans le dossier `docs/`.

Extraits :
- MLOps est une discipline visant à industrialiser les projets de machine learning.
- MLflow est un outil open-source qui centralise la gestion du cycle de vie des modèles.
- L’approche adoptée permet de suivre les performances, versionner les modèles, automatiser les déploiements et superviser leur comportement en production.

---

## Résultat

L’application permet de :

- Prédire si une eau est potable ou non.
- Visualiser les expériences depuis l’interface MLflow.
- Déployer en local un service de prédiction.

---

## Tests

Les tests implémentés couvrent :

- Le bon fonctionnement de l’API (tests fonctionnels)
- Le comportement du modèle (tests unitaires)
- La non-régression entre les versions

Exécution :

```bash
uv run pytest -v
```

---

## Auteurs

Projet réalisé dans le cadre d’un module sur les **Machine Learning Operations**.