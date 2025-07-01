**MLflow** est une plateforme open source qui facilite la gestion du cycle de vie des modèles de Machine Learning (_MLOps_). Il répond aux principaux problèmes rencontrés en Data Science et MLOps : difficulté à **reproduire** les expériences, à **suivre** les performances, à **standardiser** le déploiement et à **centraliser** la gestion des modèles.

L'un des grands atouts de MLflow est sa **compatibilité** avec de nombreux frameworks (Scikit-Learn, XGBoost, TensorFlow, etc.) et son API Python simple, facilitant son **intégration** rapide dans les projets existants. A noter qu'il est également **language-agnostic**, permettant de gérer des modèles écrits en Python, R, Java ou C++.

MLflow agit comme un **hub** centralisé pour logguer, versionner, reproduire et déployer des modèles. Il permet de conserver **l’historique** complet des expériences, incluant hyper-paramètres, artefacts, et métriques de performance.

La plateforme repose sur quatre composants clés :
- **MLflow Tracking** : suivi des expériences, paramètres, métriques et artefacts.
- **MLflow Projects** : standardisation et portabilité du code source.
- **MLflow Models** : format d'exportation et de déploiement des modèles.
- **MLflow Registry** : gestion centralisée des versions de modèles, facilitant leur validation, promotion et déploiement en production.
