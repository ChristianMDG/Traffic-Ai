
# Traffic Volume Prediction – Smart City

## 1. Description du projet

Ce projet vise à prédire le **volume de trafic horaire** sur une autoroute à Washington en utilisant des données historiques de trafic et météo. Le but est d’anticiper les congestions et de fournir un outil d’aide à la gestion du trafic pour une **Smart City**.

Le projet comprend :

* Nettoyage et préparation des données
* Feature engineering (variables temporelles, météo, encodage)
* Entraînement d’un modèle de régression (Random Forest)
* Évaluation du modèle avec MAE, RMSE et R²
* Visualisation des résultats et dashboard interactif avec **Streamlit**

---

## 2. Problématique

Comment prédire efficacement le volume de trafic horaire afin de réduire les congestions et améliorer la planification urbaine ?

---

## 3. Justification

* La prédiction du trafic permet d’optimiser la circulation et de réduire la pollution.
* Les données historiques sont disponibles et fiables.
* Le projet illustre l’application concrète du Machine Learning dans une Smart City.

---

## 4. Données utilisées

* **Source** : [Metro Interstate Traffic Volume – Kaggle](https://www.kaggle.com/datasets/akhilv11/metro-interstate-traffic-volume)
* **Format** : CSV
* **Contenu** : Volume horaire du trafic, conditions météo, date/heure, jours, mois.

---

## 5. Variables clés

### A. Météo

* `temp`, `rain_1h`, `snow_1h`, `clouds_all`, `weather_main`, `weather_description`

### B. Temporalité

* `hour`, `day`, `month`, `weekday`, `is_weekend`

### C. Variables binaires

* Encodage des variables catégorielles :

  * `weather_main_Clear`, `weather_main_Rain`, `part_of_day_morning`, etc.

### D. Variable cible

* `traffic_volume` : nombre de véhicules par heure

---

## 6. Type de problème

* **Régression supervisée** (prédiction d’une variable continue)

---

## 7. Algorithme utilisé

* **Random Forest Regressor** (scikit-learn)

  * Captures non-linéarités
  * Robuste aux outliers
  * Fournit l’importance des features

---

## 8. Métriques d’évaluation

* MAE : Mean Absolute Error
* RMSE : Root Mean Squared Error
* R² : Coefficient de détermination

---

## 9. Résultats principaux

* MAE : ~420 véhicules
* RMSE : ~610 véhicules
* R² : 0.92
* Les heures de pointe et la météo sont les features les plus importantes.

---

## 10. Limites

* Événements rares non pris en compte (accidents, fermetures)
* Données locales à Washington
* Facteurs sociaux ou événements spéciaux non inclus

---

## 11. Organisation des fichiers

```
/data/raw/Metro_Interstate_Traffic_Volume.csv
/data/processed/traffic_features.csv
/models/traffic_model.pkl
/notebooks/traffic_prediction.ipynb
/dashboard/streamlit.py
requirements.txt
README.md
```

---

## 12. Installation

1. Cloner le dépôt :

```bash
git clone https://github.com/tonnom/traffic-smartcity.git
cd traffic-smartcity
```

2. Créer un environnement virtuel :

```bash
python -m venv venv
```

3. Activer l’environnement :

* Windows : `venv\Scripts\activate`
* Linux/Mac : `source venv/bin/activate`

4. Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

## 13. Utilisation

### Jupyter Notebook

```bash
jupyter notebook notebooks/traffic_prediction.ipynb
```

* Permet de visualiser le nettoyage, le feature engineering, l’entraînement et l’évaluation du modèle.

### Dashboard Streamlit

```bash
streamlit run dashboard/streamlit.py
```

* Permet de tester les prédictions en temps réel et explorer les résultats graphiques.

---

## 14. Rapport synthétique du modèle

* Contient l’objectif, la méthode, les résultats, les limites et les conclusions.
* Lien PDF : `traffic_model_report.pdf`

---

## 15. Auteurs

* Christian RAVELOJAONA / christianravelojaona186@gmail.com

---

## 16. Remarques

* Projet développé pour l’évaluation d’un projet Data Science type Smart City.
* Les prédictions peuvent être améliorées avec plus de données, événements spéciaux, ou modèles avancés (XGBoost, LSTM).
