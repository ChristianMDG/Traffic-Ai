
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Dashboard Trafic", layout="wide")


st.title("Dashboard de Prédiction du Trafic")
st.markdown("Visualisation des prédictions du modèle de trafic et analyse des features importantes")


@st.cache_data
def load_data():
    df = pd.read_csv("../data/processed/traffic_features.csv")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("../models/traffic_model.pkl")
    return model

df = load_data()
model = load_model()


cols_to_drop = ['traffic_volume', 'date_time']
cols_to_drop = [c for c in cols_to_drop if c in df.columns]

X = df.drop(cols_to_drop, axis=1)
y = df['traffic_volume']


y_pred = model.predict(X)

st.subheader("Métriques du modèle")
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

st.write(f"MAE : {mae:.2f}")
st.write(f"RMSE : {rmse:.2f}")
st.write(f"R² : {r2:.2f}")

st.subheader("Trafic réel vs prédit (500 premiers points)")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(y.values[:500], label="Réel")
ax.plot(y_pred[:500], label="Prévu")
ax.set_xlabel("Échantillons")
ax.set_ylabel("Volume de trafic")
ax.legend()
st.pyplot(fig)


st.subheader("Distribution des erreurs (résidus)")
residuals = y - y_pred
fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(residuals, bins=50, kde=True, ax=ax)
ax.set_xlabel("Erreur")
st.pyplot(fig)


st.subheader("Top 20 des features importantes")
importances = model.feature_importances_
features = X.columns
feat_imp = pd.DataFrame({'feature': features, 'importance': importances})
feat_imp = feat_imp.sort_values(by='importance', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=feat_imp, ax=ax)
st.pyplot(fig)


st.markdown("Dashboard prêt pour exploration et analyse.")