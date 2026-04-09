import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(
    page_title="Dashboard Trafic I-94",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background: #f5f5f2;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("../data/processed/traffic_features.csv")
    return df

@st.cache_resource
def load_model():
    return joblib.load("../models/traffic_model.pkl")

@st.cache_data
def compute_predictions(_model, df):
    df = df.copy()
    df = df.drop(columns=["date_time"], errors="ignore")

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    model_features = _model.feature_names_in_
    df = df.reindex(columns=model_features, fill_value=0)

    X = df.drop(columns=["traffic_volume"], errors="ignore")
    y = df["traffic_volume"] if "traffic_volume" in df.columns else None

    y_pred = _model.predict(X)
    return X, y, y_pred


try:
    df_raw = load_data()
    model  = load_model()
    X, y, y_pred = compute_predictions(model, df_raw)
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"Erreur de chargement : {e}")
    st.info("Vérifiez les chemins `../data/processed/traffic_features.csv` et `../models/traffic_model.pkl`")
    st.stop()


mae  = mean_absolute_error(y, y_pred)
mse  = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y, y_pred)


st.title("🚗 Trafic autoroutier I-94 — Tableau de bord")
st.caption("Random Forest · 20% sample · n_estimators=20 · max_depth=10")
st.divider()


c1, c2, c3, c4 = st.columns(4)
c1.metric("R²",     f"{r2:.3f}",          "score de prédiction")
c2.metric("MAE",    f"{mae:,.0f}",         "véhicules / h")
c3.metric("RMSE",   f"{rmse:,.0f}",        "véhicules / h")
c4.metric("Dataset", f"{len(df_raw):,}",   "lignes totales")

st.divider()

col1, col2 = st.columns([1.6, 1])

with col1:
    st.subheader("Prédictions vs Réel")
    n = min(500, len(y))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y.values[:n],    label="Réel",  color="#378ADD", linewidth=1.2)
    ax.plot(y_pred[:n],      label="Prévu", color="#D85A30", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Échantillons")
    ax.set_ylabel("Volume de trafic")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Distribution du trafic")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(y.values, bins=50, color="#5DCAA5", edgecolor="none")
    ax.set_xlabel("Traffic Volume")
    ax.set_ylabel("Fréquence")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.divider()


col3, col4 = st.columns(2)

with col3:
    st.subheader("Top 20 features importantes")
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature":    X.columns,
        "importance": importances
    }).sort_values("importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.barh(feat_imp["feature"][::-1], feat_imp["importance"][::-1], color="#7F77DD")
    ax.set_xlabel("Importance")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col4:
    st.subheader("Volume moyen par heure")

    if "hour" in df_raw.columns and "traffic_volume" in df_raw.columns:
        hour_avg = df_raw.groupby("hour")["traffic_volume"].mean().reset_index()

        fig, ax = plt.subplots(figsize=(6, 7))
        ax.bar(hour_avg["hour"], hour_avg["traffic_volume"], color="#1D9E75")
        ax.set_xlabel("Heure")
        ax.set_ylabel("Volume moyen")
        ax.set_xticks(range(0, 24))
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Colonne `hour` non trouvée dans les données.")

st.divider()


col5, col6 = st.columns(2)

with col5:
    st.subheader("Distribution des résidus")
    residuals = y.values - y_pred

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=60, color="#D85A30", edgecolor="none", alpha=0.85)
    ax.axvline(0, color="#1a1a1a", linestyle="--", linewidth=1)
    ax.set_xlabel("Erreur (Réel − Prévu)")
    ax.set_ylabel("Fréquence")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col6:
    st.subheader("Volume moyen par jour de la semaine")

    if "weekday" in df_raw.columns and "traffic_volume" in df_raw.columns:
        day_labels = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
        day_avg = df_raw.groupby("weekday")["traffic_volume"].mean().reset_index()
        day_avg["jour"] = day_avg["weekday"].apply(lambda x: day_labels[x] if x < 7 else str(x))

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#378ADD" if d < 5 else "#D85A30" for d in day_avg["weekday"]]
        ax.bar(day_avg["jour"], day_avg["traffic_volume"], color=colors)
        ax.set_xlabel("Jour")
        ax.set_ylabel("Volume moyen")
        ax.tick_params(axis="x", rotation=30)
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Colonne `weekday` non trouvée dans les données.")

st.divider()


st.subheader("🔮 Prédiction manuelle")
st.caption("Entrez des valeurs pour obtenir une prédiction du volume de trafic.")

pm1, pm2, pm3 = st.columns(3)
with pm1:
    hour_input    = st.slider("Heure", 0, 23, 8)
    weekday_input = st.selectbox("Jour", ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"])
with pm2:
    month_input   = st.slider("Mois", 1, 12, 6)
    temp_input    = st.number_input("Température (K)", value=280.0, step=1.0)
with pm3:
    rain_input    = st.number_input("Pluie 1h (mm)", value=0.0, step=0.1)
    clouds_input  = st.slider("Nuages (%)", 0, 100, 20)

if st.button("Prédire le volume de trafic ↗"):
    day_map = {"Lundi":0,"Mardi":1,"Mercredi":2,"Jeudi":3,"Vendredi":4,"Samedi":5,"Dimanche":6}
    weekday_val = day_map[weekday_input]
    is_weekend  = 1 if weekday_val >= 5 else 0

    input_dict = {
        "hour":      hour_input,
        "weekday":   weekday_val,
        "is_weekend": is_weekend,
        "month":     month_input,
        "temp":      temp_input,
        "rain_1h":   rain_input,
        "clouds_all": clouds_input,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Volume de trafic prévu : **{prediction:,.0f} véhicules / heure**")
    except Exception as e:
        st.error(f"Erreur de prédiction : {e}")


st.caption("Dashboard · Metro Interstate Traffic Volume · Random Forest Regressor")