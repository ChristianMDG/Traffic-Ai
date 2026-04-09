import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(
    page_title="Traffic · Ai",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #E8E4DC;
}
.stApp { background: #0F0F0F; }
.block-container {
    padding: 2.5rem 3rem 4rem 3rem !important;
    max-width: 1400px;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Hero */
.hero {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid #222;
    margin-bottom: 2.5rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
    color: #F5F0E8;
    margin: 0;
}
.hero-title span { color: #FF5C2B; }
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #555;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.6rem;
}
.hero-badge {
    background: #1A1A1A;
    border: 1px solid #2A2A2A;
    border-radius: 8px;
    padding: 0.6rem 1.1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #666;
    letter-spacing: 0.06em;
    text-align: right;
}
.hero-badge strong { color: #FF5C2B; display: block; font-size: 0.85rem; }

/* KPI cards */
.kpi-row { display: flex; gap: 1rem; margin-bottom: 2.5rem; }
.kpi {
    flex: 1;
    background: #141414;
    border: 1px solid #1E1E1E;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi:hover { border-color: #333; }
.kpi::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, #FF5C2B);
}
.kpi-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #F5F0E8;
    line-height: 1;
}
.kpi-unit {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #444;
    margin-top: 0.35rem;
}

/* Section titles */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #E8E4DC;
    letter-spacing: -0.01em;
    margin: 0 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1E1E1E;
    margin-left: 0.8rem;
}

/* Prediction result */
.pred-result {
    background: #FF5C2B18;
    border: 1px solid #FF5C2B44;
    border-radius: 10px;
    padding: 1.2rem 1.6rem;
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #FF5C2B;
    text-align: center;
    margin-top: 1rem;
}

/* Button */
.stButton > button {
    background: #FF5C2B !important;
    color: #0F0F0F !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.65rem 1.8rem !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

hr { border-color: #1A1A1A !important; }
</style>
""", unsafe_allow_html=True)


BG     = "#141414"
FG     = "#E8E4DC"
GRID   = "#1E1E1E"
ACCENT = "#FF5C2B"
BLUE   = "#3B82F6"
GREEN  = "#22C55E"
PURPLE = "#A78BFA"
TEAL   = "#14B8A6"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   "#222",
    "axes.labelcolor":  "#666",
    "xtick.color":      "#555",
    "ytick.color":      "#555",
    "text.color":       FG,
    "grid.color":       GRID,
    "grid.linewidth":   0.6,
    "legend.facecolor": "#1A1A1A",
    "legend.edgecolor": "#2A2A2A",
    "legend.fontsize":  9,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "font.family":      "monospace",
})

@st.cache_data
def load_data():
    return pd.read_csv("./data/processed/traffic_features.csv")

@st.cache_resource
def load_model():
    return joblib.load("./models/traffic_model.pkl")

@st.cache_data
def compute_predictions(_model, df):
    df = df.copy()
    df = df.drop(columns=["date_time"], errors="ignore")
    y  = df["traffic_volume"].copy() if "traffic_volume" in df.columns else None

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df = df.reindex(columns=_model.feature_names_in_, fill_value=0)

    y_pred = _model.predict(df)
    return df, y, y_pred


try:
    df_raw       = load_data()
    model        = load_model()
    X, y, y_pred = compute_predictions(model, df_raw)
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.info("Vérifiez les chemins vers le CSV et le modèle .pkl")
    st.stop()

if y is None:
    st.error("❌ Colonne `traffic_volume` introuvable dans les données.")
    st.stop()

mae  = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2   = r2_score(y, y_pred)

# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <div>
    <p class="hero-title">Traffic <span>· Ai</span> </p>
    <p class="hero-sub">Metro Interstate Traffic · Random Forest Regressor</p>
  </div>
  <div class="hero-badge">
    <strong>{len(df_raw):,}</strong>
    observations · 20 % sample
  </div>
</div>
""", unsafe_allow_html=True)


st.markdown(f"""
<div class="kpi-row">
  <div class="kpi" style="--accent:#FF5C2B">
    <div class="kpi-label">R² Score</div>
    <div class="kpi-value">{r2:.3f}</div>
    <div class="kpi-unit">variance expliquée</div>
  </div>
  <div class="kpi" style="--accent:#3B82F6">
    <div class="kpi-label">MAE</div>
    <div class="kpi-value">{mae:,.0f}</div>
    <div class="kpi-unit">véhicules / heure</div>
  </div>
  <div class="kpi" style="--accent:#22C55E">
    <div class="kpi-label">RMSE</div>
    <div class="kpi-value">{rmse:,.0f}</div>
    <div class="kpi-unit">véhicules / heure</div>
  </div>
  <div class="kpi" style="--accent:#A78BFA">
    <div class="kpi-label">Estimateurs</div>
    <div class="kpi-value">20</div>
    <div class="kpi-unit">max_depth = 10</div>
  </div>
</div>
""", unsafe_allow_html=True)


st.markdown('<p class="section-title">📈 Analyse des prédictions</p>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1], gap="medium")

with col1:
    n   = min(600, len(y))
    idx = np.arange(n)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.fill_between(idx, y.values[:n], alpha=0.10, color=BLUE)
    ax.plot(idx, y.values[:n], color=BLUE,   linewidth=1.1, label="Réel",  alpha=0.9)
    ax.plot(idx, y_pred[:n],   color=ACCENT,  linewidth=1.1, label="Prévu", linestyle="--", alpha=0.9)
    ax.set_xlabel("Échantillons")
    ax.set_ylabel("Véhicules / h")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(loc="upper right")
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.yaxis.grid(True); ax.xaxis.grid(False)
    fig.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col2:
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.hist(y.values, bins=55, color=TEAL, edgecolor="none", alpha=0.85)
    ax.set_xlabel("Volume de trafic")
    ax.set_ylabel("Fréquence")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.yaxis.grid(True); ax.xaxis.grid(False)
    fig.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.markdown("<br>", unsafe_allow_html=True)


st.markdown('<p class="section-title">🔍 Importance & patterns horaires</p>', unsafe_allow_html=True)

col3, col4 = st.columns(2, gap="medium")

with col3:
    feat_imp = (
        pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    colors_fi = [ACCENT if i == 0 else PURPLE for i in range(len(feat_imp))]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.barh(feat_imp["feature"][::-1], feat_imp["importance"][::-1],
            color=colors_fi[::-1], height=0.65)
    ax.set_xlabel("Importance relative")
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.xaxis.grid(True); ax.yaxis.grid(False)
    fig.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col4:
    if "hour" in df_raw.columns and "traffic_volume" in df_raw.columns:
        hour_avg = df_raw.groupby("hour")["traffic_volume"].mean()
        peak_h   = hour_avg.idxmax()
        c_bars   = [ACCENT if h == peak_h else GREEN for h in hour_avg.index]

        fig, ax = plt.subplots(figsize=(6, 5.5))
        ax.bar(hour_avg.index, hour_avg.values, color=c_bars, width=0.75)
        ax.set_xlabel("Heure de la journée")
        ax.set_ylabel("Volume moyen")
        ax.set_xticks(range(0, 24, 2))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.yaxis.grid(True); ax.xaxis.grid(False)
        fig.tight_layout(pad=0.4)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Colonne `hour` non trouvée.")

st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<p class="section-title">📊 Résidus & volume hebdomadaire</p>', unsafe_allow_html=True)

col5, col6 = st.columns(2, gap="medium")

with col5:
    residuals = y.values - y_pred
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(residuals, bins=70, color=ACCENT, edgecolor="none", alpha=0.8)
    ax.axvline(0, color=FG, linestyle="--", linewidth=1, alpha=0.5, label="Zéro")
    ax.axvline(residuals.mean(), color=BLUE, linestyle=":", linewidth=1.2,
               alpha=0.9, label=f"Moyenne {residuals.mean():+.0f}")
    ax.set_xlabel("Erreur (Réel − Prévu)")
    ax.set_ylabel("Fréquence")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax.legend()
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.yaxis.grid(True); ax.xaxis.grid(False)
    fig.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col6:
    if "weekday" in df_raw.columns and "traffic_volume" in df_raw.columns:
        from matplotlib.patches import Patch
        day_labels = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"]
        day_avg    = df_raw.groupby("weekday")["traffic_volume"].mean()
        c_days     = [BLUE if d < 5 else ACCENT for d in day_avg.index]

        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.bar([day_labels[d] for d in day_avg.index], day_avg.values,
               color=c_days, width=0.6)
        ax.set_ylabel("Volume moyen")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
        ax.legend(handles=[Patch(color=BLUE, label="Semaine"),
                            Patch(color=ACCENT, label="Week-end")],
                  loc="lower right")
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.yaxis.grid(True); ax.xaxis.grid(False)
        fig.tight_layout(pad=0.4)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Colonne `weekday` non trouvée.")

st.markdown("<br>", unsafe_allow_html=True)


st.markdown('<p class="section-title">🔮 Simulateur de trafic</p>', unsafe_allow_html=True)

pm1, pm2, pm3 = st.columns(3, gap="large")

with pm1:
    st.markdown("**⏰ Temporel**")
    hour_input    = st.slider("Heure", 0, 23, 8, format="%dh")
    weekday_input = st.selectbox("Jour",
                                 ["Lundi","Mardi","Mercredi","Jeudi",
                                  "Vendredi","Samedi","Dimanche"])
    month_input   = st.slider("Mois", 1, 12, 6)

with pm2:
    st.markdown("**🌡️ Météo**")
    temp_input   = st.number_input("Température (K)", value=280.0, step=1.0,
                                    min_value=200.0, max_value=330.0)
    rain_input   = st.number_input("Pluie 1h (mm)", value=0.0, step=0.1, min_value=0.0)
    clouds_input = st.slider("Couverture nuageuse (%)", 0, 100, 20)

with pm3:
    st.markdown("**🚦 Résultat**")
    predict_btn = st.button("Estimer le volume ↗", use_container_width=True)

    if predict_btn:
        day_map     = {"Lundi":0,"Mardi":1,"Mercredi":2,"Jeudi":3,
                       "Vendredi":4,"Samedi":5,"Dimanche":6}
        weekday_val = day_map[weekday_input]
        is_weekend  = 1 if weekday_val >= 5 else 0

        input_df = pd.DataFrame([{
            "hour": hour_input, "weekday": weekday_val,
            "is_weekend": is_weekend, "month": month_input,
            "temp": temp_input, "rain_1h": rain_input,
            "clouds_all": clouds_input,
        }])
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        try:
            pred    = model.predict(input_df)[0]
            max_vol = int(df_raw["traffic_volume"].max()) if "traffic_volume" in df_raw.columns else 7000
            pct     = min(pred / max_vol, 1.0)
            level   = "🟢 Fluide" if pct < 0.4 else ("🟡 Modéré" if pct < 0.7 else "🔴 Dense")

            st.markdown(f'<div class="pred-result">{pred:,.0f} véh / h</div>',
                        unsafe_allow_html=True)
            st.progress(float(pct), text=f"{level}  ·  {pct*100:.0f} % de la capacité max")
        except Exception as e:
            st.error(f"Erreur : {e}")


st.markdown("""
<div style="text-align:center; font-family:'DM Mono',monospace; font-size:0.65rem;
            color:#333; padding: 2rem 0 0.5rem 0; border-top: 1px solid #1A1A1A; margin-top:2rem;">
  UrbanFlow · Metro Interstate Traffic Volume Dataset · Random Forest Regressor
</div>
""", unsafe_allow_html=True)