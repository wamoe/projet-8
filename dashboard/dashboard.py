import os
import time
from typing import List, Tuple, Dict

import streamlit as st
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# =============================
# CONFIGURATION
# =============================
LOCAL_API_BASE = "http://127.0.0.1:5001"
RENDER_API_BASE = "https://projet-8-gyq1.onrender.com"
DEFAULT_API_BASE = os.environ.get("API_BASE_URL", LOCAL_API_BASE)

# Critère WCAG 2.4.2 : Titre de page descriptif
st.set_page_config(
    page_title="Scoring Crédit - Prêt à dépenser",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# CSS (Design Premium + Accessibilité WCAG)
# =============================
st.markdown("""
<style>
:root{
  --bg: #ffffff;
  --panel: #f8fafc;
  --panel2: #ffffff;
  --border: #e2e8f0;
  --text: #0f172a;
  --muted: #475569; /* WCAG 1.4.3: Contraste OK */
  --primary: #2563EB;
  --success: #15803D; /* WCAG 1.4.3: Vert sombre OK */
  --danger:  #DC2626;
  --warn:    #D97706;
  --slate:   #0F172A;
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
}

/* WCAG 1.1.1 : Lecteur d'écran */
.sr-only {
  position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; 
  overflow: hidden; clip: rect(0, 0, 0, 0); border: 0;
}

/* Typographie et espacements (WCAG 1.4.4 rem) */
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.02em; font-weight: 800; color: var(--slate); }
.small { color: var(--muted); font-size: 0.92rem; }
.kicker { font-size: 0.8rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; color: var(--primary); }

/* Cartes Globales */
.card {
  background: var(--bg); border: 1px solid var(--border); border-radius: 16px; 
  padding: 1.5rem; box-shadow: var(--shadow-sm); transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
.hr { height: 1px; background: var(--border); border: none; margin: 1rem 0; }
.rowgap { margin-top: 1.5rem; }

/* Badges / Pills */
.pill {
  display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.35rem 0.8rem; 
  border-radius: 999px; border: 1px solid var(--border); background: #f1f5f9; 
  font-weight: 700; font-size: 0.85rem; color: var(--slate);
}
.pill-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; background: var(--primary); }
.pill-ok { background: #dcfce7; border-color: #bbf7d0; color: #166534; }
.pill-ok .pill-dot { background: var(--success); }
.pill-ko { background: #fee2e2; border-color: #fecaca; color: #991b1b; }
.pill-ko .pill-dot { background: var(--danger); }

/* KPI (Chiffres clés) */
.kpi {
  border: 1px solid var(--border); border-radius: 16px; padding: 1.2rem; 
  background: var(--bg); box-shadow: var(--shadow-sm); text-align: center;
}
.kpi .t { font-weight: 700; color: var(--muted); font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi .v { font-weight: 900; font-size: 1.8rem; color: var(--slate); margin: 0.5rem 0; }
.kpi .s { color: var(--muted); font-size: 0.85rem; }

/* Nouvelles cartes "Profil Client" (Design amélioré) */
.profile-card {
  background: var(--panel); border: 1px solid var(--border); border-radius: 16px; 
  padding: 1.2rem; box-shadow: var(--shadow-sm); height: 100%;
}
.profile-head {
  display: flex; align-items: center; justify-content: space-between; gap: 1rem; 
  padding-bottom: 1rem; border-bottom: 2px dashed var(--border); margin-bottom: 1rem;
}
.profile-title-container { display: flex; align-items: center; gap: 0.8rem; }
.profile-icon { font-size: 1.8rem; background: #ffffff; padding: 0.5rem; border-radius: 12px; border: 1px solid var(--border); box-shadow: var(--shadow-sm); }
.profile-title { font-weight: 800; color: var(--slate); font-size: 1.1rem; }
.profile-sub { color: var(--muted); font-size: 0.85rem; margin-top: 0.2rem; }
.profile-tag {
  background: var(--primary); color: white; padding: 0.2rem 0.6rem; 
  border-radius: 999px; font-weight: 700; font-size: 0.75rem;
}
.profile-grid { display: flex; flex-direction: column; gap: 0.6rem; }
.profile-row {
  display: flex; justify-content: space-between; align-items: center; padding: 0.7rem 1rem; 
  border-radius: 10px; border: 1px solid var(--border); background: var(--panel2);
}
.profile-k { font-weight: 600; font-size: 0.9rem; color: var(--muted); }
.profile-v { font-weight: 800; font-size: 1rem; color: var(--slate); text-align: right; }

/* Tweaks Streamlit */
.stButton button { border-radius: 12px !important; padding: 0.6rem 1.2rem !important; font-weight: 700 !important; transition: all 0.2s; }
.stButton button:hover { transform: translateY(-1px); box-shadow: var(--shadow-sm); }
[data-testid="stDataFrame"] { border-radius: 12px; border: 1px solid var(--border); box-shadow: var(--shadow-sm); }

/* Mode Sombre */
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f172a; --panel: #1e293b; --panel2: #0f172a; --border: #334155; 
    --text: #f8fafc; --muted: #94a3b8; --slate: #f8fafc;
  }
  .card, .profile-card, .kpi { background: var(--panel); border-color: var(--border); box-shadow: none; }
  .profile-icon { background: var(--bg); border-color: var(--border); }
  .pill { background: #334155; border-color: #475569; color: #f8fafc; }
  .pill-ok { background: #064e3b; border-color: #065f46; color: #a7f3d0; }
  .pill-ko { background: #7f1d1d; border-color: #991b1b; color: #fecaca; }
}
</style>
""", unsafe_allow_html=True)


# =============================
# FONCTIONS UTILITAIRES
# =============================
def normalize_predict_url(base_url: str) -> str:
    url = (base_url or "").strip().rstrip("/")
    if not url: return LOCAL_API_BASE + "/predict"
    return url if url.endswith("/predict") else url + "/predict"

def explain_url_from_predict(predict_url: str) -> str:
    u = predict_url.strip().rstrip("/")
    base = u[:-len("/predict")] if u.endswith("/predict") else u
    return base + "/explain"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# Dictionnaire de traduction des features
LABELS_FR = {
    "CNT_CHILDREN": "Nombre d’enfants", "CNT_FAM_MEMBERS": "Taille du foyer", "CODE_GENDER": "Genre",
    "NAME_FAMILY_STATUS": "Statut marital", "NAME_INCOME_TYPE": "Type de revenu", "OCCUPATION_TYPE": "Profession",
    "NAME_CONTRACT_TYPE": "Type de contrat", "AMT_INCOME_TOTAL": "Revenu annuel", "AMT_CREDIT": "Montant du crédit",
    "AMT_ANNUITY": "Mensualité (annuité)", "AMT_GOODS_PRICE": "Prix du bien",
    "REGION_POPULATION_RELATIVE": "Densité régionale", "DAYS_BIRTH": "Âge",
    "DAYS_EMPLOYED": "Ancienneté emploi", "DAYS_REGISTRATION": "Ancienneté dossier", "DAYS_ID_PUBLISH": "Ancienneté pièce ID",
}
MONEY_COLS = {"AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"}

def fmt_number(v):
    if pd.isna(v) or v is None: return "—"
    if isinstance(v, (int, np.integer)): return f"{int(v):,}".replace(",", " ")
    if isinstance(v, (float, np.floating)):
        return f"{v:,.2f}".replace(",", " ").replace(".", ",") if abs(v) < 1000 else f"{v:,.0f}".replace(",", " ")
    return str(v)

def pretty_label(col: str) -> str:
    return LABELS_FR.get(col, col.replace("_", " ").title())

def pretty_value(col: str, v):
    if pd.isna(v) or v is None: return "—"
    if col.startswith("DAYS_"):
        years = abs(float(v)) / 365.25
        return f"{years:.1f} ans".replace(".", ",")
    if col in MONEY_COLS: 
        return f"{fmt_number(v)} €"
    if col == "REGION_POPULATION_RELATIVE":
        return f"{float(v)*100:.2f} %".replace(".", ",")
    if col.startswith("CNT_"):
        return f"{int(round(float(v)))}"
    return fmt_number(v)

def clean_record_for_json(record: dict):
    out = {}
    for k, v in record.items():
        if isinstance(v, (float, np.floating)): out[k] = None if np.isnan(v) or np.isinf(v) else float(v)
        elif isinstance(v, (int, np.integer)): out[k] = int(v)
        else: out[k] = None if pd.isna(v) else v
    return out

# =============================
# UI COMPONENTS (HTML)
# =============================
def pill(label: str, kind: str = "default"):
    cls = f"pill pill-{kind}" if kind in ["ok", "ko", "warn"] else "pill"
    return f"<span class='{cls}'><span class='pill-dot'></span>{label}</span>"

def kpi(title: str, value: str, sub: str):
    st.markdown(f"<div class='kpi'><div class='t'>{title}</div><div class='v'>{value}</div><div class='s'>{sub}</div></div>", unsafe_allow_html=True)

def profile_card_html(title: str, subtitle: str, icon: str, items: List[Tuple[str, str]]) -> str:
    count = len(items)
    tag = f"<span class='profile-tag'>{count} champ{'s' if count>1 else ''}</span>" if count > 0 else ""
    rows = "".join([f"<div class='profile-row'><div class='profile-k'>{k}</div><div class='profile-v'>{v}</div></div>" for k, v in items])
    if not rows: rows = "<div class='small' style='padding:1rem; text-align:center;'>Aucune donnée disponible.</div>"
    
    return f"""
    <div class="profile-card">
      <div class="profile-head">
        <div class="profile-title-container">
          <div class="profile-icon">{icon}</div>
          <div><div class="profile-title">{title}</div><div class="profile-sub">{subtitle}</div></div>
        </div>
        {tag}
      </div>
      <div class="profile-grid">{rows}</div>
    </div>
    """

def bullet_risk(prob: float, threshold: float):
    prob, threshold = float(np.clip(prob, 0, 1)), float(np.clip(threshold, 0, 1))
    color = "#DC2626" if prob >= threshold else "#15803D"
    
    fig, ax = plt.subplots(figsize=(9, 2))
    ax.barh([0], [1.0], height=0.3, color="#f1f5f9", edgecolor="#cbd5e1") # Fond clair
    ax.barh([0], [prob], height=0.3, color=color)
    ax.axvline(threshold, linestyle="--", color="#0f172a", linewidth=2.5, zorder=5)
    
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontweight='bold', color="#475569")
    ax.set_xlabel("Probabilité de défaut estimée", fontweight='bold', color="#0f172a", labelpad=10)
    for spine in ax.spines.values(): spine.set_visible(False)
    
    st.markdown("<div class='sr-only'>Graphique jauge montrant la probabilité de défaut estimée par rapport au seuil d'acceptation.</div>", unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)

# =============================
# BARRE LATÉRALE (SIDEBAR)
# =============================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bank-building.png", width=60) # Petit logo
    st.header("Paramètres")
    st.caption("Configuration & sélection")
    
    st.subheader("🌐 Connexion API")
    api_mode = st.radio("Mode de déploiement", ["Local", "Render", "Custom"], index=0, horizontal=True)
    api_base = LOCAL_API_BASE if api_mode == "Local" else (RENDER_API_BASE if api_mode == "Render" else st.text_input("URL", value=DEFAULT_API_BASE))
    api_url = normalize_predict_url(api_base)
    
    if st.button("🔌 Tester l'API", use_container_width=True):
        try:
            t0 = time.time()
            base_url = api_url[:-len("/predict")] if api_url.endswith("/predict") else api_url
            r = requests.get(base_url + "/", timeout=10)
            if r.status_code == 200:
                st.success(f"Connecté en {time.time()-t0:.2f}s")
            else:
                st.warning(f"Erreur HTTP {r.status_code}")
        except Exception as e:
            st.error(f"Inaccessible: {str(e)[:50]}...")

    st.divider()
    st.subheader("📂 Données")
    data_path = st.text_input("Chemin du dataset", value="model_production/test_sample_processed.csv")
    
    try:
        df = load_data(data_path)
    except Exception as e:
        st.error(f"Erreur de chargement des données: {e}")
        st.stop()

    st.divider()
    st.subheader("👤 Sélection Client")
    if "SK_ID_CURR" in df.columns:
        client_id = st.selectbox("ID Client (SK_ID_CURR)", df["SK_ID_CURR"].tolist())
        client_row_df = df[df["SK_ID_CURR"] == client_id]
    else:
        client_id = st.selectbox("Index Client", df.index)
        client_row_df = df.loc[[client_id]]

    row = client_row_df.iloc[0]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    st.divider()
    call_api = st.button("🚀 Calculer le Score", use_container_width=True, type="primary")

# =============================
# APPELS API
# =============================
if "api_result" not in st.session_state: st.session_state.api_result = None
if "api_error" not in st.session_state: st.session_state.api_error = None

if call_api:
    payload = [clean_record_for_json(client_row_df.to_dict(orient="records")[0])]
    try:
        with st.spinner("Analyse du dossier en cours..."):
            t0 = time.time()
            r = requests.post(api_url, json=payload, timeout=60)
            if r.status_code == 200:
                st.session_state.api_result = r.json()
                st.session_state.api_result["_latency_s"] = time.time() - t0
                st.session_state.api_error = None
            else:
                st.session_state.api_result = None
                st.session_state.api_error = f"Erreur API {r.status_code}"
    except Exception as e:
        st.session_state.api_result = None
        st.session_state.api_error = f"Connexion impossible"

res = st.session_state.api_result
err = st.session_state.api_error

# =============================
# EN-TÊTE PRINCIPAL (HERO)
# =============================
col_title, col_badges = st.columns([2, 1])
with col_title:
    st.title("Tableau de bord - Octroi de Crédit")
    st.markdown(f"**Client sélectionné :** `{client_id}` | **Endpoint :** `{api_url}`")
with col_badges:
    st.write("") # Espacement
    if err: st.markdown(pill("API Injoignable", "ko"), unsafe_allow_html=True)
    elif res: st.markdown(pill("API Connectée", "ok") + " " + pill(f"{res.get('_latency_s', 0):.2f}s", "default"), unsafe_allow_html=True)
    else: st.markdown(pill("En attente d'analyse", "warn"), unsafe_allow_html=True)

if err:
    st.error(err)
    st.stop()

if res is None:
    st.info("👈 Veuillez cliquer sur **Calculer le Score** dans le menu latéral pour analyser ce dossier.")
    st.stop()

# =============================
# RÉSULTATS DU SCORING
# =============================
proba = float(res.get("probability", 0.0))
threshold = float(res.get("threshold", 0.5))
status = res.get("status", "—")
prediction = int(res.get("prediction", 0))
is_refused = (prediction == 1) or (str(status).strip().lower() == "refusé")

st.markdown("<div class='rowgap'></div>", unsafe_allow_html=True)

col_decision, col_risk = st.columns([1, 1.5])
with col_decision:
    badge = pill("DÉCISION : REFUSÉ", "ko") if is_refused else pill("DÉCISION : ACCORDÉ", "ok")
    st.markdown(f"""
    <div class="card" style="height: 100%;">
      <div class="kicker">Résultat de l'algorithme</div>
      <div style="margin-top: 1rem; margin-bottom: 1.5rem;">{badge}</div>
      <div class="small">Classe prédite : <b>{prediction}</b></div>
      <div class="small">Proba de défaut : <b>{proba:.2%}</b></div>
      <div class="small">Seuil limite : <b>{threshold:.2f}</b></div>
    </div>
    """, unsafe_allow_html=True)

with col_risk:
    st.markdown("<div class='card' style='height: 100%;'><div class='kicker'>Positionnement du risque</div>", unsafe_allow_html=True)
    bullet_risk(proba, threshold)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='rowgap'></div>", unsafe_allow_html=True)

# =============================
# PROFIL CLIENT (Design amélioré)
# =============================
st.markdown("## 👤 Profil du client")

GROUPS = {
    "Identité & ménage": ["CNT_CHILDREN", "CNT_FAM_MEMBERS"],
    "Revenus & emploi":  ["AMT_INCOME_TOTAL", "DAYS_EMPLOYED"],
    "Crédit":           ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"],
    "Région & historique": ["REGION_POPULATION_RELATIVE", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"]
}

def pick_items(cols: List[str]) -> List[Tuple[str, str]]:
    items = []
    for c in cols:
        if c in client_row_df.columns:
            items.append((pretty_label(c), pretty_value(c, row.get(c, None))))
    return items

col_p1, col_p2 = st.columns(2)
col_p3, col_p4 = st.columns(2)

with col_p1:
    st.markdown(profile_card_html("Identité & ménage", "Situation familiale", "🧑‍🤝‍🧑", pick_items(GROUPS["Identité & ménage"])), unsafe_allow_html=True)
with col_p2:
    st.markdown(profile_card_html("Revenus & emploi", "Stabilité financière", "💼", pick_items(GROUPS["Revenus & emploi"])), unsafe_allow_html=True)
with col_p3:
    st.markdown(profile_card_html("Crédit", "Détails de la demande", "💳", pick_items(GROUPS["Crédit"])), unsafe_allow_html=True)
with col_p4:
    st.markdown(profile_card_html("Région & historique", "Ancienneté et localisation", "📍", pick_items(GROUPS["Région & historique"])), unsafe_allow_html=True)

st.markdown("<div class='rowgap'></div>", unsafe_allow_html=True)

# =============================
# ANALYSE & EXPLICABILITÉ (Onglets avec Emojis)
# =============================
st.markdown("## 🔍 Analyse détaillée")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Explication Modèle", 
    "📈 Comparaison Client", 
    "🔀 Analyse Bivariée", 
    "⚙️ Simulateur (What-If)", 
    "🗄️ Données brutes"
])

# 1. EXPLICATION DU MODÈLE
with tab1:
    st.markdown("### Pourquoi cette décision ?")
    st.caption("Comparaison entre l'importance globale du modèle et les contributions locales spécifiques à ce client.")
    
    url_explain = explain_url_from_predict(api_url) + "?k=10"
    try:
        with st.spinner("Récupération de l'explicabilité..."):
            r_exp = requests.post(url_explain, json=[clean_record_for_json(client_row_df.to_dict(orient="records")[0])], timeout=60)
        
        if r_exp.status_code == 200:
            explain_data = r_exp.json()
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                st.markdown("#### 🎯 Impact Local (Ce client)")
                df_local = pd.DataFrame(explain_data.get("top", []))
                if not df_local.empty:
                    df_local["feature"] = df_local["feature"].apply(pretty_label)
                    fig_local = px.bar(df_local.sort_values("contribution"), x="contribution", y="feature", orientation="h", color="contribution", color_continuous_scale=["#15803D", "#E2E8F0", "#DC2626"])
                    fig_local.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0), height=350)
                    st.markdown("<div class='sr-only'>Graphique d'importance locale.</div>", unsafe_allow_html=True)
                    st.plotly_chart(fig_local, use_container_width=True)
                else: st.info("Aucune donnée locale.")
                
            with col_exp2:
                st.markdown("#### 🌍 Impact Global (Le modèle)")
                df_global = pd.DataFrame(explain_data.get("global_importance", [])).tail(10)
                if not df_global.empty:
                    df_global["feature"] = df_global["feature"].apply(pretty_label)
                    fig_global = px.bar(df_global, x="importance", y="feature", orientation="h", color_discrete_sequence=["#2563EB"])
                    fig_global.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=350)
                    st.markdown("<div class='sr-only'>Graphique d'importance globale.</div>", unsafe_allow_html=True)
                    st.plotly_chart(fig_global, use_container_width=True)
                else: st.info("Données globales indisponibles.")
        else:
            st.warning(f"Impossible de récupérer l'explication (Erreur {r_exp.status_code})")
    except Exception as e:
        st.warning(f"Service d'explication indisponible.")

# 2. COMPARAISON CLIENT
with tab2:
    st.markdown("### Distributions univariées")
    candidate_cols = [c for c in num_cols if c not in ("SK_ID_CURR", "TARGET")]
    feat = st.selectbox("Sélectionnez une variable pour voir où se situe le client :", options=candidate_cols, format_func=pretty_label)
    
    # Echantillonnage pour la performance
    df_samp = df[feat].dropna().sample(min(8000, len(df.dropna(subset=[feat]))), random_state=42)
    x_val = float(row.get(feat, np.nan))
    
    fig_hist = px.histogram(x=df_samp, nbins=40, labels={"x": pretty_label(feat), "count": "Fréquence"}, color_discrete_sequence=["#94a3b8"])
    if np.isfinite(x_val):
        fig_hist.add_vline(x=x_val, line_color="#DC2626", line_width=3, annotation_text="Client", annotation_position="top left")
    st.markdown("<div class='sr-only'>Histogramme de distribution.</div>", unsafe_allow_html=True)
    st.plotly_chart(fig_hist, use_container_width=True)

# 3. ANALYSE BIVARIÉE
with tab3:
    st.markdown("### Analyse Bivariée")
    colX, colY = st.columns(2)
    with colX: feat_x = st.selectbox("Axe Horizontal (X)", options=candidate_cols, index=0, format_func=pretty_label)
    with colY: feat_y = st.selectbox("Axe Vertical (Y)", options=candidate_cols, index=min(1, len(candidate_cols)-1), format_func=pretty_label)
    
    df_biv = df[[feat_x, feat_y]].dropna().sample(min(5000, len(df.dropna(subset=[feat_x, feat_y]))), random_state=42)
    fig_biv = px.scatter(df_biv, x=feat_x, y=feat_y, opacity=0.4, labels={feat_x: pretty_label(feat_x), feat_y: pretty_label(feat_y)}, color_discrete_sequence=["#cbd5e1"])
    
    cx, cy = float(row.get(feat_x, np.nan)), float(row.get(feat_y, np.nan))
    if np.isfinite(cx) and np.isfinite(cy):
        fig_biv.add_scatter(x=[cx], y=[cy], mode='markers', marker=dict(color='#DC2626', size=16, symbol='star', line=dict(color='white', width=1)), name="Client Actuel")
    st.markdown("<div class='sr-only'>Nuage de points bivarié.</div>", unsafe_allow_html=True)
    st.plotly_chart(fig_biv, use_container_width=True)

# 4. SIMULATEUR
with tab4:
    st.markdown("### ⚙️ Simulateur de décision (What-If)")
    st.info("Modifiez les critères ci-dessous pour voir comment le score du client évolue en temps réel.")
    
    c_age = abs(float(row.get("DAYS_BIRTH", 0))) / 365.25 if pd.notna(row.get("DAYS_BIRTH")) else 30.0
    c_inc = float(row.get("AMT_INCOME_TOTAL", 0)) if pd.notna(row.get("AMT_INCOME_TOTAL")) else 0.0
    c_cre = float(row.get("AMT_CREDIT", 0)) if pd.notna(row.get("AMT_CREDIT")) else 0.0
    
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1: n_age = st.number_input("Âge (années)", value=c_age, step=1.0)
    with col_s2: n_inc = st.number_input("Revenus annuels (€)", value=c_inc, step=5000.0)
    with col_s3: n_cre = st.number_input("Montant Crédit (€)", value=c_cre, step=10000.0)
    
    if st.button("Lancer la simulation", type="primary"):
        sim_rec = client_row_df.to_dict(orient="records")[0].copy()
        sim_rec["DAYS_BIRTH"] = -(n_age * 365.25)
        sim_rec["AMT_INCOME_TOTAL"] = n_inc
        sim_rec["AMT_CREDIT"] = n_cre
        
        with st.spinner("Calcul..."):
            try:
                r_sim = requests.post(api_url, json=[clean_record_for_json(sim_rec)], timeout=30)
                if r_sim.status_code == 200:
                    r_data = r_sim.json()
                    new_prob = float(r_data.get("probability", 0))
                    new_pred = int(r_data.get("prediction", 0))
                    
                    st.divider()
                    c_rs1, c_rs2 = st.columns(2)
                    with c_rs1:
                        st.metric("Nouvelle Probabilité", f"{new_prob:.2%}", delta=f"{new_prob - proba:+.2%}", delta_color="inverse")
                    with c_rs2:
                        b_sim = pill("REFUSÉ", "ko") if new_pred == 1 else pill("ACCORDÉ", "ok")
                        st.markdown(f"**Nouveau Statut :** <br><br> {b_sim}", unsafe_allow_html=True)
                else: st.error("Erreur de simulation.")
            except: st.error("Connexion perdue.")

# 5. DONNÉES BRUTES
with tab5:
    st.markdown("### 🗄️ Extraction complète")
    raw = client_row_df.T.reset_index()
    raw.columns = ["Variable Source", "Valeur Brute"]
    raw.insert(1, "Variable (Métier)", raw["Variable Source"].apply(pretty_label))
    raw["Valeur (Formatée)"] = raw.apply(lambda r: pretty_value(r["Variable Source"], r["Valeur Brute"]), axis=1)
    st.dataframe(raw, use_container_width=True, hide_index=True)

st.divider()
st.markdown("<p style='text-align: center; color: var(--muted); font-size: 0.8rem;'>Dashboard interactif - Prêt à dépenser © 2026</p>", unsafe_allow_html=True)