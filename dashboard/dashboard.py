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
# CONFIG
# =============================
LOCAL_API_BASE = "http://127.0.0.1:5001"
RENDER_API_BASE = "https://projet-8-gyq1.onrender.com"
DEFAULT_API_BASE = os.environ.get("API_BASE_URL", LOCAL_API_BASE)

# Critère WCAG 2.4.2 : Titre de page descriptif
st.set_page_config(
    page_title="Scoring Crédit - Prêt à dépenser",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# CSS (clair + dark mode + accessibilité)
# =============================
st.markdown("""
<style>
:root{
  --bg: #ffffff;
  --panel: rgba(17, 24, 39, 0.035);
  --panel2: rgba(17, 24, 39, 0.02);
  --border: rgba(17, 24, 39, 0.11);
  --text: rgba(17, 24, 39, 0.95);
  --muted: rgba(17, 24, 39, 0.75); /* WCAG 1.4.3 */
  --primary: #2563EB;
  --success: #15803D; /* WCAG 1.4.3 */
  --danger:  #DC2626;
  --warn:    #D97706;
  --slate:   #0F172A;
}

.sr-only {
  position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; 
  overflow: hidden; clip: rect(0, 0, 0, 0); border: 0;
}

/* WCAG 1.4.4 : rem pour le redimensionnement du texte */
.block-container{padding-top: 2.4rem; padding-bottom: 2rem; max-width: 1400px;}
[data-testid="stSidebar"]{border-right: 1px solid var(--border);}
[data-testid="stSidebar"] .block-container{padding-top: 1.2rem;}

h1,h2,h3{letter-spacing:-0.02em; line-height: 1.12; margin-top: 0.2rem;}
.small{color: var(--muted); font-size:0.92rem;}
.tiny{color: var(--muted); font-size:0.82rem;}
.kicker{font-size:.78rem; font-weight:900; letter-spacing:.12em; text-transform:uppercase; color: var(--muted);}

.card{background: var(--bg); border: 1px solid var(--border); border-radius: 18px; padding: 1rem 1.1rem; box-shadow: 0 10px 26px rgba(0,0,0,.06);}
.hr{height:1px; background: var(--border); border:none; margin:.85rem 0;}
.rowgap{margin-top:.95rem;}

.pill{display:inline-flex; align-items:center; gap:.5rem; padding:.30rem .70rem; border-radius:999px; border:1px solid var(--border); background: rgba(17,24,39,.03); font-weight:900; font-size:.85rem; color: rgba(17,24,39,.78);}
.pill-dot{width:8px;height:8px;border-radius:999px;background: var(--primary); display:inline-block;}
.pill-ok{background: rgba(22,163,74,.10); border-color: rgba(22,163,74,.30); color: rgba(6,95,70,1);}
.pill-ok .pill-dot{background: var(--success);}
.pill-ko{background: rgba(220,38,38,.10); border-color: rgba(220,38,38,.30); color: rgba(153,27,27,1);}
.pill-ko .pill-dot{background: var(--danger);}
.pill-warn{background: rgba(217,119,6,.10); border-color: rgba(217,119,6,.30); color: rgba(120,53,15,1);}
.pill-warn .pill-dot{background: var(--warn);}

.kpi{border:1px solid var(--border); border-radius: 18px; padding: .9rem 1rem; background: var(--bg);}
.kpi .t{font-weight:900; color: rgba(17,24,39,.72); font-size:.92rem;}
.kpi .v{font-weight:950; font-size:1.65rem; letter-spacing:-.02em; color: var(--slate); margin-top:.15rem;}
.kpi .s{color: var(--muted); font-size:.85rem; margin-top:.15rem;}

.profile-card{background: var(--panel); border: 1px solid var(--border); border-radius: 18px; padding: 1rem 1.05rem;}
.profile-head{display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; padding-bottom:.7rem; border-bottom: 1px solid rgba(17,24,39,.09);}
.profile-title{font-weight:950; letter-spacing:-.02em; color: var(--slate); font-size: 1.02rem;}
.profile-sub{margin-top:.18rem; color: var(--muted); font-size: .90rem;}
.profile-tag{display:inline-flex; align-items:center; gap:.45rem; padding:.28rem .60rem; border-radius:999px; border: 1px solid var(--border); background: rgba(255,255,255,.55); font-weight:950; font-size:.82rem; color: rgba(17,24,39,.78);}
.profile-grid{display:flex; flex-direction:column; gap:.55rem; margin-top:.85rem;}
.profile-row{display:flex; justify-content:space-between; gap:1rem; padding:.62rem .75rem; border-radius: 14px; border: 1px solid rgba(17,24,39,.10); background: var(--panel2);}
.profile-k{font-weight:900; font-size:.88rem; color: rgba(17,24,39,.72);}
.profile-v{font-weight:950; font-size:.92rem; color: rgba(17,24,39,.92); text-align:right;}

.stButton button{border-radius: 14px !important; padding: .65rem 1.0rem !important; font-weight: 950 !important;}
.stTextInput input{border-radius: 14px !important;}
[data-testid="stDataFrame"]{border-radius:14px; overflow:hidden; border:1px solid var(--border);}

@media (prefers-color-scheme: dark) {
  :root{
    --bg: #0b1220; --panel: rgba(255,255,255,.075); --panel2: rgba(255,255,255,.055);
    --border: rgba(255,255,255,.13); --text: rgba(255,255,255,.92); --muted: rgba(255,255,255,.75);
    --primary: #60A5FA; --success: #34D399; --danger: #F87171; --warn: #FBBF24; --slate: rgba(255,255,255,.92);
  }
  .card{background: rgba(255,255,255,.04); border: 1px solid var(--border); box-shadow:none;}
  .kpi{background: rgba(255,255,255,.04); border: 1px solid var(--border);}
  .kpi .t{color: rgba(255,255,255,.74);} .kpi .v{color: var(--text);} .kpi .s{color: var(--muted);}
  .pill{background: rgba(255,255,255,.07); border-color: rgba(255,255,255,.16); color: rgba(255,255,255,.88);}
  .profile-card{background: rgba(255,255,255,.055); border-color: rgba(255,255,255,.14);}
  .profile-head{border-bottom: 1px solid rgba(255,255,255,.12);}
  .profile-title{color: rgba(255,255,255,.92);} .profile-sub{color: rgba(255,255,255,.68);}
  .profile-tag{background: rgba(255,255,255,.06); color: rgba(255,255,255,.86); border-color: rgba(255,255,255,.16);}
  .profile-row{background: rgba(255,255,255,.04); border-color: rgba(255,255,255,.12);}
  .profile-k{color: rgba(255,255,255,.72);} .profile-v{color: rgba(255,255,255,.92);}
  [data-testid="stSidebar"]{border-right: 1px solid rgba(255,255,255,.10);}
}
</style>
""", unsafe_allow_html=True)

# =============================
# URL helpers & Data load
# =============================
def normalize_predict_url(base_or_full_url: str) -> str:
    url = (base_or_full_url or "").strip()
    if not url: return LOCAL_API_BASE.rstrip("/") + "/predict"
    url = url.rstrip("/")
    return url if url.endswith("/predict") else url + "/predict"

def base_from_predict(predict_url: str) -> str:
    u = (predict_url or "").strip().rstrip("/")
    return u[:-len("/predict")] if u.endswith("/predict") else u

def explain_url_from_predict(predict_url: str) -> str:
    return base_from_predict(predict_url).rstrip("/") + "/explain"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def split_cols(df_: pd.DataFrame):
    num = df_.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in df_.columns if c not in num]
    return num, cat

# =============================
# Formatting / cleaning
# =============================
LABELS_FR = {
    "CNT_CHILDREN": "Nombre d’enfants", "CNT_FAM_MEMBERS": "Taille du foyer", "CODE_GENDER": "Genre",
    "NAME_FAMILY_STATUS": "Statut marital", "NAME_INCOME_TYPE": "Type de revenu", "OCCUPATION_TYPE": "Profession",
    "NAME_CONTRACT_TYPE": "Type de contrat", "AMT_INCOME_TOTAL": "Revenu annuel", "AMT_CREDIT": "Montant du crédit",
    "AMT_ANNUITY": "Mensualité (annuité)", "AMT_GOODS_PRICE": "Prix du bien",
    "REGION_POPULATION_RELATIVE": "Densité régionale (rel.)", "DAYS_BIRTH": "Âge",
    "DAYS_EMPLOYED": "Ancienneté emploi", "DAYS_REGISTRATION": "Ancienneté dossier", "DAYS_ID_PUBLISH": "Ancienneté pièce ID",
}
MONEY_COLS = {"AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"}

def fmt_number(v):
    if v is None: return "—"
    try:
        if pd.isna(v): return "—"
    except Exception: pass
    if isinstance(v, (int, np.integer)): return f"{int(v):,}".replace(",", " ")
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v): return "—"
        x = float(v)
        return f"{x:,.2f}".replace(",", " ").replace(".", ",") if abs(x) < 1000 else f"{x:,.0f}".replace(",", " ")
    return str(v)

def humanize_special(feature: str, value):
    if value is None: return value
    if feature.startswith("DAYS_") and isinstance(value, (int, float, np.integer, np.floating)):
        return abs(float(value)) / 365.25  
    return value

def pretty_label(col: str) -> str:
    return LABELS_FR.get(col, col.replace("_", " ").title())

def pretty_value(col: str, v):
    if v is None: return "—"
    if col.startswith("DAYS_"):
        vv = humanize_special(col, v)
        return f"{float(vv):.1f} ans".replace(".", ",") if isinstance(vv, (int, float, np.integer, np.floating)) else "—"
    if col in MONEY_COLS: return f"{fmt_number(v)} €"
    if col == "REGION_POPULATION_RELATIVE":
        try: return f"{float(v)*100:.1f} %".replace(".", ",")
        except Exception: return fmt_number(v)
    if col.startswith("CNT_"):
        try: return f"{int(round(float(v)))}"
        except Exception: return fmt_number(v)
    return fmt_number(v)

def clean_record_for_json(record: dict):
    out = {}
    for k, v in record.items():
        if isinstance(v, (float, np.floating)): out[k] = None if np.isnan(v) or np.isinf(v) else float(v)
        elif isinstance(v, (int, np.integer)): out[k] = int(v)
        else:
            try: out[k] = None if pd.isna(v) else v
            except Exception: out[k] = v
    return out

# =============================
# UI blocks
# =============================
def pill(label: str, kind: str = "default"):
    cls = "pill"
    if kind == "ok": cls += " pill-ok"
    elif kind == "ko": cls += " pill-ko"
    elif kind == "warn": cls += " pill-warn"
    return f"<span class='{cls}'><span class='pill-dot'></span>{label}</span>"

def hero(title_left: str, subtitle: str, pills: List[str], endpoint: str):
    pills_html = " ".join(pills)
    st.markdown(f"""
    <div class="card">
      <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:1rem;">
        <div style="flex:1;">
          <div class="kicker">Credit scoring</div>
          <div style="font-size:2.05rem; font-weight:950; letter-spacing:-.03em; margin-top:.15rem; color: var(--slate);">{title_left}</div>
          <div class="small" style="margin-top:.3rem;">{subtitle}</div>
          <div class="tiny" style="margin-top:.35rem;">Endpoint: <code>{endpoint}</code></div>
        </div>
        <div style="display:flex; flex-direction:column; align-items:flex-end; gap:.45rem;">{pills_html}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

def kpi(title: str, value: str, sub: str):
    st.markdown(f"<div class='kpi'><div class='t'>{title}</div><div class='v'>{value}</div><div class='s'>{sub}</div></div>", unsafe_allow_html=True)

def profile_card_html(title: str, subtitle: str, items: List[Tuple[str, str]]) -> str:
    count = len(items)
    tag = f"<span class='profile-tag'>{count} champ{'s' if count>1 else ''}</span>" if count > 0 else ""
    rows = "".join([f"<div class='profile-row'><div class='profile-k'>{k}</div><div class='profile-v'>{v}</div></div>" for k, v in items])
    if not rows: rows = "<div class='small' style='margin-top:.85rem;'>Aucune donnée disponible dans ce dataset.</div>"
    return f"""
    <div class="profile-card">
      <div class="profile-head"><div><div class="profile-title">{title}</div><div class="profile-sub">{subtitle}</div></div>{tag}</div>
      <div class="profile-grid">{rows}</div>
    </div>
    """

def bullet_risk(prob: float, threshold: float):
    prob = float(np.clip(prob, 0, 1))
    threshold = float(np.clip(threshold, 0, 1))
    color = "#DC2626" if prob >= threshold else "#15803D"
    fig, ax = plt.subplots(figsize=(9, 1.6))
    ax.barh([0], [1.0], height=0.28, alpha=0.12)
    ax.barh([0], [prob], height=0.28, color=color)
    ax.axvline(threshold, linestyle="--", linewidth=2)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Probabilité de défaut")
    ax.set_title("Risque estimé (barre) + seuil (ligne)")
    for spine in ax.spines.values(): spine.set_visible(False)
    
    st.markdown("<div class='sr-only'>Graphique jauge montrant la probabilité de défaut estimée par rapport au seuil d'acceptation.</div>", unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)

def top_deviations(df_all: pd.DataFrame, row: pd.Series, numeric_cols: List[str], sample_size: int = 8000, k: int = 12):
    df_num = df_all[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df_num_s = df_num.sample(sample_size, random_state=42) if len(df_num) > sample_size else df_num
    mu = df_num_s.mean(skipna=True)
    sigma = df_num_s.std(skipna=True).replace(0, np.nan)
    client_num = row[numeric_cols]
    z = ((client_num - mu) / sigma).abs().sort_values(ascending=False)
    topk = z.head(k).index.tolist()
    out = pd.DataFrame({
        "feature": topk, "valeur_client": [client_num[c] for c in topk],
        "moyenne": [mu[c] for c in topk], "z_abs": [z[c] for c in topk],
    }).sort_values("z_abs", ascending=False)
    return out, df_num_s

# =============================
# SIDEBAR
# =============================
st.sidebar.header("Paramètres")
st.sidebar.caption("API • données • sélection")
st.sidebar.divider()
st.sidebar.subheader("API")

api_mode = st.sidebar.radio("Mode", ["Local", "Render", "Custom"], index=0)
api_base = LOCAL_API_BASE if api_mode == "Local" else (RENDER_API_BASE if api_mode == "Render" else st.sidebar.text_input("API base URL", value=DEFAULT_API_BASE))
api_url = normalize_predict_url(api_base)
st.sidebar.caption("Endpoint utilisé")
st.sidebar.code(api_url, language="text")

if st.sidebar.button("Tester l’API", use_container_width=True):
    try:
        t0 = time.time()
        r = requests.get(base_from_predict(api_url) + "/", timeout=20)
        st.sidebar.success(f"OK • {time.time()-t0:.2f}s") if r.status_code == 200 else st.sidebar.warning(f"HTTP {r.status_code}")
    except Exception as e:
        st.sidebar.error(f"Indispo: {e}")

st.sidebar.divider()
st.sidebar.subheader("Données")
data_path = st.sidebar.text_input("Chemin dataset", value="model_production/test_sample_processed.csv")
sample_size = st.sidebar.slider("Échantillon stats", 2000, 30000, 8000, 1000)

st.sidebar.divider()
st.sidebar.subheader("Client")

try:
    df = load_data(data_path)
except Exception as e:
    st.error(f"Impossible de charger: {data_path}\n\nDétail: {e}")
    st.stop()

if "SK_ID_CURR" in df.columns:
    client_id = st.sidebar.selectbox("SK_ID_CURR", df["SK_ID_CURR"].tolist())
    client_row_df = df[df["SK_ID_CURR"] == client_id]
else:
    client_id = st.sidebar.selectbox("Index", df.index)
    client_row_df = df.loc[[client_id]]

row = client_row_df.iloc[0]
num_cols, cat_cols = split_cols(client_row_df)

st.sidebar.divider()
call_api = st.sidebar.button("Calculer le scoring", use_container_width=True)
show_raw = st.sidebar.checkbox("Afficher données brutes", value=False)
show_details = st.sidebar.checkbox("Analyse variables", value=True)

# =============================
# API CALL
# =============================
if "api_result" not in st.session_state: st.session_state.api_result = None
if "api_error" not in st.session_state: st.session_state.api_error = None

def do_api_call():
    payload = [clean_record_for_json(client_row_df.to_dict(orient="records")[0])]
    try:
        with st.spinner("Appel API..."):
            t0 = time.time()
            r = requests.post(api_url, json=payload, timeout=60)
            dt = time.time() - t0
        if r.status_code == 200:
            st.session_state.api_result = r.json()
            st.session_state.api_result["_latency_s"] = dt
            st.session_state.api_error = None
        else:
            st.session_state.api_result = None
            st.session_state.api_error = f"Erreur API {r.status_code} — {r.text}"
    except Exception as e:
        st.session_state.api_result = None
        st.session_state.api_error = f"Connexion API impossible: {e}"

if call_api: do_api_call()

def do_explain_call(k: int = 12):
    payload = [clean_record_for_json(client_row_df.to_dict(orient="records")[0])]
    url = explain_url_from_predict(api_url) + f"?k={k}"
    try:
        with st.spinner("Récupération explication..."):
            r = requests.post(url, json=payload, timeout=60)
        return (r.json(), None) if r.status_code == 200 else (None, f"Erreur explain {r.status_code} — {r.text}")
    except Exception as e:
        return None, f"Explain indisponible: {e}"

res = st.session_state.api_result
err = st.session_state.api_error

# =============================
# HERO & HEADER
# =============================
pills = [pill(f"Mode: {api_mode}", "warn" if api_mode == "Custom" else "default")]
if err: pills.append(pill("API: erreur", "ko"))
else: pills.append(pill("API: prête", "ok"))
lat = res.get("_latency_s") if isinstance(res, dict) else None
if lat is not None: pills.append(pill(f"Latence: {lat:.2f}s", "warn" if lat > 5 else "ok"))

hero("Décision & Analyse client", "Décision, seuil, et compréhension du profil client.", pills, api_url)

c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1])
with c1:
    st.markdown("<div class='card'><div class='kicker'>Client sélectionné</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:1.35rem; font-weight:950; color: var(--slate);'>ID: {client_id}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small'>Dataset: {len(df):,} lignes • Features: {client_row_df.shape[1]}</div>".replace(",", " "), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with c2: kpi("Mode API", api_mode, "Local / Render / Custom")
with c3: kpi("Endpoint", "…/predict", "Affiché dans l’en-tête")
with c4: kpi("Statut", "Erreur" if err else "OK", "Vérifier l’URL" if err else "API disponible")

if err:
    st.error(err)
    st.stop()

# =============================
# RESULTS
# =============================
if res is None:
    st.warning("Clique sur « Calculer le scoring » pour obtenir une décision.")
    st.stop() # Arrête le rendu du reste tant qu'on n'a pas scoré

proba = float(res.get("probability", 0.0))
threshold = float(res.get("threshold", 0.5))
status = res.get("status", "—")
prediction = int(res.get("prediction", 0))
is_refused = (prediction == 1) or (str(status).strip().lower() == "refusé")

st.markdown("<div class='rowgap'></div>", unsafe_allow_html=True)

left, right = st.columns([1.2, 1])
with left:
    badge = pill("Décision: REFUSÉ", "ko") if is_refused else pill("Décision: ACCORDÉ", "ok")
    st.markdown(f"""
    <div class="card">
      <div class="kicker">Décision</div>
      <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem; margin-top:.35rem;">
        <div style="font-size:1.25rem; font-weight:950; color: var(--slate);">{status}</div>{badge}
      </div>
      <div class="hr"></div>
      <div class="small">Probabilité = <b>{proba:.2%}</b> • Seuil = <b>{threshold:.2f}</b> • Classe = <b>{prediction}</b></div>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'><div class='kicker'>Risque</div>", unsafe_allow_html=True)
    bullet_risk(proba, threshold)
    st.markdown("</div>", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1: kpi("Probabilité défaut", f"{proba:.2%}", "Plus élevé = plus risqué")
with k2: kpi("Seuil", f"{threshold:.2f}", "Règle de décision")
with k3: kpi("Écart au seuil", f"{(proba-threshold):+.2%}", "Proba - seuil")
with k4: kpi("Latence", f"{res.get('_latency_s', 0):.2f}s", "Temps API")

# =============================
# PROFIL CLIENT 
# =============================
st.markdown("## Profil client")
st.caption("Synthèse structurée : informations clés par thématique.")

GROUPS: Dict[str, List[str]] = {
    "Identité & ménage": ["CODE_GENDER", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "NAME_FAMILY_STATUS"],
    "Revenus & emploi":  ["AMT_INCOME_TOTAL", "NAME_INCOME_TYPE", "DAYS_EMPLOYED", "OCCUPATION_TYPE"],
    "Crédit":           ["NAME_CONTRACT_TYPE", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"],
    "Région & historique": ["REGION_POPULATION_RELATIVE", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"],
}

def pick_items(cols: List[str], max_items: int = 7) -> List[Tuple[str, str]]:
    items = []
    for c in cols:
        if c in client_row_df.columns: items.append((pretty_label(c), pretty_value(c, row.get(c, None))))
        if len(items) >= max_items: break
    return items

cards = [
    ("Identité & ménage", "Situation familiale", pick_items(GROUPS["Identité & ménage"])),
    ("Revenus & emploi", "Revenus & emploi", pick_items(GROUPS["Revenus & emploi"])),
    ("Crédit", "Détails du prêt", pick_items(GROUPS["Crédit"])),
    ("Région & historique", "Contexte géographique", pick_items(GROUPS["Région & historique"])),
]

cA, cB = st.columns(2)
cC, cD = st.columns(2)
for col, (title, sub, items) in zip([cA, cB, cC, cD], cards):
    with col:
        if items: st.markdown(profile_card_html(title, sub, items), unsafe_allow_html=True)

# =============================
# ANALYSE VARIABLES (LES 6 ONGLETS)
# =============================
st.markdown("<div class='rowgap'></div>", unsafe_allow_html=True)

# Ajout du nouvel onglet "Simulateur"
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Top variables (z-score)",
    "Distributions",
    "Analyse Bivariée",
    "Explication modèle",
    "Simulateur (What-If)",
    "Données brutes"
])

with tab1:
    st.markdown("### Top variables atypiques")
    if not show_details: st.info("Active « Analyse variables » dans la sidebar.")
    elif len(num_cols) == 0: st.info("Aucune variable numérique détectée.")
    else:
        top_df, df_num_sample = top_deviations(df, row, num_cols, sample_size=sample_size, k=12)
        show = top_df.copy()
        show["feature"] = show["feature"].apply(pretty_label)
        show["valeur_client"] = show["valeur_client"].apply(fmt_number)
        show["moyenne"] = show["moyenne"].apply(fmt_number)
        show["z_abs"] = show["z_abs"].astype(float).round(3)

        st.dataframe(show, use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.barh(show["feature"][::-1], show["z_abs"][::-1])
        ax.set_xlabel("z-score absolu")
        ax.set_title("Top 12 variables les plus atypiques")
        st.markdown("<div class='sr-only'>Graphique à barres montrant les variables atypiques.</div>", unsafe_allow_html=True)
        st.pyplot(fig, use_container_width=True)

with tab2:
    st.markdown("### Distributions (client vs population)")
    if not show_details or 'df_num_sample' not in locals() or df_num_sample is None: st.warning("Active l'analyse des variables dans la barre latérale.")
    else:
        candidate_cols = [c for c in df_num_sample.columns if c not in ("SK_ID_CURR", "TARGET")]
        if not candidate_cols: st.warning("Aucune variable numérique exploitable.")
        else:
            feat = st.selectbox("Choisir une variable pour l'histogramme", options=candidate_cols, index=0)
            s = df_num_sample[feat].replace([np.inf, -np.inf], np.nan).dropna()
            x_client = float(row.get(feat, np.nan)) if pd.notna(row.get(feat, np.nan)) else np.nan

            fig = px.histogram(x=s, nbins=30, labels={"x": pretty_label(feat), "count": "Nombre de clients"})
            if np.isfinite(x_client): fig.add_vline(x=x_client, line_color="red", annotation_text="Client")
            st.markdown(f"<div class='sr-only'>Histogramme pour {feat}.</div>", unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Analyse Bivariée")
    if not show_details or 'df_num_sample' not in locals() or df_num_sample is None: st.warning("Active l'analyse des variables.")
    else:
        candidate_cols = [c for c in df_num_sample.columns if c not in ("SK_ID_CURR", "TARGET")]
        colX, colY = st.columns(2)
        with colX: feat_x = st.selectbox("Axe X (Variable 1)", options=candidate_cols, index=0, key="biv_x")
        with colY: feat_y = st.selectbox("Axe Y (Variable 2)", options=candidate_cols, index=min(1, len(candidate_cols)-1), key="biv_y")
        
        fig = px.scatter(df_num_sample, x=feat_x, y=feat_y, opacity=0.3, labels={feat_x: pretty_label(feat_x), feat_y: pretty_label(feat_y)})
        x_client = float(row.get(feat_x, np.nan))
        y_client = float(row.get(feat_y, np.nan))
        
        if np.isfinite(x_client) and np.isfinite(y_client):
            fig.add_scatter(x=[x_client], y=[y_client], mode='markers', marker=dict(color='red', size=14, symbol='star'), name=f"Client sélectionné")
        st.markdown(f"<div class='sr-only'>Nuage de points croisant {feat_x} et {feat_y}.</div>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### Explication de la décision")
    explain, eerr = do_explain_call(k=12)
    if eerr: st.warning(eerr)
    else:
        top_local = explain.get("top", [])
        top_global = explain.get("global_importance", [])
        if not top_local: st.info("Pas d’explication renvoyée.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Importance LOCALE")
                df_local = pd.DataFrame(top_local)
                df_local["abs"] = df_local["contribution"].abs()
                df_local = df_local.sort_values("abs", ascending=True)
                fig_local = px.bar(df_local, x="contribution", y="feature", orientation="h", hover_data=["value"], color="contribution", color_continuous_scale=["#15803D", "#E2E8F0", "#DC2626"])
                fig_local.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=30, b=0))
                st.markdown("<div class='sr-only'>Graphique montrant l'importance locale.</div>", unsafe_allow_html=True)
                st.plotly_chart(fig_local, use_container_width=True)
            with col2:
                st.markdown("#### Importance GLOBALE")
                if top_global:
                    df_global = pd.DataFrame(top_global).sort_values("importance", ascending=True).tail(12)
                    fig_global = px.bar(df_global, x="importance", y="feature", orientation="h", color_discrete_sequence=["#2563EB"])
                    fig_global.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                    st.markdown("<div class='sr-only'>Graphique montrant l'importance globale.</div>", unsafe_allow_html=True)
                    st.plotly_chart(fig_global, use_container_width=True)
                else: st.info("Indisponible.")

# Le NOUVEL onglet de simulation
with tab5:
    st.markdown("### Simulateur d'octroi de crédit (What-If)")
    st.caption("Modifiez les informations du client pour voir l'impact immédiat sur sa probabilité de défaut. Permet d’obtenir un score rafraîchi[cite: 161].")

    # Extraction des valeurs actuelles pour pré-remplir le formulaire
    curr_income = float(row.get("AMT_INCOME_TOTAL", 0)) if pd.notna(row.get("AMT_INCOME_TOTAL")) else 0.0
    curr_credit = float(row.get("AMT_CREDIT", 0)) if pd.notna(row.get("AMT_CREDIT")) else 0.0
    
    curr_age_days = float(row.get("DAYS_BIRTH", 0)) if pd.notna(row.get("DAYS_BIRTH")) else 0.0
    curr_age_years = abs(curr_age_days) / 365.25 if curr_age_days != 0 else 30.0
    
    curr_emp_days = float(row.get("DAYS_EMPLOYED", 0)) if pd.notna(row.get("DAYS_EMPLOYED")) else 0.0
    # Dans la DB, l'ancienneté est souvent négative (-1000 = embauché il y a 1000 jours)
    curr_emp_years = abs(curr_emp_days) / 365.25 if curr_emp_days < 0 else 0.0 

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        new_income = st.number_input("Revenu annuel (€)", value=curr_income, step=5000.0)
        new_age_years = st.number_input("Âge (années)", value=curr_age_years, step=1.0)
    with col_sim2:
        new_credit = st.number_input("Montant du crédit souhaité (€)", value=curr_credit, step=10000.0)
        new_emp_years = st.number_input("Ancienneté dans l'emploi (années)", value=curr_emp_years, step=1.0)

    if st.button("Lancer la simulation", use_container_width=True):
        # Créer une copie modifiée du dossier client
        sim_record = client_row_df.to_dict(orient="records")[0].copy()
        sim_record["AMT_INCOME_TOTAL"] = new_income
        sim_record["AMT_CREDIT"] = new_credit
        sim_record["DAYS_BIRTH"] = - (new_age_years * 365.25) # Reconversion en jours négatifs
        if curr_emp_days < 0 or new_emp_years > 0:
            sim_record["DAYS_EMPLOYED"] = - (new_emp_years * 365.25)
            
        payload_sim = [clean_record_for_json(sim_record)]
        
        with st.spinner("Calcul du nouveau score..."):
            try:
                r_sim = requests.post(api_url, json=payload_sim, timeout=60)
                if r_sim.status_code == 200:
                    res_sim = r_sim.json()
                    new_proba = float(res_sim.get("probability", 0.0))
                    new_pred = int(res_sim.get("prediction", 0))
                    new_status = res_sim.get("status", "—")
                    
                    st.markdown("#### Résultat de la simulation")
                    c_s1, c_s2, c_s3 = st.columns(3)
                    delta_proba = new_proba - proba
                    
                    with c_s1:
                        st.metric("Nouvelle Probabilité", f"{new_proba:.2%}", delta=f"{delta_proba:+.2%}", delta_color="inverse")
                    with c_s2:
                        badge_sim = pill("REFUSÉ", "ko") if new_pred == 1 else pill("ACCORDÉ", "ok")
                        st.markdown(f"**Nouveau Statut:** <br> {badge_sim}", unsafe_allow_html=True)
                    with c_s3:
                        st.metric("Seuil actuel", f"{threshold:.2f}")
                else:
                    st.error(f"Erreur API lors de la simulation: {r_sim.status_code}")
            except Exception as e:
                st.error(f"Erreur de connexion pour la simulation: {e}")

with tab6:
    st.markdown("### Données brutes")
    if not show_raw: st.info("Active « Afficher données brutes » dans la sidebar.")
    else:
        raw = client_row_df.T.reset_index()
        raw.columns = ["feature", "value"]
        raw["feature"] = raw["feature"].apply(pretty_label)
        raw["value"] = raw.apply(lambda r: pretty_value(r["feature"].upper().replace(" ", "_"), r["value"]), axis=1)
        a, b = st.columns([1.2, 1])
        with a: st.dataframe(raw, use_container_width=True, hide_index=True)
        with b: st.dataframe(client_row_df, use_container_width=True)

st.divider()
st.caption("Dashboard interactif pour chargé de clientèle - Prêt à dépenser.")