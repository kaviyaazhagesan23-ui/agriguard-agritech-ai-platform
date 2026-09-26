from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sqlalchemy.orm import joinedload

from src.decision_engine import DecisionInputs, calculate_decision
from src.current_price_estimator import estimate_price, forecast_future_prices
from src.market_comparison import DEFAULT_TRANSPORT_COST_PER_KM, compare_markets, results_to_dataframe
from src.blockchain import load_or_create_blockchain, validate_blockchain, verify_record, create_decision_record

from src.auth.service import authenticate_user, register_user
from src.database.db import init_database, make_engine, session_factory, session_scope
from src.database.models import BuyerRequest, FarmerListing as DBFarmerListing, ListingStatus, Market, Message, OrderStatus, RequestStatus, User, UserLocation
from src.marketplace.service import accept_request, advance_order, create_buyer_request, create_farmer_listing, find_live_listings, match_buyer_to_listings, order_history, reject_request, request_purchase, send_message

ROOT = Path(__file__).resolve().parent
PRICE_DATA = ROOT / "data" / "processed" / "master_dataset.csv"
CURRENT_PRICE_DATA = ROOT / "data" / "current" / "synthetic_current_prices.csv"
BLOCKCHAIN_DATA = ROOT / "data" / "blockchain" / "paddywise_blockchain.json"
HERO_IMAGE = ROOT / "assets" / "images" / "farmer-hero.jpg"
FORECAST_DIR = ROOT / "reports" / "phase6" / "forecasts"

MARKET_COORDINATES = {
    "Budalur": (10.7968, 78.9769), "Kumbakonam": (10.9617, 79.3880),
    "Orathanadu": (10.6280, 79.2531), "Papanasam": (10.9252, 79.2708),
    "Pattukottai": (10.4250, 79.3140), "Thanjavur": (10.7870, 79.1378),
    "Vallam": (10.7140, 79.7020), "Thiruppananthal": (None, None),
}


# Simplified Thanjavur district boundary sourced from publicly available Indian
# district GeoJSON geometry. Used only for presentation in the interactive map.
THANJAVUR_BOUNDARY = {
    "type": "Feature",
    "properties": {"district": "Thanjavur", "state": "Tamil Nadu"},
    "geometry": {
        "type": "Polygon",
        "coordinates": [[
            [79.5067437630439, 10.354357642679279],
            [79.22886486925161, 10.140275464798826],
            [79.15289940453277, 10.197088480145732],
            [79.15638735687335, 10.198774600107932],
            [79.14579263158139, 10.382674531696072],
            [79.14810796037453, 10.388659385539674],
            [78.81846405629524, 10.836009275949277],
            [79.00752716900635, 10.889332691302767],
            [79.1319003633223, 10.930967566898945],
            [79.36886441662846, 10.794980091554661],
            [79.37333176739537, 10.800086097533029],
            [79.49328348713613, 10.747599391119724],
            [79.5067437630439, 10.354357642679279],
        ]]
    },
}

st.set_page_config(page_title="AgriGuard AI", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

# Remote, publicly accessible agriculture photography used only as presentation assets.
# If local assets/images are added later, the local hero image is preferred for the hero/auth panel.
IMAGE_URLS = {
    "farmer": "https://images.pexels.com/photos/14915250/pexels-photo-14915250.jpeg?auto=compress&cs=tinysrgb&w=1400",
    "paddy": "https://images.pexels.com/photos/30260658/pexels-photo-30260658.jpeg?auto=compress&cs=tinysrgb&w=1400",
    "field": "https://commons.wikimedia.org/wiki/Special:FilePath/Paddy_field_in_Tamil_Nadu_India.jpg?width=1400",
    "grain": "https://images.unsplash.com/photo-1770617474928-3c7554a51941?auto=format&fit=crop&fm=jpg&q=82&w=1400",
    "truck": "https://images.unsplash.com/photo-1774013607215-9f21488e9a19?auto=format&fit=crop&fm=jpg&q=82&w=1400",
    "warehouse": "https://images.unsplash.com/photo-1774655852740-7cbe1fd4ba02?auto=format&fit=crop&fm=jpg&q=82&w=1400",
    "blockchain": "https://images.unsplash.com/photo-1764712097764-94caf81dd1b3?auto=format&fit=crop&fm=jpg&q=82&w=1400",
}


def _image_src(key: str) -> str:
    # HTML image elements cannot safely address arbitrary local filesystem paths in Streamlit.
    # Keep the remote public asset as the browser-safe fallback; local assets are detected separately.
    return IMAGE_URLS[key]


def _image_card(key: str, label: str, subtitle: str = "", height: int = 180) -> None:
    src = html.escape(_image_src(key), quote=True)
    label_html = html.escape(label)
    subtitle_html = html.escape(subtitle)
    st.markdown(
        f"""<div class="pw-image-card">
            <img src="{src}" alt="{label_html}" loading="lazy"
                 onerror="this.style.display='none';this.parentElement.classList.add('pw-image-fallback');">
            <div class="pw-image-overlay"><div class="pw-image-label">{label_html}</div>{f'<div class="pw-image-sub">{subtitle_html}</div>' if subtitle else ''}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _image_strip(items) -> None:
    cols = st.columns(len(items), gap="small")
    for col, (key, label) in zip(cols, items):
        with col:
            _image_card(key, label, height=130)


def _section_heading(kicker: str, title: str, subtitle: str = "", icon: str = "🌾") -> None:
    st.markdown(
        f"""<div class="pw-section-heading"><div class="pw-kicker">{icon} {html.escape(kicker)}</div>
        <div class="pw-section-title">{html.escape(title)}</div>
        {f'<div class="pw-section-sub">{html.escape(subtitle)}</div>' if subtitle else ''}</div>""",
        unsafe_allow_html=True,
    )


def _hero_ai_price(market: str | None = None, variety: str | None = None, grade: str | None = None):
    """Read an actual value from the existing forecast engine for display; never invent a price."""
    try:
        supported = load_current_price_combinations()
        if supported.empty:
            return None
        if market not in set(supported["market"]):
            market = str(supported.iloc[0]["market"])
        subset = supported[supported["market"] == market]
        if variety not in set(subset["variety"]):
            variety = str(subset.iloc[0]["variety"])
        subset = subset[subset["variety"] == variety]
        if grade not in set(subset["grade"]):
            grade = str(subset.iloc[0]["grade"])
        forecast = forecast_future_prices(market, variety, grade, date.today(), 1)
        row = forecast.loc[pd.to_datetime(forecast["date"]).dt.date == date.today()]
        if row.empty:
            return None
        return float(row.iloc[0]["predicted_price_rs_per_quintal"])
    except Exception:
        return None



@st.cache_resource

def get_engine():
    engine = make_engine()
    init_database(engine)
    return engine


def db_session():
    return session_factory(get_engine())()


@st.cache_data(show_spinner=False)
def load_current_price_combinations() -> pd.DataFrame:
    """Load market/variety/grade combinations supported by current demo data."""
    if not CURRENT_PRICE_DATA.exists():
        return pd.DataFrame(columns=["market", "variety", "grade"])
    current = pd.read_csv(CURRENT_PRICE_DATA, usecols=["market", "variety", "grade"])
    return current.dropna().astype(str).drop_duplicates().reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_prices() -> pd.DataFrame:
    if not PRICE_DATA.exists():
        raise FileNotFoundError(f"Price dataset not found: {PRICE_DATA}")
    df = pd.read_csv(PRICE_DATA)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def money(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"₹{float(value):,.0f}"


def latest_price(df: pd.DataFrame, market: str, variety: str | None = None, grade: str | None = None):
    x = df[df["market"].astype(str) == market].copy()
    if variety is not None: x = x[x["variety"].astype(str) == variety]
    if grade is not None: x = x[x["grade"].astype(str) == grade]
    x = x.dropna(subset=["date", "modal_price_rs_per_quintal"]).sort_values("date")
    return None if x.empty else float(x.iloc[-1]["modal_price_rs_per_quintal"])


def latest_actual_date(df: pd.DataFrame, market: str | None = None, variety: str | None = None, grade: str | None = None):
    x = df.copy()
    if market is not None: x = x[x["market"].astype(str) == market]
    if variety is not None: x = x[x["variety"].astype(str) == variety]
    if grade is not None: x = x[x["grade"].astype(str) == grade]
    x = x.dropna(subset=["date"])
    return None if x.empty else x["date"].max().date()


def forecast_csv_path(market: str, variety: str, grade: str, horizon: int = 7) -> Path:
    filename = (
        f"forecast_{market.replace(' ', '_')}_{variety.replace(' ', '_')}_"
        f"{grade.replace(' ', '_')}_{horizon}d.csv"
    )
    return FORECAST_DIR / filename


def classify_forecast_dates(forecast_path: Path, latest_series_date, application_date):
    if not forecast_path.exists():
        return None
    forecast = pd.read_csv(forecast_path)
    if "date" not in forecast.columns or forecast.empty:
        return None
    forecast["date"] = pd.to_datetime(forecast["date"], errors="coerce")
    forecast = forecast.dropna(subset=["date"])
    if forecast.empty:
        return None
    start_date = forecast["date"].min().date()
    end_date = forecast["date"].max().date()
    if latest_series_date is not None and start_date > latest_series_date:
        generation = "future forecast relative to the selected historical series"
    else:
        generation = "forecast output that does not begin after the selected series' latest actual date"
    if end_date < application_date:
        temporal_status = "historical forecast output — not a current 2026 forecast"
    elif start_date > application_date:
        temporal_status = "future-dated forecast relative to the application date"
    else:
        temporal_status = "forecast overlaps the application date"
    return start_date, end_date, generation, temporal_status


def inject_css():
    st.markdown(
        """
        <style>
        :root {
            --pw-deep: #14532d;
            --pw-green: #2f8f46;
            --pw-lime: #79b84a;
            --pw-gold: #e8b84b;
            --pw-amber: #f59e0b;
            --pw-orange: #e97835;
            --pw-teal: #168b86;
            --pw-blue: #3478c9;
            --pw-sky: #dff2ff;
            --pw-purple: #6d5bd0;
            --pw-cream: #fffaf0;
            --pw-bg: #f5f7f2;
            --pw-ink: #17324d;
            --pw-muted: #667085;
            --pw-border: #e2e8df;
        }
        html, body { scroll-behavior: auto !important; }
        body { margin: 0 !important; }
        [data-testid="stAppViewContainer"] { overflow-x: hidden; }
        [data-testid="stHeader"] { height: 2.9rem !important; background: rgba(255,255,255,.88) !important; backdrop-filter: blur(10px); }
        [data-testid="stToolbar"] { top: .35rem !important; }
        .block-container { max-width: 1380px; padding-top: 3.9rem !important; padding-bottom: 4rem !important; }
        [data-testid="stSidebar"] > div:first-child { padding-top: 3.25rem !important; }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: .6rem !important; }
        .pw-page-top { scroll-margin-top: 4rem; }
        .pw-section-heading { scroll-margin-top: 4rem !important; }

        .stApp { background:
            radial-gradient(circle at 8% 0%, rgba(232,184,75,.13), transparent 22%),
            radial-gradient(circle at 96% 8%, rgba(22,139,134,.10), transparent 20%),
            var(--pw-bg); }
        [data-testid="stSidebar"] { background: linear-gradient(180deg,#073b24 0%,#0e5a36 48%,#123f2b 100%); border-right: 0; box-shadow: 10px 0 35px rgba(7,59,36,.16); }
        [data-testid="stSidebar"]::before { content:""; position:absolute; inset:0; pointer-events:none; background: radial-gradient(circle at 15% 8%,rgba(247,207,97,.22),transparent 23%), radial-gradient(circle at 90% 72%,rgba(74,190,145,.16),transparent 26%); }
        [data-testid="stSidebar"] [role="radiogroup"] { gap:.35rem !important; }
        [data-testid="stSidebar"] [role="radiogroup"] label { border-radius:14px !important; padding:.25rem .55rem !important; transition:transform .18s ease,background .18s ease,box-shadow .18s ease; }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover { background:rgba(255,255,255,.10) !important; transform:translateX(3px); }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:linear-gradient(90deg,rgba(247,207,97,.24),rgba(255,255,255,.10)) !important; box-shadow:inset 3px 0 #f7cf61,0 8px 20px rgba(0,0,0,.10); }
        [data-testid="stSidebar"] * { color: #f5fff7; }
        [data-testid="stSidebar"] .block-container { padding: 1rem .8rem; }
        [data-testid="stSidebar"] .muted, [data-testid="stSidebar"] small { color: #c9e5d0 !important; }
        [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.16); }
        [data-testid="stSidebar"] div[role="radiogroup"] { gap: .35rem; }
        [data-testid="stSidebar"] div[role="radiogroup"] label {
            border-radius: 13px; padding: .42rem .5rem; margin: 0; border: 1px solid transparent;
            background: rgba(255,255,255,.045); transition: .18s ease;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: rgba(255,255,255,.11); }
        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(90deg,rgba(232,184,75,.26),rgba(121,184,74,.22));
            border-color: rgba(232,184,75,.38); box-shadow: 0 6px 18px rgba(0,0,0,.12);
        }
        [data-testid="stSidebar"] div[role="radiogroup"] input { opacity: 0; width: 0; margin: 0; }
        [data-testid="stSidebar"] .stButton > button { background: rgba(255,255,255,.09); color: #fff; border: 1px solid rgba(255,255,255,.14); }
        h1,h2,h3,h4 { color: var(--pw-ink); letter-spacing: -.035em; }
        .eyebrow,.pw-kicker { color: var(--pw-green); font-size: .72rem; font-weight: 850; letter-spacing: .13em; text-transform: uppercase; }
        .muted,.pw-section-sub { color: var(--pw-muted); font-size: .9rem; line-height: 1.5; }
        .pw-brand-mark { width:44px;height:44px;border-radius:14px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,var(--pw-green),var(--pw-lime));color:#fff;font-size:1.35rem;font-weight:900;box-shadow:0 8px 20px rgba(47,143,70,.28); }
        .pw-card { background:#fff;border:1px solid var(--pw-border);border-radius:20px;padding:1.1rem;box-shadow:0 12px 30px rgba(31,58,39,.07); }
        .pw-card-green { background:linear-gradient(135deg,#edf9ef,#fff); border-color:#d2ead6; }
        .pw-card-gold { background:linear-gradient(135deg,#fff9e8,#fff); border-color:#f1ddb1; }
        .pw-card-blue { background:linear-gradient(135deg,#eff8ff,#fff); border-color:#cfe7fa; }
        .pw-card-teal { background:linear-gradient(135deg,#ecfbf7,#fff); border-color:#cdeee6; }
        .pw-card-orange { background:linear-gradient(135deg,#fff2e8,#fff); border-color:#f4d3bd; }
        .pw-card-purple { background:linear-gradient(135deg,#f4f1ff,#fff); border-color:#ddd5ff; }
        .pw-price { font-size:clamp(2rem,4vw,3.1rem);font-weight:900;color:var(--pw-deep);line-height:1.02; }
        .pw-price-gold { color:#a76800; }
        .pw-section-heading { margin:1.35rem 0 .8rem; }
        .pw-section-title { font-size:1.45rem;font-weight:850;color:var(--pw-ink);margin:.16rem 0; }
        .pw-image-card { position:relative;height:145px;border-radius:18px;overflow:hidden;background:linear-gradient(135deg,#dbeedb,#f8edd2);border:1px solid rgba(20,83,45,.12);box-shadow:0 8px 20px rgba(31,58,39,.08); }
        .pw-image-card img { width:100%;height:100%;object-fit:cover;display:block; }
        .pw-image-card::after { content:"";position:absolute;inset:0;background:linear-gradient(180deg,transparent 38%,rgba(8,32,20,.74));pointer-events:none; }
        .pw-image-overlay { position:absolute;z-index:2;left:.7rem;right:.7rem;bottom:.55rem;color:#fff; }
        .pw-image-label { font-weight:850;font-size:.92rem;text-shadow:0 2px 8px rgba(0,0,0,.3); }
        .pw-image-sub { font-size:.72rem;opacity:.9; }
        .pw-image-fallback { background:linear-gradient(135deg,#e6f4e8,#fff2c8); }
        .pw-image-fallback::before { content:"🌾";position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:3rem; }
        .pw-hero { border-radius:28px;padding:1rem;background:linear-gradient(135deg,#0f512f 0%,#287746 48%,#e6b64b 150%);box-shadow:0 20px 45px rgba(20,83,45,.22);margin-bottom:1.2rem;color:#fff; }
        .pw-hero h1,.pw-hero h2,.pw-hero p { color:#fff; }
        .pw-hero-inner { display:grid;grid-template-columns:1.02fr .98fr;gap:1rem;align-items:stretch; }
        .pw-hero-copy { padding:2rem 1.6rem 2rem 1.8rem;display:flex;flex-direction:column;justify-content:center; }
        .pw-hero-copy h1 { font-size:clamp(2.4rem,5vw,4.4rem);line-height:.95;margin:.35rem 0 .75rem; }
        .pw-hero-copy p { font-size:1.05rem;max-width:650px;opacity:.9; }
        .pw-hero-media { min-height:330px;border-radius:22px;overflow:hidden;position:relative;background:#d8ead9; }
        .pw-hero-media img { width:100%;height:100%;min-height:330px;object-fit:cover;display:block; }
        .pw-floating-price { position:absolute;right:1rem;bottom:1rem;background:rgba(255,255,255,.94);backdrop-filter:blur(12px);color:var(--pw-ink);border-radius:17px;padding:.8rem 1rem;min-width:175px;box-shadow:0 14px 30px rgba(0,0,0,.18); }
        .pw-floating-price .label { font-size:.7rem;font-weight:850;color:var(--pw-green);letter-spacing:.08em; }
        .pw-floating-price .value { font-size:1.55rem;font-weight:900;color:var(--pw-deep); }
        .pw-floating-price .sub { font-size:.72rem;color:var(--pw-muted); }
        .pw-stat-grid { display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.8rem;margin:1rem 0; }
        .pw-stat { border-radius:18px;padding:1rem;border:1px solid rgba(0,0,0,.05);min-height:120px;box-shadow:0 8px 20px rgba(31,58,39,.06); }
        .pw-stat .icon { font-size:1.35rem; }
        .pw-stat .label { font-size:.73rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em;color:#657166; }
        .pw-stat .value { font-size:1.65rem;font-weight:900;color:var(--pw-ink);margin:.25rem 0; }
        .pw-stat.green { background:linear-gradient(135deg,#e9f8ed,#fff); }
        .pw-stat.gold { background:linear-gradient(135deg,#fff6d9,#fff); }
        .pw-stat.blue { background:linear-gradient(135deg,#eaf5ff,#fff); }
        .pw-stat.orange { background:linear-gradient(135deg,#fff0e5,#fff); }
        .pw-stat.teal { background:linear-gradient(135deg,#e8faf5,#fff); }
        .pw-stat.purple { background:linear-gradient(135deg,#f4f1ff,#fff); }
        .pw-form-shell { border-radius:24px;padding:1.1rem;background:linear-gradient(135deg,#fffdf7,#eef8f0);border:1px solid #dfeadf;box-shadow:0 12px 30px rgba(31,58,39,.06); }
        .pw-result { border-radius:24px;padding:1.25rem;background:linear-gradient(135deg,#f0fbf2,#fff8df);border:1px solid #dce9d3;box-shadow:0 14px 35px rgba(31,58,39,.08); }
        .pw-result .big { font-size:clamp(2.7rem,5vw,4.2rem);font-weight:950;color:var(--pw-deep);line-height:1; }
        .pw-chip { display:inline-flex;align-items:center;gap:.35rem;border-radius:999px;padding:.32rem .65rem;font-size:.74rem;font-weight:800;background:#edf6ef;color:#1c6b39;border:1px solid #d6e9d9;margin-right:.35rem; }
        .pw-callout { border-radius:18px;padding:.85rem 1rem;background:#fff;border:1px dashed #d6dfd7; }
        .pw-audit-banner { border-radius:22px;padding:1rem;background:linear-gradient(135deg,#161c43,#5a4bb7);color:#fff;box-shadow:0 15px 32px rgba(56,49,128,.18); }
        .pw-audit-banner * { color:#fff !important; }
        .pw-market-card { border-radius:20px;padding:.95rem;background:#fff;border:1px solid var(--pw-border);box-shadow:0 10px 24px rgba(31,58,39,.06); }
        .pw-market-card h4 { margin:.35rem 0; }
        .pw-order-card { border-radius:22px;padding:1rem;background:linear-gradient(135deg,#edf9ff,#f4fffb);border:1px solid #cfe8ee;box-shadow:0 10px 26px rgba(22,104,128,.08); }
        .pw-profile { border-radius:25px;padding:1.15rem;background:linear-gradient(135deg,#0f512f,#2e8148 55%,#e4b44a);color:#fff;box-shadow:0 18px 36px rgba(20,83,45,.18); }
        .pw-profile h2,.pw-profile .muted { color:#fff; }
        .pw-avatar-img { width:100px;height:100px;border-radius:50%;object-fit:cover;border:5px solid rgba(255,255,255,.82);box-shadow:0 8px 24px rgba(0,0,0,.18); }
        .pw-note { font-size:.76rem;color:#69766c;background:#fffdf6;border-left:4px solid var(--pw-gold);padding:.65rem .8rem;border-radius:10px; }
        .stButton > button { border-radius:12px;font-weight:800;min-height:2.65rem; }
        .stButton > button[kind="primary"] { background:linear-gradient(135deg,var(--pw-green),var(--pw-deep));border-color:var(--pw-green);box-shadow:0 8px 18px rgba(47,143,70,.18); }
        div[data-testid="stMetric"] { background:#fff;border:1px solid var(--pw-border);border-radius:16px;padding:.72rem .85rem;box-shadow:0 6px 16px rgba(31,58,39,.04); }
        div[data-testid="stMetricValue"] { color:var(--pw-deep); }
        [data-testid="stVerticalBlockBorderWrapper"] { border-radius:20px !important;border-color:#e1e8df !important;box-shadow:0 8px 22px rgba(31,58,39,.05); }
        .stTabs [data-baseweb="tab-list"] { gap:.4rem; }
        .stTabs [data-baseweb="tab"] { border-radius:12px 12px 0 0;padding:.55rem .9rem; }
        .pw-landing { position:relative; overflow:hidden; }
        .pw-landing::before,.pw-landing::after { content:""; position:absolute; pointer-events:none; border-radius:999px; filter:blur(2px); opacity:.35; animation:pw-drift 10s ease-in-out infinite alternate; }
        .pw-landing::before { width:220px;height:220px;background:radial-gradient(circle,#f6c453,transparent 68%);right:-70px;top:60px; }
        .pw-landing::after { width:180px;height:180px;background:radial-gradient(circle,#4bb477,transparent 68%);left:-60px;bottom:40px;animation-delay:-3s; }
        .pw-landing-hero { position:relative; border-radius:32px; overflow:hidden; min-height:540px; background:linear-gradient(120deg,#0b3d25 0%,#145c37 43%,#d89b2b 130%); box-shadow:0 26px 60px rgba(17,67,40,.24); }
        .pw-landing-hero::before { content:""; position:absolute; inset:0; background:linear-gradient(90deg,rgba(7,37,23,.88) 0%,rgba(11,70,41,.72) 43%,rgba(8,28,18,.05) 72%); z-index:1; }
        .pw-landing-copy { position:relative; z-index:2; width:54%; min-height:540px; padding:4.2rem 2.6rem 3rem; display:flex; flex-direction:column; justify-content:center; }
        .pw-landing-copy h1 { color:#fff; font-size:clamp(3rem,6.3vw,6.1rem); line-height:.9; margin:.55rem 0 1rem; letter-spacing:-.055em; }
        .pw-landing-copy p { color:rgba(255,255,255,.9); max-width:620px; font-size:1.1rem; line-height:1.6; }
        .pw-landing-image { position:absolute; inset:0 0 0 45%; z-index:0; }
        .pw-landing-image img { width:100%;height:100%;object-fit:cover; display:block; animation:pw-zoom 14s ease-in-out infinite alternate; transform-origin:center; }
        .pw-landing-badge { display:inline-flex; width:max-content; align-items:center; gap:.45rem; padding:.42rem .7rem; border:1px solid rgba(255,255,255,.25); background:rgba(255,255,255,.12); backdrop-filter:blur(10px); color:#fff; border-radius:999px; font-weight:800; font-size:.76rem; }
        .pw-landing-cta { display:flex; flex-wrap:wrap; gap:.7rem; margin-top:1.25rem; }
        .pw-landing-cta a { text-decoration:none; padding:.78rem 1.1rem; border-radius:14px; font-weight:850; display:inline-flex; align-items:center; justify-content:center; transition:transform .18s ease,box-shadow .18s ease,background .18s ease; }
        .pw-landing-cta a:hover { transform:translateY(-2px); box-shadow:0 12px 24px rgba(0,0,0,.16); }
        .pw-cta-primary { background:linear-gradient(135deg,#f7cf61,#e49a2d); color:#17324d !important; }
        .pw-cta-secondary { background:rgba(255,255,255,.12); color:#fff !important; border:1px solid rgba(255,255,255,.25); }
        .pw-ai-pulse { display:inline-flex; width:9px;height:9px;border-radius:50%;background:#7fe28e;box-shadow:0 0 0 0 rgba(127,226,142,.55); animation:pw-pulse 2s infinite; }
        .pw-feature-card { min-height:178px; transition:transform .2s ease,box-shadow .2s ease; }
        .pw-feature-card:hover { transform:translateY(-4px); box-shadow:0 18px 34px rgba(31,58,39,.11); }
        .pw-auth-shell { border-radius:28px; padding:1.2rem; background:linear-gradient(135deg,rgba(255,255,255,.98),rgba(255,250,239,.96)); border:1px solid #e4eadf; box-shadow:0 24px 50px rgba(31,58,39,.13); }
        .pw-role-card { border-radius:16px; padding:.8rem; border:1px solid #dce8de; background:linear-gradient(135deg,#f4fbf4,#fff); }
        .pw-greeting { border-radius:22px; padding:1.05rem 1.2rem; background:linear-gradient(100deg,#eff9f1,#fff7df 55%,#edf7ff); border:1px solid #dce8de; box-shadow:0 10px 24px rgba(31,58,39,.06); margin-bottom:1rem; }
        .pw-greeting-title { font-size:clamp(1.45rem,3vw,2.1rem); font-weight:900; color:var(--pw-ink); margin:0; }
        .pw-greeting-sub { color:#68756c; margin-top:.2rem; }
        .pw-dashboard-hero .pw-hero-copy { padding-top:1.45rem; padding-bottom:1.45rem; }
        .pw-dashboard-hero .pw-hero-copy h1 { font-size:clamp(2.2rem,4.5vw,4rem); }
        .pw-dashboard-hero .pw-hero-media { min-height:300px; }
        .pw-dashboard-hero .pw-hero-media img { min-height:300px; }
        .pw-page-top { animation: pw-page-in .42s ease both; }
        .pw-card,.pw-stat,.pw-market-card,.pw-order-card,.pw-form-shell,.pw-result,.pw-callout,.pw-feature-card { animation: pw-rise .48s ease both; }
        .pw-stat:hover,.pw-market-card:hover,.pw-order-card:hover,.pw-card:hover { transform:translateY(-3px); box-shadow:0 18px 34px rgba(31,58,39,.12); transition:transform .2s ease,box-shadow .2s ease; }
        .pw-price { text-shadow: 0 5px 20px rgba(232,184,75,.22); }
        .pw-form-shell { background:linear-gradient(135deg,#fffdf7 0%,#eff9f1 48%,#edf7ff 100%); background-size:180% 180%; animation:pw-gradient 9s ease infinite; }
        .pw-ai-pulse { animation:pw-pulse 2s infinite; }
        .pw-image-card:hover img { transform:scale(1.055); transition:transform .45s ease; }
        .pw-image-card img { transition:transform .45s ease; }
        .pw-section-heading { scroll-margin-top:1rem; }
        @keyframes pw-page-in { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:none} }
        @keyframes pw-rise { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:none} }
        @keyframes pw-gradient { 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }
        @keyframes pw-zoom { from{transform:scale(1.02)} to{transform:scale(1.08)} }
        @keyframes pw-drift { from{transform:translate3d(0,0,0)} to{transform:translate3d(12px,-16px,0)} }
        @keyframes pw-pulse { 0%{box-shadow:0 0 0 0 rgba(127,226,142,.5)} 70%{box-shadow:0 0 0 10px rgba(127,226,142,0)} 100%{box-shadow:0 0 0 0 rgba(127,226,142,0)} }
        .pw-audit-banner { border-radius:24px; padding:1.2rem; color:#fff; background:linear-gradient(135deg,#20205d 0%,#4b3ca7 45%,#197c92 100%); box-shadow:0 18px 42px rgba(53,47,130,.22); margin-bottom:1rem; overflow:hidden; position:relative; }
        .pw-audit-banner::before { content:""; position:absolute; width:240px; height:240px; right:25%; top:-150px; border-radius:50%; background:rgba(255,255,255,.10); filter:blur(3px); }
        .pw-audit-banner h2,.pw-audit-banner .pw-kicker { color:#fff !important; }
        .pw-audit-sub { color:rgba(255,255,255,.82); line-height:1.5; }
        .pw-audit-image { height:170px; border-radius:18px; overflow:hidden; box-shadow:0 14px 30px rgba(0,0,0,.20); }
        .pw-audit-image img { width:100%; height:100%; object-fit:cover; }
        .pw-verified-banner,.pw-failed-banner { display:flex; align-items:center; gap:.85rem; padding:1rem 1.15rem; border-radius:18px; margin:1rem 0; animation:pw-rise .45s ease both; }
        .pw-verified-banner { background:linear-gradient(135deg,#e9faee,#f5fff7); border:1px solid #bce7c7; color:#14532d; box-shadow:0 12px 26px rgba(25,122,57,.10); }
        .pw-failed-banner { background:linear-gradient(135deg,#fff0ed,#fff8f5); border:1px solid #f0c2b8; color:#9d2d20; box-shadow:0 12px 26px rgba(180,65,45,.09); }
        .pw-verified-icon,.pw-failed-icon { width:42px; height:42px; flex:0 0 42px; display:flex; align-items:center; justify-content:center; border-radius:50%; font-weight:950; font-size:1.35rem; }
        .pw-verified-icon { background:#1f9d4b; color:#fff; box-shadow:0 0 0 7px rgba(31,157,75,.12); }
        .pw-failed-icon { background:#d94835; color:#fff; }
        .pw-verified-title,.pw-failed-title { font-weight:950; letter-spacing:.07em; font-size:.78rem; }
        .pw-audit-record { background:linear-gradient(145deg,#f7f5ff,#eef8ff); border:1px solid #d9d8f4; border-radius:22px; padding:1.1rem; margin:1rem 0; box-shadow:0 14px 30px rgba(49,55,126,.08); }
        .pw-audit-record-top { display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; }
        .pw-audit-record h3 { margin:.35rem 0 0; color:#25235d; }
        .pw-audit-block { display:inline-block; padding:.25rem .55rem; border-radius:999px; background:#e4e0ff; color:#5143b8; font-size:.7rem; font-weight:900; letter-spacing:.08em; }
        .pw-verified-pill { padding:.35rem .65rem; border-radius:999px; background:#dff7e7; color:#18733a; font-weight:800; font-size:.78rem; }
        .pw-hash-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.7rem; margin-top:1rem; }
        .pw-hash-grid > div { background:rgba(255,255,255,.72); border:1px solid rgba(111,91,208,.14); border-radius:14px; padding:.75rem; min-width:0; }
        .pw-hash-grid span { display:block; color:#68708a; font-size:.72rem; font-weight:800; text-transform:uppercase; letter-spacing:.06em; margin-bottom:.3rem; }
        .pw-hash-grid code { display:block; color:#28265e; font-size:.73rem; overflow-wrap:anywhere; white-space:normal; }
        .pw-verified-strong { border-width:2px; }
        .pw-hero, .pw-landing-hero { position:relative; overflow:hidden; }
        .pw-hero::before { content:""; position:absolute; width:360px; height:360px; right:-150px; top:-190px; border-radius:50%; background:radial-gradient(circle,rgba(247,207,97,.30),transparent 68%); animation:pw-drift 8s ease-in-out infinite alternate; pointer-events:none; }
        .pw-hero-float { position:absolute; z-index:3; pointer-events:none; font-size:1.25rem; filter:drop-shadow(0 8px 12px rgba(0,0,0,.16)); opacity:.8; animation:pw-float 5.5s ease-in-out infinite alternate; }
        .pw-float-a { left:3%; top:12%; animation-delay:-1.2s; }
        .pw-float-b { right:43%; top:9%; animation-delay:-2.8s; }
        .pw-float-c { right:5%; bottom:12%; animation-delay:-4s; }
        .pw-landing-hero { background-size:170% 170%; animation:pw-gradient 12s ease infinite; }
        .pw-landing-image::after { content:""; position:absolute; inset:0; background:linear-gradient(120deg,rgba(4,34,20,.18),rgba(4,34,20,0) 55%,rgba(247,190,65,.12)); pointer-events:none; }
        .pw-auth-shell { background:linear-gradient(135deg,rgba(255,255,255,.97),rgba(255,247,228,.97) 48%,rgba(237,249,255,.97)); border:1px solid rgba(42,112,72,.18); position:relative; overflow:hidden; }
        .pw-auth-shell::before { content:""; position:absolute; width:240px;height:240px; right:-100px; top:-120px;border-radius:50%;background:radial-gradient(circle,rgba(110,91,208,.17),transparent 68%); animation:pw-drift 7s ease-in-out infinite alternate; pointer-events:none; }
        .pw-form-shell, .pw-market-card, .pw-order-card, .pw-card, .pw-stat { backdrop-filter:blur(5px); }
        .pw-market-card { background:linear-gradient(145deg,#fffdf6 0%,#f1fbf4 52%,#edf7ff 100%); border:1px solid rgba(55,112,75,.14); transition:transform .22s ease,box-shadow .22s ease,border-color .22s ease; }
        .pw-market-card:hover { border-color:rgba(232,184,75,.5); }
        .pw-order-card { background:linear-gradient(135deg,#eef9ff 0%,#eafbf5 52%,#fff7df 100%); }
        .pw-context-ribbon { border-radius:18px; padding:.9rem 1rem; margin-bottom:.75rem; background:linear-gradient(100deg,#0f5d36,#178a75 55%,#e3aa32); color:#fff; box-shadow:0 12px 28px rgba(20,83,45,.15); }
        .pw-context-ribbon span { display:block; font-size:.68rem; font-weight:900; letter-spacing:.12em; opacity:.86; }
        .pw-context-ribbon b { display:block; font-size:1.05rem; margin:.12rem 0; }
        .pw-context-ribbon small { opacity:.86; }
        .pw-sidebar-user { display:flex;gap:.6rem;align-items:center;padding:.7rem .65rem;border-radius:16px;background:linear-gradient(135deg,rgba(255,255,255,.10),rgba(247,207,97,.10));border:1px solid rgba(255,255,255,.10); }
        .pw-sidebar-avatar { width:38px;height:38px;border-radius:12px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#f7cf61,#5fbd73);color:#17324d;font-weight:950;box-shadow:0 8px 18px rgba(0,0,0,.15); }
        .pw-avatar-wrap { width:76px;height:76px;border-radius:22px;overflow:hidden;position:relative;flex:0 0 76px;background:linear-gradient(135deg,#f7cf61,#2f8f46);box-shadow:0 12px 26px rgba(20,83,45,.20); }
        .pw-avatar-img { width:100%;height:100%;object-fit:cover;display:block; }
        .pw-avatar-fallback { display:none;position:absolute;inset:0;align-items:center;justify-content:center;color:#17324d;font-size:1.55rem;font-weight:950; }
        .pw-message { max-width:78%; padding:.75rem .9rem; border-radius:18px; margin:.55rem 0; box-shadow:0 8px 18px rgba(31,58,39,.07); animation:pw-rise .35s ease both; }
        .pw-message-self { margin-left:auto; background:linear-gradient(135deg,#dff7e8,#e7f8ff); border:1px solid #c5e9d2; border-bottom-right-radius:6px; }
        .pw-message-other { margin-right:auto; background:linear-gradient(135deg,#fff7df,#f5efff); border:1px solid #eadfbf; border-bottom-left-radius:6px; }
        .pw-message-meta { font-size:.7rem;font-weight:850;color:#657166;margin-bottom:.25rem; }
        .pw-message-text { color:#17324d;line-height:1.5; }
        .pw-audit-page { animation:pw-page-in .45s ease both; }
        .pw-audit-page .pw-hash-grid > div { transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease; }
        .pw-audit-page .pw-hash-grid > div:hover { transform:translateY(-2px);box-shadow:0 10px 22px rgba(49,55,126,.10);border-color:rgba(111,91,208,.3); }
        @keyframes pw-float { from{transform:translate3d(0,0,0) rotate(-3deg)} to{transform:translate3d(8px,-12px,0) rotate(4deg)} }
        
        /* ===== Thanjavur visual layer v3 — presentation only ===== */
        html { scroll-behavior:smooth !important; }
        .pw-location-ribbon { display:inline-flex; width:max-content; margin:.75rem 0 .25rem; padding:.42rem .7rem; border-radius:999px; border:1px solid rgba(247,207,97,.55); background:linear-gradient(90deg,rgba(244,200,74,.22),rgba(21,148,71,.18)); color:#fff; font-size:.68rem; font-weight:900; letter-spacing:.12em; box-shadow:0 8px 20px rgba(0,0,0,.10); }
        .pw-hero-pill-row { display:flex; flex-wrap:wrap; gap:.45rem; margin-top:1rem; }
        .pw-hero-pill-row span { padding:.42rem .68rem; border-radius:999px; color:#fff; background:rgba(255,255,255,.11); border:1px solid rgba(255,255,255,.22); backdrop-filter:blur(9px); font-size:.72rem; font-weight:800; }
        .pw-learn-button { border:2px solid rgba(247,207,97,.62) !important; background:linear-gradient(135deg,rgba(255,255,255,.16),rgba(53,185,214,.12)) !important; box-shadow:inset 0 1px rgba(255,255,255,.12),0 8px 20px rgba(0,0,0,.10); }
        .pw-learn-button:hover { border-color:#f7cf61 !important; background:linear-gradient(135deg,rgba(244,200,74,.22),rgba(53,185,214,.18)) !important; }
        .pw-tamil-pride { display:grid; grid-template-columns:1.05fr .95fr; min-height:310px; border-radius:28px; overflow:hidden; border:2px solid #d6a52d; background:linear-gradient(120deg,#073b24 0%,#0b6b3c 47%,#e2ad36 130%); box-shadow:0 22px 48px rgba(7,59,36,.20); position:relative; }
        .pw-tamil-pride::before { content:""; position:absolute; width:330px;height:330px; border-radius:50%; right:31%; top:-190px; background:radial-gradient(circle,rgba(255,255,255,.15),transparent 68%); animation:pw-drift 9s ease-in-out infinite alternate; pointer-events:none; }
        .pw-pride-copy { padding:2rem 2.2rem; display:flex; flex-direction:column; justify-content:center; position:relative; z-index:2; color:#fff; }
        .pw-pride-copy h2 { color:#fff; font-size:clamp(1.9rem,4vw,3.4rem); line-height:1; margin:.45rem 0 .7rem; }
        .pw-pride-copy p { color:rgba(255,255,255,.88); max-width:650px; line-height:1.6; }
        .pw-pride-kicker { color:#f7d56b; font-size:.7rem; font-weight:900; letter-spacing:.14em; }
        .pw-pride-tags { display:flex; flex-wrap:wrap; gap:.45rem; margin-top:.8rem; }
        .pw-pride-tags span { border:1px solid rgba(255,255,255,.22); background:rgba(255,255,255,.10); border-radius:999px; padding:.38rem .62rem; font-size:.73rem; font-weight:800; backdrop-filter:blur(8px); }
        .pw-pride-photo { min-height:310px; position:relative; overflow:hidden; border-left:2px solid rgba(255,255,255,.18); }
        .pw-pride-photo::after { content:""; position:absolute; inset:0; background:linear-gradient(90deg,rgba(7,59,36,.32),rgba(7,59,36,0) 55%,rgba(244,200,74,.12)); pointer-events:none; }
        .pw-pride-photo img { width:100%;height:100%;object-fit:cover;display:block; animation:pw-hero-zoom 13s ease-in-out infinite alternate; }
        .pw-map-shell { position:relative; border-radius:26px; overflow:hidden; padding:.45rem; background:linear-gradient(135deg,#073b24,#0c9b8b 50%,#7b4bd4); border:2px solid rgba(123,75,212,.62); box-shadow:0 22px 46px rgba(16,42,67,.18),0 0 34px rgba(53,185,214,.12); animation:pw-map-glow 5s ease-in-out infinite alternate; }
        .pw-map-overlay { position:absolute; z-index:500; left:1rem; top:1rem; padding:.62rem .78rem; border-radius:14px; background:rgba(8,30,28,.80); color:#fff; border:1px solid rgba(244,200,74,.48); box-shadow:0 10px 24px rgba(0,0,0,.18); backdrop-filter:blur(10px); pointer-events:none; }
        .pw-map-overlay span { display:block; font-size:.62rem; font-weight:900; letter-spacing:.12em; color:#8fe8d7; }
        .pw-map-overlay b { display:block; margin-top:.18rem; font-size:.86rem; color:#f7d56b; }
        .pw-demo-disclosure { margin-top:1rem; border-style:solid !important; border-color:#d6a52d !important; }
        .pw-landing-cta a { min-height:46px; box-sizing:border-box; border:2px solid transparent; }
        .pw-landing-cta .pw-cta-primary { border-color:#f7cf61; box-shadow:0 12px 25px rgba(0,0,0,.16),0 0 0 1px rgba(255,255,255,.08) inset; }
        .pw-landing-cta .pw-cta-secondary { box-shadow:0 12px 25px rgba(0,0,0,.12); }
        @keyframes pw-map-glow { from{box-shadow:0 22px 46px rgba(16,42,67,.16),0 0 18px rgba(53,185,214,.08)} to{box-shadow:0 24px 52px rgba(16,42,67,.21),0 0 34px rgba(123,75,212,.18)} }
        @media (max-width: 850px) { .pw-tamil-pride { grid-template-columns:1fr; } .pw-pride-photo { min-height:230px; border-left:0; border-top:2px solid rgba(255,255,255,.18); } }
        @media (max-width: 600px) { .pw-location-ribbon { font-size:.58rem; letter-spacing:.08em; } .pw-pride-copy { padding:1.45rem; } .pw-map-overlay { left:.7rem; top:.7rem; } .pw-map-overlay b { font-size:.72rem; } }
        @media (max-width: 700px) { .pw-hash-grid { grid-template-columns:1fr; } .pw-audit-record-top { flex-direction:column; } .pw-message { max-width:92%; } }
        @media (prefers-reduced-motion: reduce) { *,*::before,*::after { animation-duration:.001ms !important; animation-iteration-count:1 !important; transition-duration:.001ms !important; scroll-behavior:auto !important; } }
        @media (max-width: 900px) { .pw-hero-inner { grid-template-columns:1fr; } .pw-stat-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .pw-landing-copy { width:68%; padding:3rem 1.8rem; } .pw-landing-image { left:32%; } .pw-landing-hero::before { background:linear-gradient(90deg,rgba(7,37,23,.92),rgba(7,37,23,.58)); } }
        @media (max-width: 600px) { .pw-stat-grid { grid-template-columns:1fr; } .pw-hero-copy { padding:1.2rem; } .pw-landing-hero { min-height:610px; } .pw-landing-copy { width:100%; min-height:610px; padding:2.2rem 1.15rem; justify-content:flex-end; } .pw-landing-image { left:0; height:58%; } .pw-landing-hero::before { background:linear-gradient(180deg,rgba(7,37,23,.18) 0%,rgba(7,37,23,.38) 42%,rgba(7,37,23,.92) 72%); } .pw-landing-copy h1 { font-size:clamp(2.7rem,14vw,4.4rem); } .pw-landing-copy p { font-size:.98rem; } }

        /* ===== AgriGuard premium visual system v2 — presentation only ===== */
        :root {
            --agri-deep:#073b24; --agri-forest:#0b5d36; --agri-emerald:#159447;
            --agri-leaf:#62b64f; --agri-gold:#f4c84a; --agri-amber:#f5a623;
            --agri-orange:#ee7d32; --agri-teal:#0c9b8b; --agri-cyan:#35b9d6;
            --agri-blue:#3d7fe0; --agri-indigo:#4b45b8; --agri-purple:#7b4bd4;
            --agri-cream:#fff8e9; --agri-paper:#fffdf8; --agri-navy:#102a43;
            --agri-line:#bfd7c4;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 7% 8%,rgba(244,200,74,.18),transparent 19%),
                radial-gradient(circle at 94% 16%,rgba(53,185,214,.14),transparent 21%),
                radial-gradient(circle at 48% 100%,rgba(21,148,71,.11),transparent 25%),
                linear-gradient(135deg,#f8fbf4 0%,#fffaf0 45%,#f2faf7 100%);
        }
        [data-testid="stAppViewContainer"]::before {
            content:""; position:fixed; inset:0; pointer-events:none; z-index:0; opacity:.32;
            background-image:radial-gradient(rgba(21,148,71,.09) 1px,transparent 1px);
            background-size:24px 24px; mask-image:linear-gradient(to bottom,black,transparent 82%);
        }
        [data-testid="stMain"] { position:relative; z-index:1; }
        .block-container { max-width:1440px !important; padding-left:clamp(1rem,3vw,3rem) !important; padding-right:clamp(1rem,3vw,3rem) !important; }

        /* Stronger section identity */
        .pw-section-heading { position:relative; padding:.35rem 0 .55rem 1rem; margin:1.55rem 0 .9rem; }
        .pw-section-accent { position:absolute; left:0; top:.35rem; bottom:.55rem; width:5px; border-radius:999px; background:linear-gradient(180deg,var(--agri-emerald),var(--agri-gold)); box-shadow:0 0 14px rgba(21,148,71,.22); }
        .pw-section-forecast .pw-section-accent, .pw-section-market-intelligence .pw-section-accent { background:linear-gradient(180deg,var(--agri-blue),var(--agri-teal)); }
        .pw-section-ai-price-result .pw-section-accent { background:linear-gradient(180deg,var(--agri-emerald),var(--agri-gold)); }
        .pw-section-sell-vs-wait .pw-section-accent { background:linear-gradient(180deg,var(--agri-orange),var(--agri-gold)); }
        .pw-section-transaction-audit .pw-section-accent { background:linear-gradient(180deg,var(--agri-purple),var(--agri-blue)); }
        .pw-section-heading .pw-section-title { font-size:clamp(1.35rem,2.3vw,1.85rem); }
        .pw-section-heading .pw-section-sub { max-width:850px; }

        /* Premium cards: clearly visible borders, not faint gray boxes */
        .pw-card, .pw-stat, .pw-market-card, .pw-order-card, .pw-form-shell, .pw-result, .pw-callout, .pw-context-ribbon {
            border-width:1.5px !important;
        }
        .pw-card:hover, .pw-stat:hover, .pw-market-card:hover, .pw-order-card:hover {
            transform:translateY(-4px); box-shadow:0 18px 38px rgba(16,42,67,.12);
        }
        .pw-card { background:linear-gradient(145deg,#fffdf8,#f6fbf6) !important; border-color:#9bc7a5 !important; transition:transform .22s ease,box-shadow .22s ease,border-color .22s ease; }
        .pw-card-green { background:linear-gradient(135deg,#dff7e6,#f8fff8) !important; border-color:#55ae6b !important; }
        .pw-card-gold { background:linear-gradient(135deg,#fff1bd,#fffaf0) !important; border-color:#e2ad27 !important; }
        .pw-card-blue { background:linear-gradient(135deg,#dff1ff,#f4fbff) !important; border-color:#56a9df !important; }
        .pw-card-teal { background:linear-gradient(135deg,#d8f7ef,#f5fffd) !important; border-color:#43b6a4 !important; }
        .pw-card-orange { background:linear-gradient(135deg,#ffe5ce,#fff9f3) !important; border-color:#e99452 !important; }
        .pw-card-purple { background:linear-gradient(135deg,#e9e2ff,#f9f7ff) !important; border-color:#8b70d7 !important; }
        .pw-stat { min-height:132px; border-color:#9bc7a5 !important; position:relative; overflow:hidden; transition:.22s ease; }
        .pw-stat::after { content:""; position:absolute; width:90px;height:90px;right:-35px;top:-35px;border-radius:50%;background:rgba(255,255,255,.36); }
        .pw-stat.green { background:linear-gradient(135deg,#bfecc8,#ecfff0) !important; border-color:#43a95c !important; }
        .pw-stat.gold { background:linear-gradient(135deg,#ffe58a,#fff7d9) !important; border-color:#d9a522 !important; }
        .pw-stat.blue { background:linear-gradient(135deg,#c7e7ff,#eaf8ff) !important; border-color:#4d9bd0 !important; }
        .pw-stat.orange { background:linear-gradient(135deg,#ffd3ad,#fff0e3) !important; border-color:#df7d3d !important; }
        .pw-stat.teal { background:linear-gradient(135deg,#bdeee5,#e8fbf7) !important; border-color:#35a997 !important; }
        .pw-stat.purple { background:linear-gradient(135deg,#ddd2ff,#f1edff) !important; border-color:#8065cc !important; }

        /* Strong form surfaces */
        .pw-form-shell { background:linear-gradient(145deg,rgba(255,253,245,.96),rgba(226,248,232,.96)) !important; border:2px solid #8fbea0 !important; box-shadow:0 18px 40px rgba(20,83,45,.09); }
        .pw-result { background:linear-gradient(135deg,#dff8e7 0%,#fff3bd 58%,#edf8ff 100%) !important; border:2px solid #62b97a !important; }
        .pw-callout { background:linear-gradient(135deg,#fff7dc,#eefaf1) !important; border:1.5px solid #d4ae4a !important; }

        /* Hero / landing: more saturated and alive */
        .pw-landing-hero, .pw-hero {
            background:linear-gradient(125deg,#06351f 0%,#0b6b3c 38%,#159447 58%,#e3aa32 125%) !important;
            background-size:180% 180% !important; animation:pw-gradient 11s ease-in-out infinite;
            border:2px solid rgba(247,207,97,.38);
        }
        .pw-hero { box-shadow:0 24px 55px rgba(7,59,36,.28),inset 0 1px rgba(255,255,255,.12); }
        .pw-hero-media, .pw-landing-image { border:2px solid rgba(255,255,255,.25); box-shadow:0 16px 35px rgba(0,0,0,.22); }
        .pw-hero-media img, .pw-landing-image img { animation:pw-hero-zoom 10s ease-in-out infinite alternate; transform-origin:center; }
        .pw-landing-copy h1 { text-shadow:0 8px 30px rgba(0,0,0,.18); }
        .pw-floating-price { border:2px solid rgba(244,200,74,.65); box-shadow:0 16px 36px rgba(0,0,0,.22); }

        /* Image surfaces */
        .pw-image-card { border:2px solid #9fc8aa !important; box-shadow:0 12px 28px rgba(16,42,67,.11); transition:transform .25s ease,box-shadow .25s ease,border-color .25s ease; }
        .pw-image-card:hover { transform:translateY(-4px); border-color:#e1b33d !important; box-shadow:0 18px 34px rgba(16,42,67,.16); }
        .pw-image-card:hover img { transform:scale(1.07) !important; }

        /* Streamlit controls */
        .stButton > button {
            border-radius:14px !important; min-height:2.65rem; font-weight:850 !important;
            border:1.5px solid #76a884 !important; background:linear-gradient(135deg,#198c48,#0d6b38) !important;
            color:#fff !important; box-shadow:0 8px 18px rgba(21,148,71,.18);
            transition:transform .18s ease,box-shadow .18s ease,filter .18s ease,border-color .18s ease !important;
        }
        .stButton > button:hover { transform:translateY(-2px); box-shadow:0 12px 24px rgba(21,148,71,.25) !important; filter:saturate(1.08); border-color:#f0c34a !important; }
        .stButton > button:focus { box-shadow:0 0 0 3px rgba(53,185,214,.18),0 12px 24px rgba(21,148,71,.2) !important; }
        [data-testid="stFormSubmitButton"] button { background:linear-gradient(135deg,#0b6b3c,#159447) !important; }
        .stButton button[kind="secondary"] { background:linear-gradient(135deg,#fff4c9,#f4b52e) !important; color:#513500 !important; border-color:#d89d19 !important; }
        .stButton button[disabled] { opacity:.55 !important; }

        /* Inputs: visible colored frames + focus glow */
        [data-baseweb="input"], [data-baseweb="select"], [data-baseweb="textarea"] {
            border-radius:13px !important; background:#fffdf8 !important; border:1.5px solid #8eb9a0 !important;
            box-shadow:0 4px 12px rgba(16,42,67,.035); transition:border-color .18s ease,box-shadow .18s ease;
        }
        [data-baseweb="input"]:focus-within, [data-baseweb="select"]:focus-within, [data-baseweb="textarea"]:focus-within {
            border-color:#159447 !important; box-shadow:0 0 0 3px rgba(53,185,214,.16),0 8px 18px rgba(21,148,71,.08) !important;
        }
        [data-testid="stTextInput"] label, [data-testid="stNumberInput"] label, [data-testid="stSelectbox"] label, [data-testid="stTextArea"] label { color:#214d34 !important; font-weight:800 !important; }

        /* Tables / dataframes */
        [data-testid="stDataFrame"] { border:1.5px solid #79ad89 !important; border-radius:16px !important; overflow:hidden; box-shadow:0 12px 25px rgba(16,42,67,.07); background:#fff; }
        [data-testid="stDataFrame"] [role="columnheader"] { background:linear-gradient(135deg,#0b6b3c,#159447) !important; color:#fff !important; font-weight:850 !important; }

        /* Audit: distinct purple/indigo technology identity */
        .pw-audit-page { background:linear-gradient(145deg,rgba(244,242,255,.68),rgba(238,248,255,.56)); border-radius:26px; padding:.2rem .2rem 1rem; }
        .pw-audit-banner { border:2px solid #7565d5 !important; background:linear-gradient(120deg,#17164f 0%,#4b3ca7 48%,#157fa0 100%) !important; }
        .pw-audit-record { border:2px solid #8876dc !important; background:linear-gradient(145deg,#eeeaff,#e9f7ff) !important; }
        .pw-hash-grid > div { border:1.5px solid #9b8de0 !important; background:rgba(255,255,255,.82) !important; }
        .pw-verified-banner { border:2px solid #38a85b !important; }
        .pw-failed-banner { border:2px solid #d94835 !important; }

        /* Sidebar premium navigation */
        [data-testid="stSidebar"] { background:linear-gradient(180deg,#052f1d 0%,#075d36 43%,#073b49 100%) !important; }
        [data-testid="stSidebar"] .stButton > button { border:1.5px solid rgba(247,207,97,.38) !important; background:linear-gradient(135deg,rgba(247,207,97,.16),rgba(255,255,255,.08)) !important; }
        .pw-sidebar-user { border:1.5px solid rgba(247,207,97,.34) !important; box-shadow:0 12px 26px rgba(0,0,0,.12); }

        /* Gentle page entrance / motion */
        .pw-page-top, .pw-section-heading, .pw-card, .pw-stat, .pw-image-card, .pw-market-card, .pw-order-card { animation:pw-rise .48s ease both; }
        @keyframes pw-hero-zoom { from{transform:scale(1.00)} to{transform:scale(1.045)} }

        @media (max-width: 900px) {
            .block-container { padding-left:1rem !important; padding-right:1rem !important; }
            .pw-hero-copy h1 { font-size:clamp(2.2rem,8vw,3.7rem); }
        }
        @media (max-width: 600px) {
            .pw-section-heading { padding-left:.8rem; }
            .pw-stat { min-height:112px; }
            .stButton > button { min-height:2.8rem; }
        }
        @media (prefers-reduced-motion: reduce) {
            .pw-landing-hero, .pw-hero, .pw-hero-media img, .pw-landing-image img, .pw-page-top, .pw-section-heading, .pw-card, .pw-stat, .pw-image-card, .pw-market-card, .pw-order-card { animation:none !important; transition:none !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def user_dict(user: User) -> dict:
    return {"id": user.id, "name": user.name, "email": user.email, "role": user.role}


def current_user():
    return st.session_state.get("user")


def _initials(name: str) -> str:
    parts = [p for p in str(name or "").split() if p]
    if not parts:
        return "AG"
    return "".join(part[0] for part in parts[:2]).upper()


def logout():
    st.session_state.clear()
    st.rerun()


def auth_page():
    st.markdown('<div class="pw-auth-shell">', unsafe_allow_html=True)
    left, right = st.columns([1.0, 1.0], gap="large")
    with left:
        st.markdown(
            '<div style="padding:1.1rem .6rem 1rem;">'
            '<div class="eyebrow">🌾 AGRIGUARD AI</div>'
            '<h2 style="font-size:2.2rem;margin:.3rem 0;">Smarter farming. Better decisions.</h2>'
            '<div class="muted" style="font-size:1rem;">Sign in to access your private AI price intelligence, marketplace, orders and transaction audit.</div>'
            '</div>', unsafe_allow_html=True
        )
        src = html.escape(_image_src("farmer"), quote=True)
        st.markdown(
            f'''<div class="pw-image-card" style="height:315px;border-radius:24px;">
                <img src="{src}" alt="Indian farmer working in a paddy field" loading="lazy"
                     onerror="this.style.display='none';this.parentElement.classList.add('pw-image-fallback');">
                <div class="pw-image-overlay"><div class="pw-image-label">🌱 Agriculture powered by intelligence</div>
                <div class="pw-image-sub">Farmer decisions · market discovery · trusted transactions</div></div></div>''', unsafe_allow_html=True
        )
        st.markdown(
            '<div class="pw-stat-grid" style="grid-template-columns:repeat(3,minmax(0,1fr));margin-top:.85rem;">'
            '<div class="pw-stat green"><div class="icon">🤖</div><div class="label">AI</div><div class="value" style="font-size:1rem;">Price insight</div></div>'
            '<div class="pw-stat gold"><div class="icon">🤝</div><div class="label">Market</div><div class="value" style="font-size:1rem;">Farmer ↔ Buyer</div></div>'
            '<div class="pw-stat purple"><div class="icon">🔐</div><div class="label">Trust</div><div class="value" style="font-size:1rem;">Audit ledger</div></div>'
            '</div>', unsafe_allow_html=True
        )
    with right:
        mode = st.session_state.get("auth_mode", "login")
        if mode == "login":
            st.markdown('<div class="eyebrow">WELCOME BACK</div><h2 style="margin:.25rem 0 .1rem;font-size:2rem;">Sign in to AgriGuard AI</h2><div class="muted">Continue to your personalized agriculture workspace.</div>', unsafe_allow_html=True)
            with st.form("login"):
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                remember = st.checkbox("Remember me", key="remember_me")
                submitted = st.form_submit_button("🔐 Sign In →", type="primary", width="stretch")
            if submitted:
                if not email.strip() or not password:
                    st.error("Please complete this field.")
                else:
                    session = db_session()
                    try:
                        user = authenticate_user(session, email=email.strip(), password=password)
                        if not user:
                            st.error("Invalid email or password.")
                        else:
                            st.session_state.user = user_dict(user)
                            st.session_state.remember_me = remember
                            st.rerun()
                    except Exception:
                        st.error("Sign in could not be completed. Please try again.")
                    finally:
                        session.close()
            st.markdown('<div style="text-align:center;margin:1rem 0 .7rem;color:#8a958d;">──────── or ────────</div>', unsafe_allow_html=True)
            if st.button("New to AgriGuard?  Create account →", key="switch_register", width="stretch"):
                st.session_state.auth_mode = "register"; st.rerun()
        else:
            st.markdown('<div class="eyebrow">GET STARTED</div><h2 style="margin:.25rem 0 .1rem;font-size:2rem;">Create your account</h2><div class="muted">Choose your role and enter the same account details used by the existing authentication backend.</div>', unsafe_allow_html=True)
            with st.form("register"):
                role = st.radio("I am a", ["🌾 Farmer", "🛒 Buyer"], horizontal=True, key="reg_role")
                name = st.text_input("Full name", key="reg_name")
                email = st.text_input("Email", key="reg_email")
                phone = st.text_input("Phone", key="reg_phone")
                password = st.text_input("Password", type="password", key="reg_pw")
                confirm = st.text_input("Confirm password", type="password", key="reg_confirm")
                submitted = st.form_submit_button("Create Account →", type="primary", width="stretch")
            if submitted:
                if not all([name.strip(), email.strip(), password, confirm]):
                    st.error("Please complete this field.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                else:
                    session = db_session()
                    try:
                        user = register_user(session, name=name.strip(), email=email.strip(), password=password, role="farmer" if "Farmer" in role else "buyer", phone=phone.strip() or None)
                        session.commit(); st.success("Account created successfully. Welcome to AgriGuard AI!"); st.session_state.auth_mode = "login"
                    except Exception as exc:
                        session.rollback(); message = str(exc).lower()
                        st.error("An account with this email already exists." if "exist" in message or "unique" in message else "Account creation could not be completed. Please check your details and try again.")
                    finally: session.close()
            st.markdown('<div style="text-align:center;margin-top:1rem;">Already have an account?</div>', unsafe_allow_html=True)
            if st.button("← Sign In", key="switch_login", width="stretch"):
                st.session_state.auth_mode = "login"; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def sidebar() -> str:
    user = current_user()
    with st.sidebar:
        st.markdown('<div style="display:flex;gap:.65rem;align-items:center;"><div class="pw-brand-mark">🌾</div><div><b style="font-size:1.05rem;">AgriGuard AI</b><div class="muted">Farmer ↔ buyer marketplace</div></div></div>', unsafe_allow_html=True)
        st.divider()
        if not user:
            return "Home"
        st.markdown(f'<div class="pw-sidebar-user"><div class="pw-sidebar-avatar">{html.escape(_initials(user["name"]))}</div><div><b>{html.escape(user["name"])}</b><div class="muted">🟢 {html.escape(user["role"].title())}</div></div></div>', unsafe_allow_html=True)
        st.markdown("<div style='height:.25rem'></div>", unsafe_allow_html=True)
        if user["role"] == "farmer":
            pages = ["🏠 Farmer Dashboard", "🌾 Sell Paddy", "📦 My Listings", "🤝 Buyer Requests", "🚚 My Orders", "💬 Messages", "📊 Market Insights", "🔐 Transaction Audit", "👤 Profile"]
        else:
            pages = ["🏠 Buyer Dashboard", "🔎 Find Paddy", "📋 My Buy Requests", "🚚 My Orders", "💬 Messages", "📊 Market Insights", "🔐 Transaction Audit", "👤 Profile"]
        page = st.radio("Navigation", pages, label_visibility="collapsed", key="main_navigation")
        st.divider()
        if st.button("↪ Log out", width="stretch"):
            logout()
        session = db_session()
        try:
            location_text = _profile_location_text(session, user["id"])
        finally:
            session.close()
        if location_text:
            st.caption(f"📍 {html.escape(location_text)}")
        return page

def home_page():
    # Public landing page only; no private dashboard, marketplace, order or profile data is shown before authentication.
    src = html.escape(_image_src("farmer"), quote=True)
    st.markdown(
        f'''<div class="pw-page-top pw-landing">
        <div class="pw-landing-hero">
            <div class="pw-hero-float pw-float-a">🌾</div><div class="pw-hero-float pw-float-b">🌱</div><div class="pw-hero-float pw-float-c">✦</div>
            <div class="pw-landing-image"><img src="{src}" alt="Indian farmer in a green paddy field" loading="eager" onerror="this.style.display='none';this.parentElement.classList.add('pw-image-fallback');"></div>
            <div class="pw-landing-copy">
                <div class="pw-landing-badge"><span class="pw-ai-pulse"></span> FARMER-FIRST AI PLATFORM</div>
                <div class="pw-kicker" style="color:#f7d56b;margin-top:1rem;">🌾 AGRI GUARD AI</div>
                <div class="pw-location-ribbon">THANJAVUR • TAMIL NADU &nbsp;|&nbsp; CAUVERY DELTA</div>
                <h1>Smarter farming.<br>Better decisions.</h1>
                <p>AI-powered paddy price intelligence and farmer-buyer connectivity, designed around the agriculture of Thanjavur.</p>
                <div class="pw-landing-cta">
                    <a class="pw-cta-primary" href="?auth=login">🌾 Get Started</a>
                    <a class="pw-cta-secondary pw-learn-button" href="#about-ai">Explore AgriGuard AI&nbsp; →</a>
                </div>
                <div class="pw-hero-pill-row"><span>🌾 Paddy Intelligence</span><span>🤖 AI Forecasting</span><span>🤝 Farmer ↔ Buyer</span></div>
            </div>
        </div>
        </div>''', unsafe_allow_html=True
    )

    _section_heading("Proudly rooted in Tamil Nadu", "Thanjavur Paddy Intelligence", "Connecting farmers, markets and AI across the Cauvery Delta.", "📍")
    st.markdown(
        '<div class="pw-tamil-pride">'
        '<div class="pw-pride-copy"><span class="pw-pride-kicker">THANJAVUR • CAUVERY DELTA</span><h2>Paddy agriculture meets modern intelligence.</h2><p>Farmer-first decision support, market intelligence and trusted connectivity — presented around the crop economy of Tamil Nadu.</p>'
        '<div class="pw-pride-tags"><span>🌾 Paddy</span><span>💧 Cauvery</span><span>🤖 AI</span><span>📊 Markets</span><span>🤝 Farmers</span></div></div>'
        '<div class="pw-pride-photo"><img src="' + html.escape(_image_src("field"), quote=True) + '" alt="Tamil Nadu paddy field" onerror="this.style.display=\'none\';this.parentElement.classList.add(\'pw-image-fallback\');"></div>'
        '</div>', unsafe_allow_html=True
    )

    _section_heading("AI agricultural intelligence network", "Markets across the Cauvery Delta", "Supported markets are shown using the application's existing market coordinates. Select a node to explore the market context.", "🗺️")
    dashboard_map(show_prices=False, height=440)

    st.markdown('<div id="about-ai"></div>', unsafe_allow_html=True)
    _section_heading("Meet the AI", "One platform for the crop journey", "Existing forecasting, decision support and marketplace workflows — wrapped in a farmer-first experience.", "🤖")
    cards = [
        ("🌾", "AI Price Intelligence", "Model-based paddy price estimates for the market, variety and grade you analyze.", "green"),
        ("📈", "Future Forecast", "Explore the existing recursive forecast to understand the upcoming price outlook.", "blue"),
        ("🤝", "Farmer ↔ Buyer", "List paddy, discover opportunities and manage the existing marketplace workflow.", "orange"),
        ("🔐", "Trusted Audit", "Verify committed transaction records through the existing blockchain verification flow.", "purple"),
    ]
    cols = st.columns(4, gap="medium")
    for col, (icon, title, body, tone) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="pw-card pw-card-{tone} pw-feature-card"><div style="font-size:2rem;">{icon}</div><h3 style="margin:.35rem 0;">{title}</h3><div class="muted">{body}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="pw-callout pw-demo-disclosure"><b>🔒 Private workspace.</b> Dashboard prices, listings, orders, messages and profile information appear only after authentication. Existing synthetic/demo market disclosures remain unchanged.</div>', unsafe_allow_html=True)


def dashboard_map(show_prices: bool = False, height: int = 430):
    """Render the existing market coordinates over the sourced Thanjavur district geometry."""
    import folium
    try:
        from streamlit_folium import st_folium
    except ImportError:
        st.info("Install streamlit-folium to enable the interactive Thanjavur market map.")
        return

    fmap = folium.Map(
        location=[10.72, 79.15], zoom_start=9, tiles="CartoDB positron",
        control_scale=True, prefer_canvas=True,
    )
    folium.GeoJson(
        THANJAVUR_BOUNDARY,
        name="Thanjavur district glow",
        style_function=lambda _: {"fillColor": "#159447", "fillOpacity": 0.10, "color": "#7b4bd4", "weight": 9, "opacity": 0.22},
    ).add_to(fmap)
    folium.GeoJson(
        THANJAVUR_BOUNDARY,
        name="Thanjavur district",
        style_function=lambda _: {"fillColor": "#dff7e6", "fillOpacity": 0.30, "color": "#0b6b3c", "weight": 3, "opacity": 0.95, "dashArray": "7 6"},
        highlight_function=lambda _: {"weight": 5, "color": "#f4c84a", "fillOpacity": 0.38},
        tooltip=folium.GeoJsonTooltip(fields=["district", "state"], aliases=["District", "State"]),
    ).add_to(fmap)

    market_names = [m for m, (lat, lon) in MARKET_COORDINATES.items() if lat is not None]
    color_cycle = ["#159447", "#f4c84a", "#ee7d32", "#3d7fe0", "#0c9b8b", "#7b4bd4", "#e05a47"]
    supported = load_current_price_combinations() if show_prices else pd.DataFrame()
    for i, name in enumerate(market_names):
        lat, lon = MARKET_COORDINATES[name]
        popup_html = f"<b>{html.escape(name)} Market</b><br><span>Thanjavur • Tamil Nadu</span>"
        if show_prices and not supported.empty:
            subset = supported[supported["market"].astype(str) == name]
            if not subset.empty:
                variety = str(subset.iloc[0]["variety"]); grade = str(subset.iloc[0]["grade"])
                try:
                    forecast = forecast_future_prices(name, variety, grade, date.today(), 7)
                    row = forecast.loc[pd.to_datetime(forecast["date"]).dt.date == date.today()]
                    ai_value = float(row.iloc[0]["predicted_price_rs_per_quintal"]) if not row.empty else None
                    future = forecast.iloc[-1]["predicted_price_rs_per_quintal"] if not forecast.empty else None
                    popup_html += f"<br><b>Variety:</b> {html.escape(variety)}<br><b>Grade:</b> {html.escape(grade)}"
                    popup_html += f"<br><b>AI estimate:</b> {money(ai_value)} / q" if ai_value is not None else "<br><b>AI estimate:</b> unavailable"
                    popup_html += f"<br><b>7-day forecast:</b> {money(future)} / q" if future is not None else ""
                    popup_html += "<br><small>Model estimate; not an official live mandi quote.</small>"
                except Exception:
                    popup_html += "<br><small>AI estimate unavailable for this market selection.</small>"
        popup = folium.Popup(popup_html, max_width=290)
        folium.CircleMarker(
            location=[lat, lon], radius=9, color="#ffffff", weight=2, fill=True,
            fill_color=color_cycle[i % len(color_cycle)], fill_opacity=0.98,
            tooltip=f"📍 {name}", popup=popup,
        ).add_to(fmap)
        folium.CircleMarker(
            location=[lat, lon], radius=15, color=color_cycle[i % len(color_cycle)],
            weight=2, fill=False, opacity=0.35,
        ).add_to(fmap)

    st.markdown('<div class="pw-map-shell"><div class="pw-map-overlay"><span>AI AGRICULTURAL INTELLIGENCE NETWORK</span><b>THANJAVUR • CAUVERY DELTA</b></div>', unsafe_allow_html=True)
    st_folium(fmap, height=height, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def _run_phase6_forecast(market: str, variety: str, grade: str, horizon: int = 7):
    """Run the existing Phase 6 CLI and return its generated forecast table."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.forecasting",
            "--market",
            market,
            "--variety",
            variety,
            "--grade",
            grade,
            "--horizon",
            str(horizon),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "Phase 6 forecast failed.")[-3000:])

    path = forecast_csv_path(market, variety, grade, horizon)
    if not path.exists():
        raise FileNotFoundError(f"Phase 6 completed but forecast file was not found: {path}")

    forecast = pd.read_csv(path)
    if "date" not in forecast.columns or "predicted_price_rs_per_quintal" not in forecast.columns:
        raise ValueError("Phase 6 forecast output is missing the expected date/price columns.")
    forecast["date"] = pd.to_datetime(forecast["date"], errors="coerce")
    forecast = forecast.dropna(subset=["date", "predicted_price_rs_per_quintal"]).sort_values("date")
    if forecast.empty:
        raise ValueError("Phase 6 returned an empty forecast.")
    return forecast, result.stdout


def _show_farmer_market_comparison(
    df: pd.DataFrame,
    market: str,
    variety: str,
    grade: str,
    quantity: float,
    current_price: float,
) -> None:
    """Render the existing Phase 8 market comparison without duplicating its calculations."""
    coords = MARKET_COORDINATES.get(market)
    if not coords or coords[0] is None or coords[1] is None:
        st.warning("Verified coordinates are unavailable for this market, so distance-based market comparison cannot be calculated.")
        return

    try:
        result = compare_markets(
            quantity_quintals=float(quantity),
            current_market=market,
            current_market_price=float(current_price),
            market_price_data=df[
                (df["variety"].astype(str) == variety)
                & (df["grade"].astype(str) == grade)
            ],
            farmer_latitude=float(coords[0]),
            farmer_longitude=float(coords[1]),
            transport_cost_per_km=float(DEFAULT_TRANSPORT_COST_PER_KM),
        )
        table = results_to_dataframe(result)
    except Exception as exc:
        st.error(f"Market comparison failed: {exc}")
        return

    if table.empty:
        st.info("No market-comparison results are available for this selection.")
        return

    display = table.rename(
        columns={
            "market": "Market",
            "current_price": "Latest actual price / q",
            "distance_km": "Distance (km)",
            "transportation_cost": "Estimated transport",
            "gross_revenue": "Gross revenue",
            "net_revenue": "Estimated net revenue",
        }
    ).copy()
    for column in ["Latest actual price / q", "Estimated transport", "Gross revenue", "Estimated net revenue"]:
        if column in display:
            display[column] = display[column].map(money)
    if "Distance (km)" in display:
        display["Distance (km)"] = display["Distance (km)"].map(
            lambda value: "—" if pd.isna(value) else f"{float(value):.1f}"
        )
    st.dataframe(display, width="stretch", hide_index=True)
    st.caption(
        "Market prices are the latest available historical observations for the selected variety/grade. "
        "Transport is an estimate using the existing configurable Phase 8 assumption; distance is straight-line geographic distance."
    )


def _show_buyer_opportunities(variety: str, grade: str, quantity: float) -> None:
    """Show compatible open buyer requests already stored in the live marketplace database."""
    session = db_session()
    try:
        rows = list(
            session.query(BuyerRequest)
            .filter(
                BuyerRequest.status == RequestStatus.PENDING.value,
                BuyerRequest.farmer_listing_id.is_(None),
                BuyerRequest.crop == "Paddy",
                BuyerRequest.variety == variety,
                BuyerRequest.grade == grade,
            )
            .order_by(BuyerRequest.created_at.desc())
            .limit(20)
        )
        buyer_names = {}
        for request in rows:
            buyer = session.get(User, request.buyer_id)
            buyer_names[request.buyer_id] = buyer.name if buyer else "Buyer"
    finally:
        session.close()

    compatible = [
        request
        for request in rows
        if float(request.quantity_quintals) <= float(quantity)
    ]

    if not compatible:
        st.info("No open buyer requests currently match this variety, grade and available quantity.")
        return

    data = [
        {
            "Buyer": buyer_names.get(request.buyer_id, "Buyer"),
            "Quantity requested (q)": float(request.quantity_quintals),
            "Offered price / q": money(request.offered_price),
            "Preferred market": request.preferred_market or "—",
            "Pickup": request.pickup_location or "—",
            "Required by": request.required_by_date or "—",
        }
        for request in compatible
    ]
    st.dataframe(pd.DataFrame(data), width="stretch", hide_index=True)
    st.caption("These are live pending buyer requests stored in MySQL. They are not synthetic demonstration records.")


def _show_ai_market_intelligence(
    selected_market: str,
    variety: str,
    grade: str,
    quantity: float,
    wait_date: date,
    transport_cost_per_km: float,
) -> None:
    """Compare today's AI estimates and the selected future AI date across supported markets."""
    supported = load_current_price_combinations()
    supported = supported[
        (supported["variety"] == str(variety))
        & (supported["grade"] == str(grade))
    ]
    if supported.empty:
        st.info("No supported current-demo market combinations are available for this variety and grade.")
        return

    today = date.today()
    today_prices: dict[str, float] = {}
    wait_prices: dict[str, float] = {}
    rows = []
    for market_name in sorted(supported["market"].unique()):
        try:
            series = forecast_future_prices(market_name, variety, grade, today, 30)
        except Exception:
            continue
        today_row = series.loc[pd.to_datetime(series["date"]).dt.date == today]
        wait_row = series.loc[pd.to_datetime(series["date"]).dt.date == wait_date]
        if today_row.empty or wait_row.empty:
            continue
        today_value = float(today_row.iloc[0]["predicted_price_rs_per_quintal"])
        wait_value = float(wait_row.iloc[0]["predicted_price_rs_per_quintal"])
        today_prices[market_name] = today_value
        wait_prices[market_name] = wait_value
        rows.append({
            "market": market_name,
            "today": today_value,
            "wait": wait_value,
        })

    if selected_market not in today_prices:
        st.info("Price Intelligence is unavailable because the selected market does not have a complete current synthetic forecast for this variety/grade.")
        return

    ai_prices = pd.DataFrame([
        {
            "market": row["market"],
            "variety": variety,
            "grade": grade,
            "date": today,
            "modal_price_rs_per_quintal": row["today"],
        }
        for row in rows
    ])
    selected_coords = MARKET_COORDINATES.get(selected_market)
    if not selected_coords or selected_coords[0] is None or selected_coords[1] is None:
        st.warning("Verified coordinates are unavailable for the selected market, so distance-based AI market comparison cannot be calculated.")
        return

    selected_today = today_prices[selected_market]
    try:
        comparison = compare_markets(
            quantity_quintals=quantity,
            current_market=selected_market,
            current_market_price=selected_today,
            market_price_data=ai_prices,
            farmer_latitude=float(selected_coords[0]),
            farmer_longitude=float(selected_coords[1]),
            transport_cost_per_km=transport_cost_per_km,
            forecast_prices=wait_prices,
        )
        table = results_to_dataframe(comparison)
    except Exception as exc:
        st.warning("⚠️ Forecast temporarily unavailable")
        st.caption("Please try again.")
        with st.expander("Technical details"):
            st.code(str(exc))
        return

    if table.empty:
        st.info("No AI market-comparison results are available for this selection.")
        return

    display = table.rename(columns={
        "market": "Market",
        "current_price": "Today's AI estimate / q",
        "forecast_price": f"AI estimate {wait_date.strftime('%d %b')} / q",
        "distance_km": "Distance (km)",
        "transportation_cost": "Transportation cost",
        "net_revenue": "Estimated net revenue",
        "net_revenue_difference": "Net revenue difference",
    }).copy()
    for column in ["Today's AI estimate / q", f"AI estimate {wait_date.strftime('%d %b')} / q", "Transportation cost", "Estimated net revenue", "Net revenue difference"]:
        if column in display:
            display[column] = display[column].map(lambda value: "—" if pd.isna(value) else money(value))
    if "Distance (km)" in display:
        display["Distance (km)"] = display["Distance (km)"].map(lambda value: "—" if pd.isna(value) else f"{float(value):.1f}")
    st.dataframe(display, width="stretch", hide_index=True)
    st.caption(
        "Today's prices in this table are model-based AI estimates for the application date, not historical prices or live official mandi prices. "
        "Transportation uses the existing Phase 8 straight-line-distance and cost-per-km calculation."
    )

    alternatives = table[
        (table["market"] != selected_market) & table["net_revenue_difference"].notna()
    ].sort_values("net_revenue_difference", ascending=False)
    if not alternatives.empty:
        alternative = alternatives.iloc[0]
        st.info(
            f"Selected market: {selected_market} at {money(selected_today)} / quintal today. "
            f"Alternative market: {alternative['market']} at {money(alternative['current_price'])} / quintal today. "
            f"Estimated price difference: {money(float(alternative['current_price']) - selected_today)} / quintal. "
            f"Estimated net revenue difference after transportation: {money(alternative['net_revenue_difference'])}."
        )

def _greeting_for(name: str) -> str:
    hour = datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
    return f"{greeting}, {name} 👋"


def _profile_location_text(session, user_id: int) -> str:
    loc = session.query(UserLocation).filter(UserLocation.user_id == user_id).first()
    if not loc:
        return ""
    parts = [loc.village, loc.taluk, loc.district]
    values = [str(x).strip() for x in parts if x and str(x).strip()]
    return ", ".join(values)


def _dashboard_greeting(user: dict, session, role_label: str) -> None:
    location = _profile_location_text(session, user["id"])
    subtitle = role_label
    if location:
        subtitle += f" • {location}"
    st.markdown(
        f'''<div class="pw-greeting"><div class="pw-kicker">🌱 Your workspace</div>
        <div class="pw-greeting-title">{html.escape(_greeting_for(user["name"]))}</div>
        <div class="pw-greeting-sub">{html.escape(subtitle)}</div></div>''', unsafe_allow_html=True
    )


def farmer_dashboard():
    user = current_user(); session = db_session()
    _dashboard_greeting(user, session, "Farmer")
    try:
        listings = list(session.query(DBFarmerListing).filter(DBFarmerListing.farmer_id == user["id"]).order_by(DBFarmerListing.created_at.desc()).limit(10))
        requests = list(session.query(BuyerRequest).options(joinedload(BuyerRequest.buyer)).join(DBFarmerListing, BuyerRequest.farmer_listing_id == DBFarmerListing.id).filter(DBFarmerListing.farmer_id == user["id"], BuyerRequest.status == RequestStatus.PENDING.value))
        orders = order_history(session, user["id"])
    finally: session.close()

    context = st.session_state.get("farmer_ai_context", {})
    hero_price = _hero_ai_price(context.get("market"), context.get("variety"), context.get("grade"))
    src = html.escape(_image_src("farmer"), quote=True)
    price_text = money(hero_price) if hero_price is not None else "Ready"
    st.markdown(
        f'''<div class="pw-hero pw-dashboard-hero"><div class="pw-hero-float pw-float-a">🌾</div><div class="pw-hero-float pw-float-b">✦</div><div class="pw-hero-inner"><div class="pw-hero-copy"><div class="pw-kicker" style="color:#f9df8d;">🌾 AgriGuard AI</div><h1>Sell smarter.<br>Earn better.</h1><p>AI-powered paddy price intelligence and farmer-buyer connectivity.</p><div style="margin-top:.8rem;"><span class="pw-chip">🌱 Farmer first</span><span class="pw-chip">🤖 AI price</span><span class="pw-chip">🚚 Marketplace</span></div></div><div class="pw-hero-media"><img src="{src}" alt="Indian farmer and agriculture" onerror="this.style.display='none';"><div class="pw-floating-price"><div class="label">🤖 AI PRICE</div><div class="value">{price_text}</div><div class="sub">per quintal · existing AI engine</div></div></div></div></div>''', unsafe_allow_html=True)
    h1,h2=st.columns(2,gap="small")
    if h1.button("🌾 Analyze Paddy",type="primary",key="hero_analyze_paddy",width="stretch"):
        st.session_state.farmer_ai_analyzed=True
    if h2.button("📊 Explore Markets",key="hero_market_insights",width="stretch"):
        st.session_state.nav_override="Market Insights"; st.rerun()

    df = load_prices()
    historical_markets=set(df.market.dropna().astype(str).unique()); historical_varieties=set(df.variety.dropna().astype(str).unique()); historical_grades=set(df.grade.dropna().astype(str).unique())
    current_combinations=load_current_price_combinations(); current_combinations=current_combinations[current_combinations["market"].isin(historical_markets)&current_combinations["variety"].isin(historical_varieties)&current_combinations["grade"].isin(historical_grades)]
    markets=sorted(current_combinations["market"].unique())
    if not markets: st.error("Today's AI estimate is unavailable because the current demo dataset has no supported combinations."); return
    default_market=context.get("market") if context.get("market") in markets else markets[0]
    market_combinations=current_combinations[current_combinations["market"]==default_market]; default_varieties=sorted(market_combinations["variety"].unique()); default_variety=context.get("variety") if context.get("variety") in default_varieties else default_varieties[0]
    default_grades=sorted(market_combinations[market_combinations["variety"]==default_variety]["grade"].unique()); default_grade=context.get("grade") if context.get("grade") in default_grades else default_grades[0]

    _section_heading("Analyze your harvest","Your AI harvest cockpit","Choose the market, variety, grade and selling conditions before running the existing decision engine.","🌾")
    left,right=st.columns([1.55,.85],gap="large")
    with left:
        st.markdown('<div class="pw-form-shell">',unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        market=c1.selectbox("Market",markets,index=markets.index(default_market),key="farmer_ai_market")
        market_combinations=current_combinations[current_combinations["market"]==market]; varieties=sorted(market_combinations["variety"].unique()); variety_context=context.get("variety") if context.get("variety") in varieties else varieties[0]
        variety=c2.selectbox("Paddy variety",varieties,index=varieties.index(variety_context),key="farmer_ai_variety")
        grades=sorted(market_combinations[market_combinations["variety"]==variety]["grade"].unique()); grade_context=context.get("grade") if context.get("grade") in grades else grades[0]
        grade=c3.selectbox("Grade",grades,index=grades.index(grade_context),key="farmer_ai_grade")
        c1,c2=st.columns(2); quantity=c1.number_input("Quantity (quintals)",min_value=0.1,value=float(context.get("quantity",100.0)),step=1.0,key="farmer_ai_quantity"); waiting_days=c2.number_input("Expected waiting days",min_value=1,max_value=30,value=min(max(int(context.get("waiting_days",7)),1),30),step=1,key="farmer_ai_waiting_days")
        c1,c2=st.columns(2); storage_per_day=c1.number_input("Storage cost (₹/quintal/day)",min_value=0.0,value=float(context.get("storage_per_day",0.0)),step=1.0,key="farmer_ai_storage"); transport_cost=c2.number_input("Transport cost for Sell vs Wait (₹)",min_value=0.0,value=float(context.get("transport_cost",0.0)),step=100.0,key="farmer_ai_transport")
        if st.button("✨ ANALYZE PADDY",type="primary",width="stretch"):
            st.session_state.farmer_ai_context={"market":market,"variety":variety,"grade":grade,"quantity":float(quantity),"waiting_days":int(waiting_days),"storage_per_day":float(storage_per_day),"transport_cost":float(transport_cost)}; st.session_state.farmer_ai_analyzed=True; st.session_state.farmer_ai_forecast=None; st.rerun()
        st.markdown('</div>',unsafe_allow_html=True)
    with right:
        _image_card("field","Paddy field intelligence","A visual home for your harvest analysis")

    if not st.session_state.get("farmer_ai_analyzed"):
        st.markdown('<div class="pw-callout"><b>✨ Ready when you are.</b><br><span class="muted">Select your paddy details and click Analyze Paddy to generate the model-based daily future price forecast.</span></div>',unsafe_allow_html=True)
    else:
        current=latest_price(df,market,variety,grade); series_date=latest_actual_date(df,market,variety,grade); overall_date=latest_actual_date(df)
        try:
            future_forecast=forecast_future_prices(market=market,variety=variety,grade=grade,start_date=date.today(),horizon_days=30); st.session_state.farmer_ai_forecast=future_forecast
            today_row=future_forecast.loc[pd.to_datetime(future_forecast["date"]).dt.date==date.today()]
            if today_row.empty: raise ValueError("The recursive forecast did not return the application date.")
            today_prediction=float(today_row.iloc[0]["predicted_price_rs_per_quintal"])
        except Exception as exc:
            st.warning("⚠️ Forecast temporarily unavailable"); st.caption("Please try again.")
            with st.expander("Technical details"): st.code(str(exc))
            return

        _section_heading("AI price result","Your AI price","Existing forecast engine output for the selected market, variety and grade.","🤖")
        result_left,result_right=st.columns([1.15,.85],gap="large")
        with result_left:
            st.markdown(f'''<div class="pw-result"><div class="pw-kicker">🌾 YOUR AI PRICE</div><div class="big">{money(today_prediction)}</div><div style="font-weight:800;color:#5f6e62;">per quintal</div><div style="margin-top:.7rem;"><span class="pw-chip">🤖 AgriGuard AI estimate</span><span class="pw-chip">📅 {date.today().strftime('%d %b %Y')}</span></div></div>''',unsafe_allow_html=True)
            a,b,c=st.columns(3); a.metric("Market",market); b.metric("Variety",variety); c.metric("Grade",grade)
        with result_right:
            _image_card("grain","Rice grain","Your paddy, turned into decision-ready insight")
        st.markdown('<div class="pw-note">AI estimate is produced by the existing XGBoost/recursive forecast pipeline and is not an official live mandi quote.</div>',unsafe_allow_html=True)

        _section_heading("Forecast","See the next 30 days","Actual forecast values from the existing recursive forecasting layer.","📈")
        fig=go.Figure(go.Scatter(x=future_forecast["date"],y=future_forecast["predicted_price_rs_per_quintal"],mode="lines+markers",name="AI estimate",line=dict(width=3),marker=dict(size=5))); fig.update_layout(height=340,margin=dict(l=10,r=10,t=20,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.8)",yaxis_title="₹ / quintal",xaxis_title=None); fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor="#e8eef0"); st.plotly_chart(fig,width="stretch",config={"displayModeBar":False})
        forecast_cards=future_forecast.head(5); cols=st.columns(5,gap="small")
        for col,(_,row) in zip(cols,forecast_cards.iterrows()):
            with col: st.markdown(f'<div class="pw-card pw-card-blue"><div class="muted">{pd.Timestamp(row["date"]).strftime("%d %b")}</div><div style="font-size:1.25rem;font-weight:900;color:#2367ad;">{money(row["predicted_price_rs_per_quintal"])}</div><div class="muted">AI / quintal</div></div>',unsafe_allow_html=True)

        if current is None or series_date is None: st.error("No actual historical price is available for this market, variety and grade selection."); return
        forecast=st.session_state.get("farmer_ai_forecast"); wait_index=int(waiting_days)-1; wait_row=forecast.iloc[wait_index]; decision_forecast=float(wait_row["predicted_price_rs_per_quintal"]); wait_date=pd.Timestamp(wait_row["date"]).date()
        _section_heading("Sell vs Wait","Choose with the existing decision engine","Compare today's AI estimate against the selected waiting period.","🟢")
        try:
            decision=calculate_decision(DecisionInputs(quantity_quintals=float(quantity),current_market=market,current_modal_price=float(today_prediction),forecast_price=decision_forecast,expected_waiting_days=int(waiting_days),transportation_cost=float(transport_cost),storage_cost_per_day=float(storage_per_day)))
            x,y=st.columns(2,gap="small")
            with x: st.markdown(f'<div class="pw-card pw-card-green"><div style="font-size:1.5rem;">🟢</div><div class="eyebrow">SELL TODAY</div><div class="pw-price">{money(decision.sell_today.net_revenue)}</div><div class="muted">Net revenue · {money(today_prediction)} / quintal</div></div>',unsafe_allow_html=True)
            with y: st.markdown(f'<div class="pw-card pw-card-orange"><div style="font-size:1.5rem;">🟠</div><div class="eyebrow">WAIT {waiting_days} DAYS</div><div class="pw-price">{money(decision.wait.net_revenue) if decision.wait else "Unavailable"}</div><div class="muted">{wait_date.strftime("%d %b %Y")} · {money(decision_forecast)} / quintal</div></div>',unsafe_allow_html=True)
            a,b=st.columns(2); a.metric("Net revenue difference",money(decision.net_revenue_difference)); b.metric("Break-even future price",money(decision.break_even_future_price)+" / quintal")
            st.markdown(f'<div class="pw-callout"><b>Decision engine interpretation</b><br>{html.escape(str(decision.interpretation))}</div>',unsafe_allow_html=True)
        except Exception as exc:
            st.warning("Sell vs Wait is temporarily unavailable.");
            with st.expander("Technical details"): st.code(str(exc))

        _section_heading("Market intelligence","Compare supported markets","Use the existing Phase 8 market comparison and AI market intelligence.","🏪")
        _show_ai_market_intelligence(selected_market=market,variety=variety,grade=grade,quantity=float(quantity),wait_date=wait_date,transport_cost_per_km=float(DEFAULT_TRANSPORT_COST_PER_KM))
        _section_heading("Marketplace","Buyer opportunities","Compatible pending buyer requests from the live marketplace database.","🤝")
        _show_buyer_opportunities(variety,grade,float(quantity))
        a,b=st.columns(2)
        if a.button("🤝 View Buyer Requests",key="farmer_ai_view_requests",width="stretch"): st.session_state.nav_override="Buyer Requests"; st.rerun()
        if b.button("🌾 Create Listing",key="farmer_ai_create_listing",type="primary",width="stretch"): st.session_state.nav_override="Sell Paddy"; st.rerun()

    st.markdown(f'''<div class="pw-stat-grid"><div class="pw-stat green"><div class="icon">📦</div><div class="label">Active listings</div><div class="value">{sum(x.status == ListingStatus.ACTIVE.value for x in listings)}</div></div><div class="pw-stat gold"><div class="icon">🤝</div><div class="label">Pending requests</div><div class="value">{len(requests)}</div></div><div class="pw-stat blue"><div class="icon">🚚</div><div class="label">Orders</div><div class="value">{len(orders)}</div></div><div class="pw-stat teal"><div class="icon">📅</div><div class="label">Application date</div><div class="value">{date.today().strftime("%d %b")}</div></div></div>''',unsafe_allow_html=True)
    _section_heading("Market map","Nearby Thanjavur markets","Existing interactive map and coordinates are unchanged.","🗺️"); dashboard_map()
    _section_heading("Recent listings","Your marketplace activity","Existing listing records from MySQL.","📦")
    if listings:
        st.dataframe(pd.DataFrame([{"Market":x.market,"Variety":x.variety,"Grade":x.grade,"Available (q)":float(x.quantity_quintals),"Minimum price":money(x.minimum_price),"Status":x.status} for x in listings]),width="stretch",hide_index=True)
    else: st.info("You have no listings yet.")

def sell_paddy():
    _section_heading("Marketplace listing","Create your paddy listing","Publish through the existing MySQL marketplace service.","🌾")
    left,right=st.columns([.9,1.1],gap="large")
    with left:
        st.markdown('<div class="pw-context-ribbon"><span>🌾 HARVEST TO MARKET</span><b>List your crop with confidence</b><small>Real agricultural imagery • existing marketplace workflow</small></div>',unsafe_allow_html=True)
        _image_card("paddy","Harvested Paddy","Fresh crop ready for the market")
        st.markdown('<div style="height:.6rem"></div>',unsafe_allow_html=True)
        _image_card("farmer","Farmer Selling","Farmer-first marketplace presentation")
    with right:
        st.markdown('<div class="pw-form-shell">',unsafe_allow_html=True)
        df=load_prices(); markets=sorted(df.market.dropna().astype(str).unique()); varieties=sorted(df.variety.dropna().astype(str).unique()); grades=sorted(df.grade.dropna().astype(str).unique())
        with st.form("listing"):
            c1,c2=st.columns(2); market=c1.selectbox("Preferred market",markets); variety=c2.selectbox("Variety",varieties); grade=c1.selectbox("Grade",grades)
            qty=c2.number_input("Quantity (quintals)",min_value=0.1,value=100.0,step=1.0); minimum=c1.number_input("Minimum expected price (₹/quintal)",min_value=0.0,value=float(latest_price(df,market,variety,grade) or 0),step=10.0); available=c2.date_input("Available from",value=date.today()); submitted=st.form_submit_button("🌾 Publish Listing",type="primary",width="stretch")
        st.markdown('</div>',unsafe_allow_html=True)
    if submitted:
        coords=MARKET_COORDINATES.get(market,(None,None)); session=db_session()
        try:
            create_farmer_listing(session,farmer_id=current_user()["id"],crop="Paddy",variety=variety,grade=grade,quantity_quintals=Decimal(str(qty)),minimum_price=Decimal(str(minimum)),market=market,latitude=Decimal(str(coords[0])) if coords[0] is not None else None,longitude=Decimal(str(coords[1])) if coords[1] is not None else None,available_from=available); session.commit(); st.success("Your paddy listing is live for buyers.")
        except Exception as exc: session.rollback(); st.error(str(exc))
        finally: session.close()

def my_listings():
    session=db_session()
    try: rows=list(session.query(DBFarmerListing).filter(DBFarmerListing.farmer_id==current_user()["id"]).order_by(DBFarmerListing.created_at.desc()))
    finally: session.close()
    _section_heading("Marketplace","My Listings","Your existing farmer listings, presented as visual inventory cards.","📦")
    if not rows: st.info("No listings yet."); return
    cols=st.columns(3,gap="small")
    for i,r in enumerate(rows):
        with cols[i%3]:
            st.markdown(f'''<div class="pw-market-card"><div style="height:125px;border-radius:14px;overflow:hidden;margin-bottom:.75rem;"><img src="{html.escape(_image_src('paddy'),quote=True)}" style="width:100%;height:100%;object-fit:cover;" alt="Paddy listing"></div><span class="pw-chip">{html.escape(str(r.status))}</span><h4>🌾 {html.escape(str(r.variety))}</h4><div class="muted">{html.escape(str(r.grade))} · 📍 {html.escape(str(r.market))}</div><div style="display:flex;justify-content:space-between;margin-top:.55rem;"><b>{float(r.quantity_quintals):,.0f} q</b><b style="color:#a76800;">{money(r.minimum_price)} / q</b></div><div class="muted">Available: {html.escape(str(r.available_from))}</div></div>''',unsafe_allow_html=True)

def farmer_requests():
    session=db_session()
    try: rows=list(session.query(BuyerRequest).options(joinedload(BuyerRequest.buyer)).join(DBFarmerListing,BuyerRequest.farmer_listing_id==DBFarmerListing.id).filter(DBFarmerListing.farmer_id==current_user()["id"]).order_by(BuyerRequest.created_at.desc()))
    finally: session.close()
    _section_heading("Marketplace","Buyer Requests","Review compatible purchase requests from buyers.","🤝")
    if not rows: st.info("No buyer purchase requests yet."); return
    for r in rows:
        buyer_name=r.buyer.name if r.buyer else "Unknown buyer"
        with st.container(border=True):
            st.markdown(f'<div class="pw-card pw-card-gold"><div style="display:flex;justify-content:space-between;gap:1rem;align-items:center;"><div><div class="eyebrow">BUYER PURCHASE REQUEST</div><h3 style="margin:.2rem 0;">🤝 {html.escape(buyer_name)}</h3></div><span class="pw-chip">{html.escape(str(r.status))}</span></div><div class="pw-stat-grid" style="grid-template-columns:repeat(3,1fr);"><div class="pw-stat green"><div class="label">Quantity</div><div class="value">{float(r.quantity_quintals):,.0f} q</div></div><div class="pw-stat gold"><div class="label">Offer</div><div class="value">{money(r.offered_price)}</div></div><div class="pw-stat blue"><div class="label">Required by</div><div class="value" style="font-size:1.15rem;">{html.escape(str(r.required_by_date or "—"))}</div></div></div><div class="muted">📍 Pickup: {html.escape(str(r.pickup_location or "Not specified"))}</div></div>',unsafe_allow_html=True)
            if r.status==RequestStatus.PENDING.value:
                a,b=st.columns(2)
                if a.button("✅ Accept",key=f"accept_{r.id}",type="primary",width="stretch"):
                    s=db_session()
                    try: accept_request(s,request_id=r.id,farmer_id=current_user()["id"]); s.commit(); st.success("Request accepted and order created."); st.rerun()
                    except Exception as exc: s.rollback(); st.error(str(exc))
                    finally:s.close()
                if b.button("Reject",key=f"reject_{r.id}",width="stretch"):
                    s=db_session()
                    try: reject_request(s,request_id=r.id,farmer_id=current_user()["id"]); s.commit(); st.rerun()
                    except Exception as exc:s.rollback();st.error(str(exc))
                    finally:s.close()

def buyer_dashboard():
    user=current_user(); session=db_session()
    _dashboard_greeting(user, session, "Buyer")
    try: requests=list(session.query(BuyerRequest).filter(BuyerRequest.buyer_id==user["id"]).order_by(BuyerRequest.created_at.desc()).limit(10)); orders=order_history(session,user["id"])
    finally: session.close()
    df=load_prices(); latest=df.dropna(subset=["date","modal_price_rs_per_quintal"]).sort_values("date").iloc[-1]
    src=html.escape(_image_src("warehouse"),quote=True)
    st.markdown(f'''<div class="pw-hero pw-dashboard-hero"><div class="pw-hero-float pw-float-a">🌾</div><div class="pw-hero-float pw-float-b">✦</div><div class="pw-hero-inner"><div class="pw-hero-copy"><div class="pw-kicker" style="color:#f9df8d;">🏪 AgriGuard Buyer</div><h1>Source smarter.<br>Move faster.</h1><p>Find compatible paddy listings, create buy requests and track agricultural orders in one place.</p><div style="margin-top:.8rem;"><span class="pw-chip">🔎 Find Paddy</span><span class="pw-chip">📦 Live Listings</span><span class="pw-chip">🚚 Orders</span></div></div><div class="pw-hero-media"><img src="{src}" alt="Grain storage and market logistics" onerror="this.style.display='none';"></div></div></div>''',unsafe_allow_html=True)
    a,b=st.columns(2)
    if a.button("🔎 Find Paddy",type="primary",width="stretch"): st.session_state.nav_override="Find Paddy"; st.rerun()
    if b.button("📋 My Buy Requests",width="stretch"): st.session_state.nav_override="My Buy Requests"; st.rerun()
    st.markdown(f'''<div class="pw-stat-grid"><div class="pw-stat gold"><div class="icon">🌾</div><div class="label">Latest market price</div><div class="value">{money(latest.modal_price_rs_per_quintal)}</div><div class="muted">per quintal</div></div><div class="pw-stat green"><div class="icon">📋</div><div class="label">Purchase requests</div><div class="value">{len(requests)}</div></div><div class="pw-stat blue"><div class="icon">🚚</div><div class="label">Active orders</div><div class="value">{len(orders)}</div></div><div class="pw-stat teal"><div class="icon">🌾</div><div class="label">Marketplace</div><div class="value">Paddy</div></div></div>''',unsafe_allow_html=True)
    _section_heading("Buyer journey","Explore the supply side","Visual context for procurement, storage and logistics.","🌾"); _image_strip([("grain","Rice Grain"),("warehouse","Warehouse"),("truck","Transport"),("paddy","Paddy Crop")])
    _section_heading("Nearby markets","Source around Thanjavur","Existing market map and coordinates are unchanged.","🗺️"); dashboard_map()

def find_paddy():
    _section_heading("Live marketplace","Find Paddy","Create a buy request using the existing MySQL marketplace service, then review matching farmer listings.","🔎")
    left,right=st.columns([.95,1.05],gap="large")
    with left:
        _image_card("grain","Rice Bags & Grain","Visual procurement context")
        _image_card("warehouse","Wholesale Storage","Warehouse and market supply")
    with right:
        st.markdown('<div class="pw-form-shell">',unsafe_allow_html=True)
        df=load_prices(); varieties=sorted(df.variety.dropna().astype(str).unique()); grades=sorted(df.grade.dropna().astype(str).unique()); markets=sorted(df.market.dropna().astype(str).unique())
        with st.form("buy_request"):
            c1,c2=st.columns(2); variety=c1.selectbox("Variety",varieties); grade=c2.selectbox("Grade",grades); qty=c1.number_input("How much do you want to buy? (quintals)",min_value=0.1,value=40.0,step=1.0); offer=c2.number_input("Offered price (₹/quintal)",min_value=0.0,value=float(latest_price(df,markets[0],variety,grade) or 0),step=10.0); market=c1.selectbox("Preferred market",markets); pickup=c2.text_input("Pickup location"); required=st.date_input("Required-by date",value=date.today()); submitted=st.form_submit_button("🔎 Find Matching Farmer Listings",type="primary",width="stretch")
        st.markdown('</div>',unsafe_allow_html=True)
    if submitted:
        session=db_session()
        try:
            request=create_buyer_request(session,buyer_id=current_user()["id"],crop="Paddy",variety=variety,grade=grade,quantity_quintals=Decimal(str(qty)),offered_price=Decimal(str(offer)),preferred_market=market,pickup_location=pickup or None,latitude=current_user_location(session,current_user()["id"])[0],longitude=current_user_location(session,current_user()["id"])[1],required_by_date=required); session.commit(); st.session_state.live_request_id=request.id
        except Exception as exc:session.rollback();st.error(str(exc))
        finally:session.close()
    rid=st.session_state.get("live_request_id")
    if not rid:return
    session=db_session()
    try: request=session.get(BuyerRequest,rid); candidates=match_buyer_to_listings(session,request)
    finally:session.close()
    _section_heading("Matches","Available farmer listings","Results are live marketplace records returned by the existing matching service.","🤝")
    if not candidates: st.info("No live farmer listings matched these crop details."); return
    cols=st.columns(2,gap="small")
    for i,c in enumerate(candidates[:20]):
        l=c.listing
        with cols[i%2]:
            st.markdown(f'''<div class="pw-market-card"><div style="height:160px;border-radius:15px;overflow:hidden;margin-bottom:.75rem;"><img src="{html.escape(_image_src('paddy'),quote=True)}" style="width:100%;height:100%;object-fit:cover;" alt="Paddy marketplace listing"></div><span class="pw-chip">🌾 {html.escape(str(l.variety))}</span><h3 style="margin:.35rem 0;">{html.escape(str(l.grade))}</h3><div class="muted">👤 Farmer #{l.farmer_id} · 📍 {html.escape(str(l.market))}</div><div style="display:flex;justify-content:space-between;margin:.55rem 0;"><b>⚖️ {float(l.quantity_quintals):,.0f} q</b><b style="color:#a76800;">{money(l.minimum_price)} / q</b></div><div class="muted">Offer: {money(request.offered_price)} / q · Distance: {f'{c.distance_km:.1f} km' if c.distance_km is not None else '—'}</div></div>''',unsafe_allow_html=True)
            if not c.price_compatible: st.caption("⚠️ Offer is below this farmer's minimum expected price.")
            max_qty=min(float(l.quantity_quintals),float(request.quantity_quintals)); requested=st.number_input("Quantity to request",min_value=0.1,max_value=max_qty,value=max_qty,step=1.0,key=f"qty_{l.id}_{rid}")
            if st.button("Request Paddy",key=f"purchase_{l.id}_{rid}",type="primary",width="stretch"):
                s=db_session()
                try: request_purchase(s,buyer_id=current_user()["id"],listing_id=l.id,quantity_quintals=Decimal(str(requested)),offered_price=Decimal(request.offered_price),pickup_location=request.pickup_location,required_by_date=request.required_by_date); s.commit(); st.success("Purchase request sent to the farmer."); st.rerun()
                except Exception as exc:s.rollback();st.error(str(exc))
                finally:s.close()

def current_user_location(session,user_id):
    loc=session.query(UserLocation).filter(UserLocation.user_id==user_id).first()
    return (loc.latitude,loc.longitude) if loc else (None,None)


def my_buy_requests():
    session=db_session()
    try: rows=list(session.query(BuyerRequest).filter(BuyerRequest.buyer_id==current_user()["id"]).order_by(BuyerRequest.created_at.desc()))
    finally: session.close()
    _section_heading("Buyer workspace","My Buy Requests","Track the requests created through the existing marketplace service.","📋")
    if not rows: st.info("No requests yet."); return
    for r in rows:
        with st.container(border=True):
            st.markdown(f'<div class="pw-card pw-card-gold"><div class="eyebrow">BUY REQUEST #{r.id}</div><h3>🌾 {html.escape(str(r.variety))} · {html.escape(str(r.grade))}</h3><div class="muted">📍 {html.escape(str(r.preferred_market or "Market not specified"))}</div><div class="pw-stat-grid" style="grid-template-columns:repeat(4,1fr);"><div class="pw-stat green"><div class="label">Quantity</div><div class="value">{float(r.quantity_quintals):,.0f} q</div></div><div class="pw-stat gold"><div class="label">Offer</div><div class="value">{money(r.offered_price)}</div></div><div class="pw-stat blue"><div class="label">Status</div><div class="value" style="font-size:1rem;">{html.escape(str(r.status))}</div></div><div class="pw-stat orange"><div class="label">Required by</div><div class="value" style="font-size:1rem;">{html.escape(str(r.required_by_date or "—"))}</div></div></div></div>',unsafe_allow_html=True)

def order_tracking(order):
    statuses=[(OrderStatus.ORDER_PLACED.value,"Order placed"),(OrderStatus.SELLER_CONFIRMED.value,"Seller confirmed"),(OrderStatus.PICKUP_SCHEDULED.value,"Pickup expected"),(OrderStatus.IN_TRANSIT.value,"In transit"),(OrderStatus.DELIVERED.value,"Delivered")]
    current=statuses.index(next(x for x in statuses if x[0]==order.status)) if order.status in [x[0] for x in statuses] else -1
    st.markdown(f'<div class="pw-callout"><b>🚚 Order timeline</b><div class="muted">Order #{order.order_date:%Y%m%d}-{order.id:03d}</div></div>',unsafe_allow_html=True)
    cols=st.columns(len(statuses),gap="small")
    for i,(code,label) in enumerate(statuses):
        mark="✓" if i<=current else "○"
        extra=""
        if code==OrderStatus.PICKUP_SCHEDULED.value: extra=f" · {order.estimated_pickup_date:%d %b %Y}"
        if code==OrderStatus.IN_TRANSIT.value: extra=f" · {order.estimated_delivery_date:%d %b %Y}"
        with cols[i]:
            tone="green" if i<=current else "blue"
            st.markdown(f'<div class="pw-stat {tone}" style="min-height:95px;"><div class="icon">{"🟢" if i<=current else "⚪"}</div><div class="label">{mark} {html.escape(label)}</div><div class="muted">{html.escape(extra.strip(" ·")) if extra else "Pending"}</div></div>',unsafe_allow_html=True)
    st.caption("Pickup and delivery dates are deterministic estimates, not guarantees.")

def my_orders():
    session=db_session()
    try: orders=order_history(session,current_user()["id"])
    finally: session.close()
    _section_heading("Logistics","My Orders","Track the existing order state machine without changing its backend behavior.","🚚")
    _image_strip([("truck","Agricultural Transport"),("warehouse","Warehouse"),("paddy","Paddy Cargo")])
    if not orders: st.info("No orders yet."); return
    for order in orders:
        with st.container(border=True):
            st.markdown(f'<div class="pw-order-card"><div style="display:flex;justify-content:space-between;gap:1rem;align-items:center;"><div><div class="eyebrow">🚚 ORDER #{order.order_date:%Y%m%d}-{order.id:03d}</div><h3 style="margin:.25rem 0;">Paddy movement</h3></div><span class="pw-chip">{html.escape(str(order.status))}</span></div><div class="pw-stat-grid" style="grid-template-columns:repeat(3,1fr);"><div class="pw-stat blue"><div class="label">Quantity</div><div class="value">{float(order.quantity_quintals):,.0f} q</div></div><div class="pw-stat teal"><div class="label">Agreed price</div><div class="value">{money(order.agreed_price)}</div></div><div class="pw-stat gold"><div class="label">Total</div><div class="value">{money(order.total_amount)}</div></div></div></div>',unsafe_allow_html=True)
            order_tracking(order)
            allowed={OrderStatus.SELLER_CONFIRMED.value:OrderStatus.PICKUP_SCHEDULED.value,OrderStatus.PICKUP_SCHEDULED.value:OrderStatus.IN_TRANSIT.value,OrderStatus.IN_TRANSIT.value:OrderStatus.DELIVERED.value}
            if order.status in allowed and st.button(f"Advance to {allowed[order.status].replace('_',' ').title()}",key=f"advance_{order.id}",type="primary"):
                s=db_session()
                try:advance_order(s,order_id=order.id,user_id=current_user()["id"],next_status=allowed[order.status]);s.commit();st.rerun()
                except Exception as exc:s.rollback();st.error(str(exc))
                finally:s.close()

def messages_page():
    session=db_session()
    try: orders=order_history(session,current_user()["id"])
    finally: session.close()
    _section_heading("Communication","Messages","Keep buyer–farmer conversations connected to existing orders.","💬")
    _image_strip([("farmer","Farmer"),("warehouse","Marketplace"),("truck","Delivery")])
    if not orders:
        st.markdown('<div class="pw-callout"><b>💬 No order conversations yet.</b><br><span class="muted">Messages become available after an order is created.</span></div>',unsafe_allow_html=True)
        return
    for order in orders:
        with st.container(border=True):
            st.markdown(f'<div class="pw-order-card"><div class="eyebrow">💬 ORDER #{order.order_date:%Y%m%d}-{order.id:03d}</div><h3 style="margin:.2rem 0;">Conversation</h3><div class="muted">Quantity: {float(order.quantity_quintals):,.0f} q · Status: {html.escape(str(order.status))}</div></div>',unsafe_allow_html=True)
            session=db_session()
            try: msgs=list(session.query(Message).options(joinedload(Message.sender)).filter_by(order_id=order.id).order_by(Message.created_at.asc()))
            finally: session.close()
            if msgs:
                for m in msgs:
                    sender_name=m.sender.name if m.sender else "User"
                    bubble_class = "pw-message-self" if m.sender_id == current_user()["id"] else "pw-message-other"
                    st.markdown(f'<div class="pw-message {bubble_class}"><div class="pw-message-meta">👤 {html.escape(sender_name)} · {html.escape(m.created_at.strftime("%d %b, %H:%M") if m.created_at else "")}</div><div class="pw-message-text">{html.escape(str(m.message))}</div></div>',unsafe_allow_html=True)
            else:
                st.caption("No messages in this order yet.")
            with st.form(f"msg_{order.id}"):
                text=st.text_input("Message",key=f"msg_text_{order.id}"); send=st.form_submit_button("💬 Send")
            if send:
                s=db_session()
                try: send_message(s,order_id=order.id,sender_id=current_user()["id"],message=text); s.commit(); st.rerun()
                except Exception as exc: s.rollback(); st.error(str(exc))
                finally: s.close()

def profile_page():
    session=db_session()
    try:
        user=session.get(User,current_user()["id"]); loc=user.location; active_count=session.query(DBFarmerListing).filter(DBFarmerListing.farmer_id==user.id,DBFarmerListing.status==ListingStatus.ACTIVE.value).count() if user.role=="farmer" else 0; order_count=len(order_history(session,user.id))
    finally: session.close()
    display_name=user.name or "AgriGuard User"; location_text=", ".join([x for x in [loc.district if loc else None,"Tamil Nadu"] if x]); avatar=''.join(part[0] for part in display_name.split()[:2]).upper() or "PW"; src=html.escape(_image_src("farmer"),quote=True)
    st.markdown(f'''<div class="pw-profile"><div style="display:flex;gap:1rem;align-items:center;"><div class="pw-avatar-wrap"><img class="pw-avatar-img" src="{src}" alt="Farmer profile visual" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';"><div class="pw-avatar-fallback">{html.escape(_initials(display_name))}</div></div><div><div class="pw-kicker" style="color:#f9df8d;">👤 PROFILE</div><h2>{html.escape(display_name)}</h2><span class="pw-chip">{html.escape(user.role.title())}</span><div class="muted" style="margin-top:.45rem;">📍 {html.escape(location_text or "Location not added")}</div></div></div></div>''',unsafe_allow_html=True)
    st.markdown(f'''<div class="pw-stat-grid"><div class="pw-stat green"><div class="icon">🌾</div><div class="label">Crop</div><div class="value">Paddy</div></div><div class="pw-stat gold"><div class="icon">📦</div><div class="label">Active Listings</div><div class="value">{active_count}</div></div><div class="pw-stat blue"><div class="icon">🤝</div><div class="label">Orders</div><div class="value">{order_count}</div></div><div class="pw-stat orange"><div class="icon">⭐</div><div class="label">Profile</div><div class="value">Active</div></div></div>''',unsafe_allow_html=True)
    left,right=st.columns([1.15,.85],gap="large")
    with left:
        _section_heading("Saved information","Profile details","The editable fields and save logic below remain the existing database-backed implementation.","🧑‍🌾")
        with st.form("profile"):
            name=st.text_input("Name",value=user.name); phone=st.text_input("Phone",value=user.phone or ""); address=st.text_input("Address",value=loc.address if loc else ""); village=st.text_input("Village",value=loc.village if loc else ""); taluk=st.text_input("Taluk",value=loc.taluk if loc else ""); district=st.text_input("District",value=loc.district if loc else "Thanjavur"); pincode=st.text_input("Pincode",value=loc.pincode if loc else ""); lat=st.number_input("Latitude",value=float(loc.latitude) if loc and loc.latitude else 10.7870,format="%.6f"); lon=st.number_input("Longitude",value=float(loc.longitude) if loc and loc.longitude else 79.1378,format="%.6f"); save=st.form_submit_button("💾 Save Profile",type="primary",width="stretch")
        if save:
            s=db_session()
            try:
                u=s.get(User,current_user()["id"]); u.name=name.strip(); u.phone=phone or None
                if not u.location: u.location=UserLocation(user_id=u.id)
                u.location.address=address or None; u.location.village=village or None; u.location.taluk=taluk or None; u.location.district=district or None; u.location.pincode=pincode or None; u.location.latitude=Decimal(str(lat)); u.location.longitude=Decimal(str(lon)); s.commit(); st.session_state.user=user_dict(u); st.success("Profile updated.")
            except Exception as exc:
                s.rollback(); st.error("Profile could not be saved. Please try again.")
                with st.expander("Technical details"): st.code(str(exc))
            finally: s.close()
    with right:
        _image_card("farmer","Farmer Profile","A visual identity for the agriculture workspace")
        st.markdown('<div class="pw-note" style="margin-top:.8rem;">Profile imagery is decorative platform photography; your saved account details remain database-backed.</div>',unsafe_allow_html=True)

def market_insights():
    _section_heading("Market intelligence","AgriGuard Intelligence Center","Today, Forecast, Markets and Sell vs Wait — using the existing price and decision services.","📊")
    df=load_prices(); supported=load_current_price_combinations()
    if supported.empty: st.warning("Market Insights is temporarily unavailable because current forecast combinations could not be loaded."); return
    markets=sorted(supported["market"].unique()); market=st.selectbox("Market",markets,key="insights_market"); market_supported=supported[supported["market"]==market]; varieties=sorted(market_supported["variety"].unique()); variety=st.selectbox("Variety",varieties,key="insights_variety"); grades=sorted(market_supported[market_supported["variety"]==variety]["grade"].unique()); grade=st.selectbox("Grade",grades,key="insights_grade"); quantity=st.number_input("Quantity (quintals)",min_value=0.1,value=100.0,step=1.0,key="insights_quantity"); waiting_days=st.number_input("Wait days",min_value=1,max_value=30,value=7,step=1,key="insights_wait_days")
    _image_strip([("paddy","Paddy Market"),("grain","Rice Grain"),("warehouse","Storage"),("field","Rice Field")])
    tabs=st.tabs(["🌾 Today","📈 Forecast","🏪 Markets","🟢 Sell vs Wait"]); forecast=None; forecast_error=None
    with tabs[0]:
        try:
            forecast=forecast_future_prices(market,variety,grade,date.today(),max(30,int(waiting_days))); today_row=forecast.loc[pd.to_datetime(forecast["date"]).dt.date==date.today()]
            if today_row.empty: raise ValueError("The application date is not present in the forecast series.")
            today_value=float(today_row.iloc[0]["predicted_price_rs_per_quintal"])
            st.markdown(f'''<div class="pw-result"><div class="pw-kicker">🤖 TODAY'S AI PRICE</div><div class="big">{money(today_value)}</div><div style="font-weight:800;color:#5f6e62;">per quintal</div><div style="margin-top:.65rem;"><span class="pw-chip">📍 {html.escape(market)}</span><span class="pw-chip">🌾 {html.escape(variety)}</span><span class="pw-chip">⭐ {html.escape(grade)}</span></div></div>''',unsafe_allow_html=True)
            st.caption("AI model estimate — not a live official market price.")
        except Exception as exc:
            forecast_error=exc; st.warning("⚠️ Forecast temporarily unavailable"); st.caption("Please try again.");
            with st.expander("Technical details"): st.code(str(exc))
    with tabs[1]:
        if forecast is None and forecast_error is None:
            try: forecast=forecast_future_prices(market,variety,grade,date.today(),max(30,int(waiting_days)))
            except Exception as exc: forecast_error=exc
        if forecast is not None:
            fig=go.Figure(go.Scatter(x=forecast["date"],y=forecast["predicted_price_rs_per_quintal"],mode="lines+markers",name="AI estimate",line=dict(width=3),marker=dict(size=6))); fig.update_layout(height=350,margin=dict(l=10,r=10,t=20,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.8)",yaxis_title="₹ / quintal",xaxis_title=None); fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor="#e8eef0"); st.plotly_chart(fig,width="stretch",config={"displayModeBar":False})
            cards=forecast.head(5); cols=st.columns(5,gap="small")
            for col,(_,row) in zip(cols,cards.iterrows()):
                with col: st.markdown(f'<div class="pw-card pw-card-blue"><div class="muted">{pd.Timestamp(row["date"]).strftime("%d %b")}</div><div style="font-size:1.3rem;font-weight:900;color:#2367ad;">{money(row["predicted_price_rs_per_quintal"])}</div><div class="muted">AI / quintal</div></div>',unsafe_allow_html=True)
        else:
            st.warning("⚠️ Forecast temporarily unavailable");
            with st.expander("Technical details"): st.code(str(forecast_error))
    with tabs[2]:
        _section_heading("Cauvery Delta market network", "Thanjavur market intelligence map", "Existing market locations with model-based context where supported.", "🗺️")
        dashboard_map(show_prices=True, height=450)
        wait_date=date.today()+pd.Timedelta(days=int(waiting_days)); _show_ai_market_intelligence(market,variety,grade,float(quantity),wait_date,float(DEFAULT_TRANSPORT_COST_PER_KM))
    with tabs[3]:
        if forecast is None and forecast_error is None:
            try: forecast=forecast_future_prices(market,variety,grade,date.today(),max(30,int(waiting_days)))
            except Exception as exc: forecast_error=exc
        if forecast is None:
            st.warning("⚠️ Forecast temporarily unavailable"); st.caption("Please try again.");
            with st.expander("Technical details"): st.code(str(forecast_error))
        else:
            today_row=forecast.iloc[0]; wait_row=forecast.iloc[int(waiting_days)-1]; today_value=float(today_row["predicted_price_rs_per_quintal"]); wait_value=float(wait_row["predicted_price_rs_per_quintal"]); wait_date=pd.Timestamp(wait_row["date"]).date()
            try:
                decision=calculate_decision(DecisionInputs(quantity_quintals=float(quantity),current_market=market,current_modal_price=today_value,forecast_price=wait_value,expected_waiting_days=int(waiting_days),transportation_cost=0.0,storage_cost_per_day=0.0)); a,b=st.columns(2,gap="small")
                with a: st.markdown(f'<div class="pw-card pw-card-green"><div style="font-size:1.6rem;">🟢</div><div class="eyebrow">SELL TODAY</div><div class="pw-price">{money(decision.sell_today.net_revenue)}</div><div class="muted">{money(today_value)} / quintal</div></div>',unsafe_allow_html=True)
                with b: st.markdown(f'<div class="pw-card pw-card-orange"><div style="font-size:1.6rem;">🟠</div><div class="eyebrow">WAIT {waiting_days} DAYS</div><div class="pw-price">{money(decision.wait.net_revenue) if decision.wait else "Unavailable"}</div><div class="muted">{wait_date.strftime("%d %b %Y")} · {money(wait_value)} / quintal</div></div>',unsafe_allow_html=True)
                a,b=st.columns(2); a.metric("Break-even future price",money(decision.break_even_future_price)+" / quintal"); b.metric("Net revenue difference",money(decision.net_revenue_difference)); st.markdown(f'<div class="pw-callout"><b>Decision engine</b><br>{html.escape(str(decision.interpretation))}</div>',unsafe_allow_html=True)
            except Exception as exc:
                st.warning("Sell vs Wait is temporarily unavailable.");
                with st.expander("Technical details"): st.code(str(exc))

def price_intelligence():
    """Backward-compatible entry point; Price Intelligence is now Market Insights."""
    market_insights()


def _load_original_audit_record(record_id: str):
    """Load an existing off-chain audit record without changing verification logic.

    The Phase 7.5 repository intentionally keeps full records off-chain. The
    committed demonstration record PW-DEMO-001 is reproducible from the
    existing blockchain demo inputs, so the UI can load that exact record.
    Other records require their existing off-chain source to be supplied.
    """
    if record_id != "PW-DEMO-001":
        return None, "The existing application does not expose a persisted full off-chain record for this transaction."
    demo_inputs = DecisionInputs(
        quantity_quintals=10, current_market="Vallam", current_modal_price=2000,
        forecast_price=2200, expected_waiting_days=5, transportation_cost=500,
        storage_cost_per_day=20, buyer_offer=2150,
    )
    decision = calculate_decision(demo_inputs)
    record = create_decision_record(
        record_id="PW-DEMO-001", decision_result=decision,
        timestamp="2026-09-20T00:00:00+00:00",
    )
    return record, None


def transaction_audit():
    src=html.escape(_image_src("blockchain"),quote=True)
    st.markdown(f'''<div class="pw-page-top pw-audit-page"><div class="pw-audit-banner"><div style="display:grid;grid-template-columns:1.15fr .85fr;gap:1rem;align-items:center;"><div><div class="pw-kicker">🔐 AGRIGUARD AUDIT LEDGER</div><h2 style="margin:.2rem 0;">Transaction Audit</h2><div class="pw-audit-sub">Tamper-evident verification for records committed to the existing local blockchain.</div></div><div class="pw-audit-image"><img src="{src}" alt="Blockchain technology" onerror="this.style.display='none';"></div></div></div>''',unsafe_allow_html=True)
    try:
        blockchain=load_or_create_blockchain(BLOCKCHAIN_DATA); integrity=validate_blockchain(blockchain)
    except Exception as exc:
        st.error("Blockchain audit is temporarily unavailable.")
        with st.expander("Technical details"): st.code(str(exc))
        return
    blocks=blockchain.blocks
    records=[b for b in blocks if b.record_id!="GENESIS"]
    status_class="green" if integrity["valid"] else "orange"
    status_icon="🟢" if integrity["valid"] else "🔴"
    st.markdown(f'''<div class="pw-stat-grid"><div class="pw-stat {status_class}"><div class="icon">{status_icon}</div><div class="label">Blockchain Status</div><div class="value">{'Verified' if integrity['valid'] else 'Failed'}</div></div><div class="pw-stat blue"><div class="icon">🧾</div><div class="label">Transactions</div><div class="value">{len(records)}</div></div><div class="pw-stat purple"><div class="icon">⛓️</div><div class="label">Blocks</div><div class="value">{len(blocks)}</div></div><div class="pw-stat teal"><div class="icon">🛡️</div><div class="label">Chain Check</div><div class="value">{'Valid' if integrity['valid'] else 'Invalid'}</div></div></div>''',unsafe_allow_html=True)
    if integrity["valid"]:
        st.markdown(f'<div class="pw-verified-banner"><div class="pw-verified-icon">✓</div><div><div class="pw-verified-title">BLOCKCHAIN INTEGRITY VERIFIED</div><div>Chain integrity confirmed · {integrity["blocks_checked"]} block(s) checked.</div></div></div>',unsafe_allow_html=True)
    else:
        st.error("Blockchain integrity check failed.")
    if not records:
        st.info("No transaction records have been committed to the local audit ledger yet.")
        return
    options={f"Block {b.index} · {b.record_id}":b for b in records}
    selected=st.selectbox("Select transaction",list(options),key="audit_transaction")
    block=options[selected]
    st.markdown(f'''<div class="pw-audit-record"><div class="pw-audit-record-top"><div><span class="pw-audit-block">BLOCK {block.index}</span><h3>{html.escape(block.record_id)}</h3></div><span class="pw-verified-pill">● Committed</span></div><div class="pw-hash-grid"><div><span>Record Hash</span><code>{html.escape(block.record_hash)}</code></div><div><span>Previous Hash</span><code>{html.escape(block.previous_hash)}</code></div><div><span>Block Hash</span><code>{html.escape(block.block_hash)}</code></div><div><span>Timestamp</span><code>{html.escape(block.timestamp)}</code></div></div></div>''',unsafe_allow_html=True)
    st.caption("The existing blockchain stores hashes and minimal verification metadata. Full off-chain records are not embedded in the ledger file.")
    record_json=st.text_area("Current record JSON",value=st.session_state.get("audit_record_json", ""),height=190,help="Use the existing off-chain record for the selected record ID. Verification still runs through the existing SHA-256 blockchain verification logic.",key="audit_record_json")
    load_col, verify_col = st.columns([1,1], gap="small")
    with load_col:
        if st.button("📄 Load Original Record",width="stretch",key="load_original_audit"):
            import json
            original, reason = _load_original_audit_record(block.record_id)
            if original is None:
                st.warning(reason)
            else:
                st.session_state.audit_record_json=json.dumps(original,indent=2,ensure_ascii=False)
                st.rerun()
    with verify_col:
        verify_clicked=st.button("🔐 Verify Selected Transaction",type="primary",width="stretch",key="verify_audit")
    if verify_clicked:
        import json
        raw=st.session_state.get("audit_record_json", record_json)
        if not raw.strip():
            st.warning("Load the original record or provide the existing off-chain record JSON before verification.")
        else:
            try:
                current_record=json.loads(raw)
                result=verify_record(blockchain,block.record_id,current_record)
                if result["verified"]:
                    st.markdown(f'<div class="pw-verified-banner pw-verified-strong"><div class="pw-verified-icon">✓</div><div><div class="pw-verified-title">TRANSACTION VERIFIED</div><div>Blockchain integrity confirmed for <b>{html.escape(block.record_id)}</b>.</div></div></div>',unsafe_allow_html=True)
                else:
                    reason=result.get("reason") or result.get("status") or "The supplied record does not match the committed blockchain record."
                    st.markdown(f'<div class="pw-failed-banner"><div class="pw-failed-icon">!</div><div><div class="pw-failed-title">VERIFICATION FAILED</div><div>{html.escape(str(reason))}</div></div></div>',unsafe_allow_html=True)
                with st.expander("Verification details"):
                    st.json(result)
            except Exception as exc:
                st.markdown(f'<div class="pw-failed-banner"><div class="pw-failed-icon">!</div><div><div class="pw-failed-title">VERIFICATION FAILED</div><div>The record could not be parsed as valid JSON.</div></div></div>',unsafe_allow_html=True)
                with st.expander("Technical details"): st.code(str(exc))
    st.markdown('</div>', unsafe_allow_html=True)

def _reset_page_scroll():
    """UI-only: reset the parent browser scroll after Streamlit navigation/reruns."""
    components.html(
        """<script>
        try { window.parent.scrollTo({top: 0, left: 0, behavior: 'auto'}); }
        catch (e) { try { window.parent.scrollTo(0, 0); } catch (_) {} }
        </script>""",
        height=0,
    )


def main():
    inject_css(); get_engine(); _reset_page_scroll()
    if not current_user():
        if st.query_params.get("auth") == "login":
            auth_page()
        else:
            home_page(); st.divider(); auth_page()
        return
    page=sidebar()
    nav_override = st.session_state.pop("nav_override", None)
    if nav_override:
        page = nav_override
    page = {
        "🏠 Farmer Dashboard":"Farmer Dashboard", "🌾 Sell Paddy":"Sell Paddy", "📦 My Listings":"My Listings",
        "🤝 Buyer Requests":"Buyer Requests", "🚚 My Orders":"My Orders", "💬 Messages":"Messages",
        "📊 Market Insights":"Market Insights", "🔐 Transaction Audit":"Transaction Audit", "👤 Profile":"Profile",
        "🏠 Buyer Dashboard":"Buyer Dashboard", "🔎 Find Paddy":"Find Paddy", "📋 My Buy Requests":"My Buy Requests",
    }.get(page, page)
    routes={
        "Farmer Dashboard":farmer_dashboard,"Sell Paddy":sell_paddy,"My Listings":my_listings,"Buyer Requests":farmer_requests,
        "Buyer Dashboard":buyer_dashboard,"Find Paddy":find_paddy,"My Buy Requests":my_buy_requests,"My Orders":my_orders,"Messages":messages_page,"Profile":profile_page,"Market Insights":market_insights,"Transaction Audit":transaction_audit,"Price Intelligence":price_intelligence,
    }
    routes.get(page, home_page)()

if __name__ == "__main__":main()
