import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design System ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* Reset & base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 2rem 2.5rem; max-width: 1100px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0a0a0a;
    border-right: 1px solid #1e1e1e;
}
[data-testid="stSidebar"] * { color: #a0a0a0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.85rem; letter-spacing: 0.02em; }

/* Main background */
.stApp { background: #0d0d0d; color: #e0e0e0; }

/* Typography */
h1 { font-size: 1.6rem !important; font-weight: 600 !important; color: #ffffff !important; letter-spacing: -0.02em !important; }
h2 { font-size: 1.1rem !important; font-weight: 500 !important; color: #cccccc !important; letter-spacing: -0.01em !important; margin-top: 2rem !important; }
h3 { font-size: 0.9rem !important; font-weight: 500 !important; color: #888888 !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #141414;
    border: 1px solid #1e1e1e;
    border-radius: 8px;
    padding: 1.2rem 1.4rem !important;
}
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: #666 !important; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 600 !important; color: #fff !important; font-family: 'JetBrains Mono', monospace !important; }

/* Buttons */
.stButton > button {
    background: #ffffff !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    padding: 0.6rem 1.4rem !important;
    transition: opacity 0.15s ease !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: #e0e0e0 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 6px !important;
    font-size: 0.82rem !important;
    width: 100%;
}
[data-testid="stDownloadButton"] > button:hover { border-color: #555 !important; }

/* Form inputs */
.stSelectbox > div > div, .stNumberInput > div > div > input,
.stTextInput > div > div > input, .stDateInput > div > div > input {
    background: #141414 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 6px !important;
    color: #e0e0e0 !important;
    font-size: 0.85rem !important;
}
.stSlider > div > div { color: #e0e0e0 !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #141414;
    border: 1px dashed #2a2a2a;
    border-radius: 8px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #1e1e1e; gap: 0; }
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #666 !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.04em !important;
    border-radius: 0 !important;
    padding: 0.6rem 1.2rem !important;
}
.stTabs [aria-selected="true"] { color: #ffffff !important; border-bottom: 1px solid #ffffff !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #1e1e1e; border-radius: 8px; overflow: hidden; }

/* Divider */
hr { border-color: #1e1e1e !important; margin: 1.5rem 0 !important; }

/* Alert boxes */
.stAlert { border-radius: 6px !important; font-size: 0.85rem !important; }

/* Custom pill badge */
.pill {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.03em;
}
.pill-green  { background: #0d2b1a; color: #4ade80; border: 1px solid #166534; }
.pill-yellow { background: #2b2200; color: #facc15; border: 1px solid #854d0e; }
.pill-red    { background: #2b0d0d; color: #f87171; border: 1px solid #991b1b; }

/* Stat card */
.stat-card {
    background: #141414;
    border: 1px solid #1e1e1e;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.75rem;
}
.stat-label { font-size: 0.72rem; color: #555; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.3rem; }
.stat-value { font-size: 1.5rem; font-weight: 600; color: #fff; font-family: 'JetBrains Mono', monospace; }
.stat-sub   { font-size: 0.75rem; color: #555; margin-top: 0.2rem; }

/* Section label */
.section-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #444;
    margin-bottom: 1rem;
    margin-top: 1.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1a1a1a;
}

/* Result card */
.result-card {
    background: #141414;
    border: 1px solid #1e1e1e;
    border-radius: 10px;
    padding: 1.8rem 2rem;
    margin: 1.2rem 0;
}
.result-card.churn   { border-left: 3px solid #f87171; }
.result-card.no-churn { border-left: 3px solid #4ade80; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("churn_model_final.pkl", "rb") as f:
        return pickle.load(f)

try:
    art = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#141414',
    'axes.facecolor': '#141414',
    'axes.edgecolor': '#2a2a2a',
    'axes.labelcolor': '#888',
    'xtick.color': '#555',
    'ytick.color': '#555',
    'text.color': '#ccc',
    'grid.color': '#1e1e1e',
    'grid.linewidth': 0.6,
})

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(df_input, art):
    df = df_input.copy()
    if "age" in df.columns:
        df["age"] = df["age"].clip(lower=0, upper=100)
    iv = art["imputation_values"]
    if "coupon_code" in df.columns: df["coupon_code"] = df["coupon_code"].fillna("No Coupon")
    if "gender" in df.columns:      df["gender"] = df["gender"].fillna(iv["mode_gender"])
    if "age" in df.columns:         df["age"] = df["age"].fillna(iv["median_age"])
    if "total_spent" in df.columns: df["total_spent"] = df["total_spent"].fillna(iv["median_spent"])
    if "satisfaction_score" in df.columns: df["satisfaction_score"] = df["satisfaction_score"].fillna(iv["median_sat"])
    ref = art["reference_date"]
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"])
    df["customer_tenure_days"] = (ref - df["signup_date"]).dt.days
    df["recency_days"] = (ref - df["last_purchase_date"]).dt.days
    df["signup_year"] = df["signup_date"].dt.year
    df["signup_month"] = df["signup_date"].dt.month
    df["spend_per_visit"] = df["total_spent"] / (df["total_visits"] + 1)
    df["engagement_score"] = (df["email_open_rate"] * 0.5) + (df["email_click_rate"] * 0.5)
    df["refund_rate"] = df["refund_requested"] / (df["support_tickets"] + 1)
    drop_cols = ["customer_id", "signup_date", "last_purchase_date", "city", "churn"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    for col in art["cat_cols"]:
        if col in df.columns:
            le = art["label_encoders"][col]
            df[col] = df[col].astype(str).map(lambda x, le=le: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
    df = df[art["feature_order"]]
    df_scaled = df.copy()
    df_scaled[art["scale_cols"]] = art["scaler"].transform(df[art["scale_cols"]])
    return df_scaled

def predict(df_proc, art, threshold):
    proba = art["model"].predict_proba(df_proc)[:, 1]
    pred  = (proba >= threshold).astype(int)
    risk  = pd.cut(proba, bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"])
    return proba, pred, risk

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='padding:1rem 0 0.5rem'><span style='font-size:1.1rem;font-weight:600;color:#fff;letter-spacing:-0.01em'>◈ Churn Predictor</span></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.75rem;color:#444;margin-bottom:1.5rem'>UAS Data Science</div>", unsafe_allow_html=True)
    st.divider()

    menu = st.radio("", ["Overview", "Single Prediction", "Batch Upload", "Model Info"], label_visibility="collapsed")

    st.divider()
    THRESHOLD = st.slider("Threshold", 0.1, 0.9, 0.42, 0.01)
    st.markdown(f"<div style='font-size:0.72rem;color:#444;margin-top:-0.5rem'>Probability cutoff for churn classification</div>", unsafe_allow_html=True)

if not model_loaded:
    st.error("churn_model_final.pkl not found. Run Week 5 notebook first.")
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if menu == "Overview":
    st.markdown("<h1>Customer Churn Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:#555;font-size:0.9rem;margin-top:-0.5rem;margin-bottom:2rem'>XGBoost · Sales & Marketing Dataset · UDINUS</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "XGBoost")
    c2.metric("Features", str(len(art["feature_order"])))
    c3.metric("Threshold", f"{THRESHOLD:.2f}")
    c4.metric("Reference Date", str(art["reference_date"].date()))

    st.divider()

    st.markdown("<div class='section-label'>How to use</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-label'>Single Prediction</div>
            <div style='color:#ccc;font-size:0.85rem;margin-top:0.5rem;line-height:1.6'>
                Fill in customer data manually via form. Get instant churn probability and risk label.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-label'>Batch Upload</div>
            <div style='color:#ccc;font-size:0.85rem;margin-top:0.5rem;line-height:1.6'>
                Upload a CSV file with multiple customers. Download results with churn probabilities.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Risk labels</div>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""
        <div class='stat-card'>
            <span class='pill pill-green'>Low Risk</span>
            <div style='color:#555;font-size:0.8rem;margin-top:0.7rem'>Probability &lt; 0.30</div>
            <div style='color:#aaa;font-size:0.82rem;margin-top:0.3rem'>Customer likely to stay</div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("""
        <div class='stat-card'>
            <span class='pill pill-yellow'>Medium Risk</span>
            <div style='color:#555;font-size:0.8rem;margin-top:0.7rem'>Probability 0.30 – 0.60</div>
            <div style='color:#aaa;font-size:0.82rem;margin-top:0.3rem'>Monitor and consider retention</div>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown("""
        <div class='stat-card'>
            <span class='pill pill-red'>High Risk</span>
            <div style='color:#555;font-size:0.8rem;margin-top:0.7rem'>Probability &gt; 0.60</div>
            <div style='color:#aaa;font-size:0.82rem;margin-top:0.3rem'>Immediate intervention needed</div>
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SINGLE PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif menu == "Single Prediction":
    st.markdown("<h1>Single Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:#555;font-size:0.85rem;margin-bottom:2rem'>Enter customer data to predict churn probability</div>", unsafe_allow_html=True)

    with st.form("form_pred"):
        st.markdown("<div class='section-label'>Demographics</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        gender  = c1.selectbox("Gender", ["Male", "Female", "Other"])
        age     = c2.number_input("Age", 18, 80, 32)
        country = c3.selectbox("Country", ["India", "USA", "Germany", "UK", "Bangladesh"])
        is_prem = c4.selectbox("Premium User", [0, 1], format_func=lambda x: "Yes" if x else "No")

        c5, c6 = st.columns(2)
        signup_date  = c5.date_input("Signup Date", value=pd.to_datetime("2022-06-01"))
        last_purch   = c6.date_input("Last Purchase Date", value=pd.to_datetime("2024-08-15"))

        st.markdown("<div class='section-label'>Subscription & Payment</div>", unsafe_allow_html=True)
        c7, c8, c9, c10 = st.columns(4)
        sub_type   = c7.selectbox("Subscription", ["Monthly", "Annual"])
        pay_method = c8.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"])
        acq_ch     = c9.selectbox("Acquisition Channel", ["Organic", "Email", "Paid Ad", "Referral", "Social Media"])
        device     = c10.selectbox("Device", ["Mobile", "Desktop", "Tablet"])

        st.markdown("<div class='section-label'>Financials</div>", unsafe_allow_html=True)
        c11, c12, c13, c14 = st.columns(4)
        total_spent  = c11.number_input("Total Spent ($)", 0.0, 10000.0, 350.0)
        avg_order    = c12.number_input("Avg Order Value ($)", 0.0, 5000.0, 55.0)
        ltv          = c13.number_input("Lifetime Value ($)", 0.0, 20000.0, 600.0)
        mktg_spend   = c14.number_input("Marketing Spend ($)", 0.0, 100.0, 12.0)

        st.markdown("<div class='section-label'>Engagement</div>", unsafe_allow_html=True)
        c15, c16, c17 = st.columns(3)
        total_visits = c15.number_input("Total Visits", 0, 500, 25)
        avg_sess     = c16.number_input("Avg Session (min)", 0.0, 60.0, 6.0)
        pages_sess   = c17.number_input("Pages/Session", 0.0, 30.0, 3.5)

        c18, c19 = st.columns(2)
        email_open  = c18.slider("Email Open Rate", 0.0, 1.0, 0.30)
        email_click = c19.slider("Email Click Rate", 0.0, 1.0, 0.08)

        st.markdown("<div class='section-label'>Support & Satisfaction</div>", unsafe_allow_html=True)
        c20, c21, c22, c23, c24 = st.columns(5)
        support_tk  = c20.number_input("Support Tickets", 0, 20, 1)
        refund_req  = c21.selectbox("Refund Requested", [0, 1], format_func=lambda x: "Yes" if x else "No")
        delay_days  = c22.number_input("Delivery Delay (days)", 0, 30, 1)
        sat_score   = c23.slider("Satisfaction (1-5)", 1.0, 5.0, 3.5)
        nps         = c24.slider("NPS (0-10)", 0, 10, 6)

        c25, c26 = st.columns([1, 3])
        disc_used   = c25.selectbox("Discount Used", [0, 1], format_func=lambda x: "Yes" if x else "No")
        purch_freq  = c26.number_input("Purchase Freq (last 3 months)", 0, 30, 2)

        submitted = st.form_submit_button("Run prediction", use_container_width=False)

    if submitted:
        input_df = pd.DataFrame([{
            "customer_id": 0, "gender": gender, "age": age, "country": country,
            "city": "Unknown", "signup_date": str(signup_date),
            "last_purchase_date": str(last_purch),
            "acquisition_channel": acq_ch, "device_type": device,
            "subscription_type": sub_type, "is_premium_user": is_prem,
            "total_visits": total_visits, "avg_session_time": avg_sess,
            "pages_per_session": pages_sess, "email_open_rate": email_open,
            "email_click_rate": email_click, "total_spent": total_spent,
            "avg_order_value": avg_order, "discount_used": disc_used,
            "coupon_code": "No Coupon", "support_tickets": support_tk,
            "refund_requested": refund_req, "delivery_delay_days": delay_days,
            "payment_method": pay_method, "satisfaction_score": sat_score,
            "nps_score": nps, "marketing_spend_per_user": mktg_spend,
            "lifetime_value": ltv, "last_3_month_purchase_freq": purch_freq
        }])

        df_proc = preprocess(input_df, art)
        proba, pred, risk = predict(df_proc, art, THRESHOLD)
        p = proba[0]
        is_churn = pred[0] == 1

        card_class = "churn" if is_churn else "no-churn"
        risk_str = str(risk[0])
        if risk_str == "High":   pill = "<span class='pill pill-red'>High Risk</span>"
        elif risk_str == "Medium": pill = "<span class='pill pill-yellow'>Medium Risk</span>"
        else: pill = "<span class='pill pill-green'>Low Risk</span>"

        verdict = "Will churn" if is_churn else "Will not churn"
        verdict_color = "#f87171" if is_churn else "#4ade80"

        st.markdown(f"""
        <div class='result-card {card_class}'>
            <div style='display:flex;justify-content:space-between;align-items:flex-start'>
                <div>
                    <div style='font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#555;margin-bottom:0.5rem'>Prediction result</div>
                    <div style='font-size:1.8rem;font-weight:600;color:{verdict_color};font-family:JetBrains Mono,monospace'>{verdict}</div>
                    <div style='margin-top:0.8rem'>{pill}</div>
                </div>
                <div style='text-align:right'>
                    <div style='font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#555;margin-bottom:0.3rem'>Churn probability</div>
                    <div style='font-size:2.8rem;font-weight:600;color:#fff;font-family:JetBrains Mono,monospace;line-height:1'>{p*100:.1f}%</div>
                    <div style='font-size:0.75rem;color:#444;margin-top:0.3rem'>Threshold · {THRESHOLD:.2f}</div>
                </div>
            </div>
            <div style='margin-top:1.2rem;background:#0d0d0d;border-radius:4px;height:4px;overflow:hidden'>
                <div style='width:{p*100:.1f}%;height:100%;background:{verdict_color};border-radius:4px;transition:width 0.4s ease'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# BATCH UPLOAD
# ═════════════════════════════════════════════════════════════════════════════
elif menu == "Batch Upload":
    st.markdown("<h1>Batch Upload</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:#555;font-size:0.85rem;margin-bottom:2rem'>Upload a CSV file to predict churn for multiple customers</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df_up = pd.read_csv(uploaded)
        has_actual = "churn" in df_up.columns
        y_actual = df_up["churn"].values if has_actual else None

        with st.spinner("Processing..."):
            df_inp = df_up.drop(columns=["churn"], errors="ignore")
            df_proc = preprocess(df_inp, art)
            proba, pred, risk = predict(df_proc, art, THRESHOLD)

        df_res = df_up.copy()
        df_res["churn_probability"] = (proba * 100).round(2)
        df_res["churn_prediction"]  = pred
        df_res["risk_label"]        = risk.astype(str)

        n_total  = len(pred)
        n_churn  = pred.sum()
        n_high   = (risk == "High").sum()
        n_medium = (risk == "Medium").sum()
        n_low    = (risk == "Low").sum()

        st.markdown("<div class='section-label'>Summary</div>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total", f"{n_total:,}")
        c2.metric("Predicted Churn", f"{n_churn:,}")
        c3.metric("High Risk", f"{n_high:,}")
        c4.metric("Medium Risk", f"{n_medium:,}")
        c5.metric("Low Risk", f"{n_low:,}")

        # Charts
        st.markdown("<div class='section-label'>Distribution</div>", unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.patch.set_facecolor('#141414')

        # Donut churn
        ax = axes[0]
        sizes = [n_total - n_churn, n_churn]
        colors = ['#1e3a2a', '#3a1e1e']
        wedge_colors = ['#4ade80', '#f87171']
        wedges, _ = ax.pie(sizes, colors=wedge_colors, startangle=90,
                           wedgeprops=dict(width=0.45, edgecolor='#141414', linewidth=2))
        ax.text(0, 0, f'{n_churn/n_total*100:.1f}%', ha='center', va='center',
                fontsize=14, fontweight='600', color='#fff', fontfamily='monospace')
        ax.text(0, -0.18, 'churn rate', ha='center', va='center', fontsize=7.5, color='#555')
        ax.set_title('Churn Rate', fontsize=9, color='#666', pad=12, fontweight='400')

        # Probability histogram
        ax2 = axes[1]
        ax2.hist(proba[pred==0], bins=25, color='#4ade8033', edgecolor='#4ade80', linewidth=0.5, alpha=0.9, label='No churn')
        ax2.hist(proba[pred==1], bins=25, color='#f8717133', edgecolor='#f87171', linewidth=0.5, alpha=0.9, label='Churn')
        ax2.axvline(THRESHOLD, color='#ffffff55', linestyle='--', linewidth=1, label=f'Threshold')
        ax2.set_xlabel('Probability', fontsize=8)
        ax2.set_title('Probability Distribution', fontsize=9, color='#666', pad=12, fontweight='400')
        ax2.legend(fontsize=7, framealpha=0)
        ax2.grid(True, alpha=0.3)

        # Risk bar
        ax3 = axes[2]
        risk_vals = [n_low, n_medium, n_high]
        risk_lbls = ['Low', 'Medium', 'High']
        risk_clrs = ['#4ade80', '#facc15', '#f87171']
        bars = ax3.bar(risk_lbls, risk_vals, color=risk_clrs, width=0.5, edgecolor='#141414', linewidth=1)
        for bar, val in zip(bars, risk_vals):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(risk_vals)*0.02,
                     str(val), ha='center', fontsize=8, color='#aaa')
        ax3.set_title('Risk Breakdown', fontsize=9, color='#666', pad=12, fontweight='400')
        ax3.grid(True, alpha=0.3, axis='y')

        plt.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close()

        # Evaluation if labels available
        if has_actual:
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            st.markdown("<div class='section-label'>Evaluation</div>", unsafe_allow_html=True)
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Accuracy",  f"{accuracy_score(y_actual, pred):.4f}")
            e2.metric("F1-Score",  f"{f1_score(y_actual, pred):.4f}")
            e3.metric("Precision", f"{precision_score(y_actual, pred):.4f}")
            e4.metric("Recall",    f"{recall_score(y_actual, pred):.4f}")

        # Table
        st.markdown("<div class='section-label'>Results</div>", unsafe_allow_html=True)
        show_cols = [c for c in ["customer_id","country","subscription_type","total_spent",
                                  "satisfaction_score","churn_probability","churn_prediction","risk_label"]
                     if c in df_res.columns]
        st.dataframe(df_res[show_cols], use_container_width=True, height=300)

        csv_out = df_res.to_csv(index=False).encode("utf-8")
        st.download_button("Download results (.csv)", csv_out,
                           "churn_predictions.csv", "text/csv")

# ═════════════════════════════════════════════════════════════════════════════
# MODEL INFO
# ═════════════════════════════════════════════════════════════════════════════
elif menu == "Model Info":
    st.markdown("<h1>Model Info</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:#555;font-size:0.85rem;margin-bottom:2rem'>XGBoost classifier trained on Sales & Marketing dataset</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Feature Importance", "Parameters"])

    with tab1:
        importances = pd.Series(
            art["model"].feature_importances_, index=art["feature_order"]
        ).sort_values(ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor('#141414')
        norm = plt.Normalize(importances.values.min(), importances.values.max())
        colors = plt.cm.Blues(norm(importances.values[::-1]) * 0.7 + 0.3)
        bars = ax.barh(importances.index[::-1], importances.values[::-1],
                       color=colors, height=0.65, edgecolor='#141414', linewidth=0.5)
        for bar, val in zip(bars, importances.values[::-1]):
            ax.text(bar.get_width() + importances.max()*0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=8, color='#666')
        ax.set_xlabel('Importance Score', fontsize=8, color='#666')
        ax.set_title('Top 20 Features', fontsize=10, color='#888', pad=14, fontweight='400')
        ax.grid(True, alpha=0.2, axis='x')
        ax.set_xlim(0, importances.max() * 1.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        params = art["model"].get_params()
        param_df = pd.DataFrame({"Parameter": list(params.keys()), "Value": [str(v) for v in params.values()]})
        st.dataframe(param_df, use_container_width=True, hide_index=True)
        st.markdown("<div class='section-label'>Feature list</div>", unsafe_allow_html=True)
        feat_df = pd.DataFrame({"#": range(1, len(art["feature_order"])+1), "Feature": art["feature_order"]})
        st.dataframe(feat_df, use_container_width=True, hide_index=True, height=300)
