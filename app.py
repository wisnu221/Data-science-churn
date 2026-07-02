import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="◈",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 3rem; max-width: 1200px; }
.stApp { background: #0d0d0d; color: #e0e0e0; }

/* Top nav bar */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0 1.5rem 0;
    border-bottom: 1px solid #1a1a1a;
    margin-bottom: 2rem;
}
.topbar-logo { font-size: 1rem; font-weight: 600; color: #fff; letter-spacing: -0.02em; }
.topbar-sub  { font-size: 0.72rem; color: #444; margin-top: 0.1rem; }

/* Tabs as nav */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0 !important;
    border-bottom: 1px solid #1a1a1a !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #555 !important;
    font-size: 0.8rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1.4rem !important;
    border-radius: 0 !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #fff !important;
    border-bottom: 1px solid #fff !important;
    font-weight: 500 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

/* Threshold slider in top right */
.threshold-bar {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    font-size: 0.75rem;
    color: #555;
}

/* Headings */
h1 { font-size: 1.5rem !important; font-weight: 600 !important; color: #fff !important; letter-spacing: -0.02em !important; margin-bottom: 0.3rem !important; }
h2 { font-size: 1rem !important; font-weight: 500 !important; color: #bbb !important; }
h3 { font-size: 0.72rem !important; font-weight: 500 !important; color: #555 !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; }

/* Metric */
[data-testid="stMetric"] {
    background: #141414 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 8px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] p { font-size: 0.7rem !important; color: #555 !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; }
[data-testid="stMetricValue"]   { font-size: 1.5rem !important; font-weight: 600 !important; color: #fff !important; font-family: 'JetBrains Mono', monospace !important; }

/* Buttons */
.stButton > button {
    background: #fff !important; color: #000 !important;
    border: none !important; border-radius: 5px !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    padding: 0.55rem 1.3rem !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.8 !important; }
[data-testid="stDownloadButton"] > button {
    background: transparent !important; color: #aaa !important;
    border: 1px solid #2a2a2a !important; border-radius: 5px !important;
    font-size: 0.8rem !important; width: 100%;
}
[data-testid="stDownloadButton"] > button:hover { border-color: #555 !important; }

/* Inputs */
.stSelectbox > div > div, .stNumberInput input, .stTextInput input, .stDateInput input {
    background: #141414 !important; border: 1px solid #1e1e1e !important;
    border-radius: 6px !important; color: #e0e0e0 !important; font-size: 0.83rem !important;
}
label { color: #777 !important; font-size: 0.78rem !important; }
.stSlider [data-testid="stTickBar"] { color: #444 !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #141414; border: 1.5px dashed #222; border-radius: 8px;
}

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #1e1e1e; border-radius: 8px; }

/* Divider */
hr { border-color: #1a1a1a !important; margin: 1.2rem 0 !important; }

/* Cards */
.card {
    background: #141414; border: 1px solid #1e1e1e;
    border-radius: 8px; padding: 1.2rem 1.4rem; margin-bottom: 0.6rem;
}
.card-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.09em; color: #444; margin-bottom: 0.5rem; }
.card-value { font-size: 1.4rem; font-weight: 600; color: #fff; font-family: 'JetBrains Mono', monospace; }
.card-desc  { font-size: 0.8rem; color: #666; margin-top: 0.4rem; line-height: 1.5; }

/* Section label */
.sec { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; color: #3a3a3a;
       border-bottom: 1px solid #1a1a1a; padding-bottom: 0.4rem; margin: 1.6rem 0 1rem 0; }

/* Pills */
.pill { display: inline-block; padding: 0.18rem 0.65rem; border-radius: 99px; font-size: 0.72rem; font-weight: 500; letter-spacing: 0.03em; }
.pl-g { background:#0d2b1a; color:#4ade80; border:1px solid #166534; }
.pl-y { background:#2b2200; color:#facc15; border:1px solid #854d0e; }
.pl-r { background:#2b0d0d; color:#f87171; border:1px solid #991b1b; }

/* Result */
.res-card { background:#141414; border:1px solid #1e1e1e; border-radius:10px; padding:1.6rem 2rem; margin:1rem 0; }
.res-card.churn    { border-left:3px solid #f87171; }
.res-card.no-churn { border-left:3px solid #4ade80; }
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

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':'#141414','axes.facecolor':'#141414',
    'axes.edgecolor':'#2a2a2a','axes.labelcolor':'#888',
    'xtick.color':'#555','ytick.color':'#555',
    'text.color':'#ccc','grid.color':'#1e1e1e','grid.linewidth':0.6,
})

# ── Pipeline ──────────────────────────────────────────────────────────────────
def preprocess(df_input, art):
    df = df_input.copy()
    if "age" in df.columns: df["age"] = df["age"].clip(0, 100)
    iv = art["imputation_values"]
    for col, val in [("coupon_code","No Coupon"),("gender",iv["mode_gender"]),
                     ("age",iv["median_age"]),("total_spent",iv["median_spent"]),
                     ("satisfaction_score",iv["median_sat"])]:
        if col in df.columns: df[col] = df[col].fillna(val)
    ref = art["reference_date"]
    df["signup_date"]       = pd.to_datetime(df["signup_date"])
    df["last_purchase_date"]= pd.to_datetime(df["last_purchase_date"])
    df["customer_tenure_days"] = (ref - df["signup_date"]).dt.days
    df["recency_days"]         = (ref - df["last_purchase_date"]).dt.days
    df["signup_year"]          = df["signup_date"].dt.year
    df["signup_month"]         = df["signup_date"].dt.month
    df["spend_per_visit"]  = df["total_spent"] / (df["total_visits"] + 1)
    df["engagement_score"] = df["email_open_rate"]*0.5 + df["email_click_rate"]*0.5
    df["refund_rate"]      = df["refund_requested"] / (df["support_tickets"] + 1)
    df = df.drop(columns=[c for c in ["customer_id","signup_date","last_purchase_date","city","churn"] if c in df.columns])
    for col in art["cat_cols"]:
        if col in df.columns:
            le = art["label_encoders"][col]
            df[col] = df[col].astype(str).map(lambda x,le=le: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
    df = df[art["feature_order"]]
    df_s = df.copy()
    df_s[art["scale_cols"]] = art["scaler"].transform(df[art["scale_cols"]])
    return df_s

def predict(df_proc, art, thr):
    proba = art["model"].predict_proba(df_proc)[:,1]
    pred  = (proba >= thr).astype(int)
    return proba, pred

# ── Top bar ───────────────────────────────────────────────────────────────────
col_logo, col_thr = st.columns([3, 1])
with col_logo:
    st.markdown("""
    <div class='topbar'>
        <div>
            <div class='topbar-logo'>◈ Churn Predictor</div>
            <div class='topbar-sub'>UAS Data Science · Sales & Marketing Dataset</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_thr:
    st.markdown("<div style='padding-top:1rem'></div>", unsafe_allow_html=True)
    THRESHOLD = st.slider("Threshold", 0.10, 0.90, 0.42, 0.01, label_visibility="visible")

if not model_loaded:
    st.error("churn_model_final.pkl not found. Run Week 5 notebook first.")
    st.stop()

# ── Navigation tabs ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Single Prediction", "Batch Upload", "Model Info"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model",     "Random Forest (Tuned)")
    c2.metric("Features",  str(len(art["feature_order"])))
    c3.metric("Threshold", f"{THRESHOLD:.2f}")
    c4.metric("Ref Date",  str(art["reference_date"].date()))

    st.markdown("<div class='sec'>How to use</div>", unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        st.markdown("""<div class='card'>
            <div class='card-label'>Single Prediction</div>
            <div class='card-desc'>Fill in one customer's data via form. Get instant churn probability.</div>
        </div>""", unsafe_allow_html=True)
    with cb:
        st.markdown("""<div class='card'>
            <div class='card-label'>Batch Upload</div>
            <div class='card-desc'>Upload a CSV with multiple customers. Download predictions with probabilities.</div>
        </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — SINGLE PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    with st.form("form_pred"):
        st.markdown("<div class='sec'>Demographics</div>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        gender  = c1.selectbox("Gender",  ["Male","Female","Other"])
        age     = c2.number_input("Age", 18, 80, 32)
        country = c3.selectbox("Country", ["India","USA","Germany","UK","Bangladesh"])
        is_prem = c4.selectbox("Premium User", [0,1], format_func=lambda x:"Yes" if x else "No")

        c5,c6 = st.columns(2)
        signup_date = c5.date_input("Signup Date",    value=pd.to_datetime("2022-06-01"))
        last_purch  = c6.date_input("Last Purchase",  value=pd.to_datetime("2024-08-15"))

        st.markdown("<div class='sec'>Subscription & Payment</div>", unsafe_allow_html=True)
        c7,c8,c9,c10 = st.columns(4)
        sub_type   = c7.selectbox("Subscription",        ["Monthly","Annual"])
        pay_method = c8.selectbox("Payment Method",      ["Credit Card","Debit Card","PayPal","Bank Transfer"])
        acq_ch     = c9.selectbox("Acquisition Channel", ["Organic","Email","Paid Ad","Referral","Social Media"])
        device     = c10.selectbox("Device",             ["Mobile","Desktop","Tablet"])

        st.markdown("<div class='sec'>Financials</div>", unsafe_allow_html=True)
        c11,c12,c13,c14 = st.columns(4)
        total_spent = c11.number_input("Total Spent ($)",      0.0, 10000.0, 350.0)
        avg_order   = c12.number_input("Avg Order Value ($)",  0.0, 5000.0,   55.0)
        ltv         = c13.number_input("Lifetime Value ($)",   0.0, 20000.0, 600.0)
        mktg_spend  = c14.number_input("Marketing Spend ($)",  0.0, 100.0,    12.0)

        st.markdown("<div class='sec'>Engagement</div>", unsafe_allow_html=True)
        c15,c16,c17 = st.columns(3)
        total_visits = c15.number_input("Total Visits",         0, 500, 25)
        avg_sess     = c16.number_input("Avg Session (min)",    0.0, 60.0, 6.0)
        pages_sess   = c17.number_input("Pages / Session",      0.0, 30.0, 3.5)
        c18,c19 = st.columns(2)
        email_open  = c18.slider("Email Open Rate",  0.0, 1.0, 0.30)
        email_click = c19.slider("Email Click Rate", 0.0, 1.0, 0.08)

        st.markdown("<div class='sec'>Support & Satisfaction</div>", unsafe_allow_html=True)
        c20,c21,c22,c23,c24 = st.columns(5)
        support_tk = c20.number_input("Support Tickets", 0, 20, 1)
        refund_req = c21.selectbox("Refund Requested", [0,1], format_func=lambda x:"Yes" if x else "No")
        delay_days = c22.number_input("Delivery Delay (days)", 0, 30, 1)
        sat_score  = c23.slider("Satisfaction (1–5)", 1.0, 5.0, 3.5)
        nps        = c24.slider("NPS (0–10)", 0, 10, 6)

        c25,c26 = st.columns([1,3])
        disc_used  = c25.selectbox("Discount Used", [0,1], format_func=lambda x:"Yes" if x else "No")
        purch_freq = c26.number_input("Purchase Freq (last 3 months)", 0, 30, 2)

        submitted = st.form_submit_button("Run prediction", use_container_width=False)

    if submitted:
        inp = pd.DataFrame([{
            "customer_id":0,"gender":gender,"age":age,"country":country,"city":"Unknown",
            "signup_date":str(signup_date),"last_purchase_date":str(last_purch),
            "acquisition_channel":acq_ch,"device_type":device,"subscription_type":sub_type,
            "is_premium_user":is_prem,"total_visits":total_visits,"avg_session_time":avg_sess,
            "pages_per_session":pages_sess,"email_open_rate":email_open,"email_click_rate":email_click,
            "total_spent":total_spent,"avg_order_value":avg_order,"discount_used":disc_used,
            "coupon_code":"No Coupon","support_tickets":support_tk,"refund_requested":refund_req,
            "delivery_delay_days":delay_days,"payment_method":pay_method,"satisfaction_score":sat_score,
            "nps_score":nps,"marketing_spend_per_user":mktg_spend,"lifetime_value":ltv,
            "last_3_month_purchase_freq":purch_freq
        }])
        df_proc = preprocess(inp, art)
        proba, pred = predict(df_proc, art, THRESHOLD)
        p = proba[0]; is_churn = pred[0]==1
        verdict = "Will churn" if is_churn else "Will not churn"
        vc = "#f87171" if is_churn else "#4ade80"
        cc = "churn" if is_churn else "no-churn"
        st.markdown(f"""
        <div class='res-card {cc}'>
            <div style='display:flex;justify-content:space-between;align-items:flex-start'>
                <div>
                    <div style='font-size:.68rem;text-transform:uppercase;letter-spacing:.09em;color:#444;margin-bottom:.4rem'>Result</div>
                    <div style='font-size:1.6rem;font-weight:600;color:{vc};font-family:JetBrains Mono,monospace'>{verdict}</div>
                </div>
                <div style='text-align:right'>
                    <div style='font-size:.68rem;text-transform:uppercase;letter-spacing:.09em;color:#444;margin-bottom:.3rem'>Churn probability</div>
                    <div style='font-size:2.6rem;font-weight:600;color:#fff;font-family:JetBrains Mono,monospace;line-height:1'>{p*100:.1f}%</div>
                    <div style='font-size:.72rem;color:#333;margin-top:.2rem'>threshold · {THRESHOLD:.2f}</div>
                </div>
            </div>
            <div style='margin-top:1.2rem;background:#0d0d0d;border-radius:3px;height:3px'>
                <div style='width:{p*100:.1f}%;height:100%;background:{vc};border-radius:3px'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — BATCH UPLOAD
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div style='color:#555;font-size:0.83rem;margin-bottom:1.2rem'>Upload a CSV file · same column format as the original dataset</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df_up = pd.read_csv(uploaded)
        has_actual = "churn" in df_up.columns
        y_actual = df_up["churn"].values if has_actual else None

        with st.spinner("Processing..."):
            df_inp  = df_up.drop(columns=["churn"], errors="ignore")
            df_proc = preprocess(df_inp, art)
            proba, pred = predict(df_proc, art, THRESHOLD)

        df_res = df_up.copy()
        df_res["churn_probability"] = (proba*100).round(2)
        df_res["churn_prediction"]  = pred

        n_total=len(pred); n_churn=pred.sum()

        st.markdown("<div class='sec'>Summary</div>", unsafe_allow_html=True)
        m1,m2 = st.columns(2)
        m1.metric("Total",         f"{n_total:,}")
        m2.metric("Churn",         f"{n_churn:,}")

        st.markdown("<div class='sec'>Distribution</div>", unsafe_allow_html=True)
        fig, axes = plt.subplots(1,2,figsize=(10,4)); fig.patch.set_facecolor('#141414')

        # Donut
        ax = axes[0]
        ax.pie([n_total-n_churn,n_churn], colors=['#4ade80','#f87171'],
               startangle=90, wedgeprops=dict(width=0.42, edgecolor='#141414', linewidth=2))
        ax.text(0,0.05,f'{n_churn/n_total*100:.1f}%',ha='center',va='center',
                fontsize=13,fontweight='600',color='#fff',fontfamily='monospace')
        ax.text(0,-0.15,'churn rate',ha='center',fontsize=7.5,color='#555')
        ax.set_title('Churn Rate',fontsize=9,color='#666',pad=10,fontweight='400')

        # Histogram
        ax2 = axes[1]
        ax2.hist(proba[pred==0],bins=25,color='#4ade8022',edgecolor='#4ade80',linewidth=0.5,alpha=0.9,label='No churn')
        ax2.hist(proba[pred==1],bins=25,color='#f8717122',edgecolor='#f87171',linewidth=0.5,alpha=0.9,label='Churn')
        ax2.axvline(THRESHOLD,color='#ffffff44',linestyle='--',linewidth=1)
        ax2.set_xlabel('Probability',fontsize=8); ax2.grid(True,alpha=0.2)
        ax2.set_title('Probability Distribution',fontsize=9,color='#666',pad=10,fontweight='400')
        ax2.legend(fontsize=7,framealpha=0)

        plt.tight_layout(pad=2); st.pyplot(fig); plt.close()

        if has_actual:
            from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
            st.markdown("<div class='sec'>Evaluation</div>", unsafe_allow_html=True)
            e1,e2,e3,e4 = st.columns(4)
            e1.metric("Accuracy",  f"{accuracy_score(y_actual,pred):.4f}")
            e2.metric("F1-Score",  f"{f1_score(y_actual,pred):.4f}")
            e3.metric("Precision", f"{precision_score(y_actual,pred):.4f}")
            e4.metric("Recall",    f"{recall_score(y_actual,pred):.4f}")

        st.markdown("<div class='sec'>Results</div>", unsafe_allow_html=True)
        show = [c for c in ["customer_id","country","subscription_type","total_spent",
                             "satisfaction_score","churn_probability","churn_prediction"]
                if c in df_res.columns]
        st.dataframe(df_res[show], use_container_width=True, height=280)
        st.download_button("Download results (.csv)",
                           df_res.to_csv(index=False).encode("utf-8"),
                           "churn_predictions.csv","text/csv")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL INFO
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    t1, t2 = st.tabs(["Feature Importance", "Parameters"])

    with t1:
        imp = pd.Series(art["model"].feature_importances_,
                        index=art["feature_order"]).sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(9,7)); fig.patch.set_facecolor('#141414')
        norm = plt.Normalize(imp.min(), imp.max())
        colors = plt.cm.Blues(norm(imp.values[::-1])*0.6+0.3)
        ax.barh(imp.index[::-1], imp.values[::-1], color=colors, height=0.62,
                edgecolor='#141414', linewidth=0.5)
        for i,(v) in enumerate(imp.values[::-1]):
            ax.text(v+imp.max()*0.01, i, f'{v:.4f}', va='center', fontsize=7.5, color='#555')
        ax.set_xlabel('Importance Score',fontsize=8); ax.grid(True,alpha=0.15,axis='x')
        ax.set_title('Top 20 Features',fontsize=10,color='#888',pad=12,fontweight='400')
        ax.set_xlim(0, imp.max()*1.2)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t2:
        params = art["model"].get_params()
        st.dataframe(pd.DataFrame({"Parameter":list(params.keys()),
                                   "Value":[str(v) for v in params.values()]}),
                     use_container_width=True, hide_index=True)
        st.markdown("<div class='sec'>All features</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"#":range(1,len(art["feature_order"])+1),
                                   "Feature":art["feature_order"]}),
                     use_container_width=True, hide_index=True, height=300)
