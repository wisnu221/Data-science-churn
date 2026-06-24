import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

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

# ── Preprocessing pipeline ───────────────────────────────────────────────────
def preprocess(df_input, art):
    df = df_input.copy()

    # 1. Cleaning
    if "age" in df.columns:
        df["age"] = df["age"].clip(lower=0, upper=100)

    # 2. Imputasi
    iv = art["imputation_values"]
    if "coupon_code" in df.columns:
        df["coupon_code"] = df["coupon_code"].fillna("No Coupon")
    if "gender" in df.columns:
        df["gender"] = df["gender"].fillna(iv["mode_gender"])
    if "age" in df.columns:
        df["age"] = df["age"].fillna(iv["median_age"])
    if "total_spent" in df.columns:
        df["total_spent"] = df["total_spent"].fillna(iv["median_spent"])
    if "satisfaction_score" in df.columns:
        df["satisfaction_score"] = df["satisfaction_score"].fillna(iv["median_sat"])

    # 3. Feature engineering tanggal
    ref_date = art["reference_date"]
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"])
    df["customer_tenure_days"] = (ref_date - df["signup_date"]).dt.days
    df["recency_days"] = (ref_date - df["last_purchase_date"]).dt.days
    df["signup_year"] = df["signup_date"].dt.year
    df["signup_month"] = df["signup_date"].dt.month

    # 4. Derived features
    df["spend_per_visit"] = df["total_spent"] / (df["total_visits"] + 1)
    df["engagement_score"] = (df["email_open_rate"] * 0.5) + (df["email_click_rate"] * 0.5)
    df["refund_rate"] = df["refund_requested"] / (df["support_tickets"] + 1)

    # 5. Drop unused cols
    drop_cols = ["customer_id", "signup_date", "last_purchase_date", "city", "churn"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 6. Label encoding
    for col in art["cat_cols"]:
        if col in df.columns:
            le = art["label_encoders"][col]
            df[col] = df[col].astype(str).map(
                lambda x, le=le: x if x in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    # 7. Reorder
    df = df[art["feature_order"]]

    # 8. Scaling
    df_scaled = df.copy()
    df_scaled[art["scale_cols"]] = art["scaler"].transform(df[art["scale_cols"]])
    return df_scaled

def predict(df_proc, art, threshold=0.42):
    proba = art["model"].predict_proba(df_proc)[:, 1]
    pred  = (proba >= threshold).astype(int)
    risk  = pd.cut(proba, bins=[0, 0.3, 0.6, 1.0],
                   labels=["🟢 Low Risk", "🟡 Medium Risk", "🔴 High Risk"])
    return proba, pred, risk

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
st.sidebar.title("Customer Churn Predictor")
st.sidebar.markdown("**UAS Data Science**  \nSales & Marketing Dataset")
st.sidebar.divider()

THRESHOLD = st.sidebar.slider(
    "🎯 Threshold Prediksi", 0.1, 0.9, 0.42, 0.01,
    help="Nilai probabilitas minimum untuk diklasifikasikan sebagai Churn. "
         "Turunkan untuk meningkatkan Recall (lebih sensitif mendeteksi churn)."
)

st.sidebar.divider()
menu = st.sidebar.radio("📂 Navigasi", ["🏠 Home", "📝 Prediksi Manual", "📁 Upload CSV", "📊 Visualisasi Model"])

# ── Header ────────────────────────────────────────────────────────────────────
if not model_loaded:
    st.error("⚠️ File `churn_model_final.pkl` tidak ditemukan. Jalankan Week 5 notebook terlebih dahulu untuk membuat file model.")
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# HOME
# ═════════════════════════════════════════════════════════════════════════════
if menu == "🏠 Home":
    st.title("📊 Customer Churn Prediction App")
    st.markdown("Aplikasi prediksi customer churn berbasis Machine Learning menggunakan dataset Sales & Marketing.")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🤖 Model", "XGBoost")
    col2.metric("📦 Fitur", str(len(art["feature_order"])))
    col3.metric("🎯 Threshold", str(THRESHOLD))
    col4.metric("📅 Reference Date", str(art["reference_date"].date()))

    st.divider()
    st.subheader("📌 Cara Penggunaan")
    c1, c2 = st.columns(2)
    with c1:
        st.info("**📝 Prediksi Manual**\n\nIsi form data satu pelanggan baru, lalu klik tombol Prediksi untuk mendapat hasil instan.")
    with c2:
        st.info("**📁 Upload CSV**\n\nUpload file CSV berisi banyak pelanggan sekaligus. Aplikasi akan memproses dan menampilkan hasil prediksi batch beserta visualisasinya.")

    st.subheader("🏷️ Label Risiko")
    r1, r2, r3 = st.columns(3)
    r1.success("🟢 **Low Risk** (prob < 0.3)\n\nPelanggan kemungkinan besar tidak akan churn.")
    r2.warning("🟡 **Medium Risk** (0.3 – 0.6)\n\nPerlu dimonitor, pertimbangkan program retensi.")
    r3.error("🔴 **High Risk** (prob > 0.6)\n\nIntervensi segera diperlukan untuk mempertahankan pelanggan.")

# ═════════════════════════════════════════════════════════════════════════════
# PREDIKSI MANUAL
# ═════════════════════════════════════════════════════════════════════════════
elif menu == "📝 Prediksi Manual":
    st.title("📝 Prediksi Manual — Satu Pelanggan")
    st.markdown("Isi form di bawah dengan data pelanggan, lalu klik **Prediksi**.")
    st.divider()

    with st.form("form_prediksi"):
        st.subheader("👤 Data Demografi")
        c1, c2, c3 = st.columns(3)
        gender       = c1.selectbox("Gender", ["Male", "Female", "Other"])
        age          = c2.number_input("Age", 18, 80, 30)
        country      = c3.selectbox("Country", ["India", "USA", "Germany", "UK", "Bangladesh"])

        c4, c5 = st.columns(2)
        signup_date        = c4.date_input("Signup Date", value=pd.to_datetime("2022-01-01"))
        last_purchase_date = c5.date_input("Last Purchase Date", value=pd.to_datetime("2024-06-01"))

        st.subheader("📦 Informasi Produk & Transaksi")
        c6, c7, c8 = st.columns(3)
        subscription_type = c6.selectbox("Subscription Type", ["Monthly", "Annual"])
        payment_method    = c7.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"])
        acquisition_channel = c8.selectbox("Acquisition Channel", ["Organic", "Email", "Paid Ad", "Referral", "Social Media"])

        c9, c10, c11 = st.columns(3)
        device_type    = c9.selectbox("Device Type", ["Mobile", "Desktop", "Tablet"])
        is_premium     = c10.selectbox("Premium User", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
        discount_used  = c11.selectbox("Discount Used", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")

        c12, c13, c14 = st.columns(3)
        total_spent    = c12.number_input("Total Spent ($)", 0.0, 10000.0, 300.0)
        avg_order_value= c13.number_input("Avg Order Value ($)", 0.0, 5000.0, 50.0)
        lifetime_value = c14.number_input("Lifetime Value ($)", 0.0, 20000.0, 500.0)

        st.subheader("📈 Engagement & Aktivitas")
        c15, c16, c17 = st.columns(3)
        total_visits       = c15.number_input("Total Visits", 0, 500, 20)
        avg_session_time   = c16.number_input("Avg Session Time (min)", 0.0, 60.0, 5.0)
        pages_per_session  = c17.number_input("Pages per Session", 0.0, 30.0, 3.0)

        c18, c19 = st.columns(2)
        email_open_rate  = c18.slider("Email Open Rate", 0.0, 1.0, 0.3)
        email_click_rate = c19.slider("Email Click Rate", 0.0, 1.0, 0.1)

        st.subheader("🎧 Layanan Pelanggan")
        c20, c21, c22 = st.columns(3)
        support_tickets    = c20.number_input("Support Tickets", 0, 20, 1)
        refund_requested   = c21.selectbox("Refund Requested", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
        delivery_delay_days= c22.number_input("Delivery Delay (days)", 0, 30, 0)

        c23, c24, c25 = st.columns(3)
        satisfaction_score        = c23.slider("Satisfaction Score (1-5)", 1.0, 5.0, 3.0)
        nps_score                 = c24.slider("NPS Score (0-10)", 0, 10, 5)
        last_3_month_purchase_freq= c25.number_input("Purchase Freq (3 bulan)", 0, 30, 2)

        marketing_spend = st.number_input("Marketing Spend per User ($)", 0.0, 100.0, 10.0)

        submitted = st.form_submit_button("🔍 Prediksi Churn", use_container_width=True, type="primary")

    if submitted:
        input_df = pd.DataFrame([{
            "customer_id": 0, "gender": gender, "age": age, "country": country,
            "city": "Unknown", "signup_date": str(signup_date),
            "last_purchase_date": str(last_purchase_date),
            "acquisition_channel": acquisition_channel, "device_type": device_type,
            "subscription_type": subscription_type, "is_premium_user": is_premium,
            "total_visits": total_visits, "avg_session_time": avg_session_time,
            "pages_per_session": pages_per_session, "email_open_rate": email_open_rate,
            "email_click_rate": email_click_rate, "total_spent": total_spent,
            "avg_order_value": avg_order_value, "discount_used": discount_used,
            "coupon_code": "No Coupon", "support_tickets": support_tickets,
            "refund_requested": refund_requested, "delivery_delay_days": delivery_delay_days,
            "payment_method": payment_method, "satisfaction_score": satisfaction_score,
            "nps_score": nps_score, "marketing_spend_per_user": marketing_spend,
            "lifetime_value": lifetime_value,
            "last_3_month_purchase_freq": last_3_month_purchase_freq
        }])

        df_proc = preprocess(input_df, art)
        proba, pred, risk = predict(df_proc, art, THRESHOLD)

        st.divider()
        st.subheader("📊 Hasil Prediksi")

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Probabilitas Churn", f"{proba[0]*100:.2f}%")
        rc2.metric("Prediksi", "CHURN" if pred[0] == 1 else "TIDAK CHURN")
        rc3.metric("Risk Level", str(risk[0]))

        if pred[0] == 1:
            st.error(f"⚠️ Pelanggan ini diprediksi akan **CHURN** dengan probabilitas **{proba[0]*100:.2f}%**. Segera lakukan tindakan retensi!")
        else:
            st.success(f"✅ Pelanggan ini diprediksi **TIDAK CHURN** (probabilitas churn: {proba[0]*100:.2f}%). Pertahankan kualitas layanan!")

        # Gauge chart probabilitas
        fig, ax = plt.subplots(figsize=(6, 1.5))
        color = "#e74c3c" if proba[0] > 0.6 else ("#f39c12" if proba[0] > 0.3 else "#2ecc71")
        ax.barh(["Churn Probability"], [proba[0]], color=color, height=0.4)
        ax.barh(["Churn Probability"], [1 - proba[0]], left=[proba[0]], color="#ecf0f1", height=0.4)
        ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.5, label=f"Threshold ({THRESHOLD})")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probabilitas")
        ax.legend(fontsize=9)
        ax.set_title(f"Probabilitas Churn: {proba[0]*100:.2f}%", fontweight="bold")
        st.pyplot(fig)
        plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# UPLOAD CSV
# ═════════════════════════════════════════════════════════════════════════════
elif menu == "📁 Upload CSV":
    st.title("📁 Prediksi Batch — Upload CSV")
    st.markdown("Upload file CSV berisi data banyak pelanggan. Format kolom harus sama dengan dataset asli.")
    st.divider()

    uploaded = st.file_uploader("📂 Pilih file CSV", type=["csv"])

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.success(f"✅ File berhasil dimuat: **{df_upload.shape[0]:,} baris**, **{df_upload.shape[1]} kolom**")

        # Simpan label aktual jika ada
        has_actual = "churn" in df_upload.columns
        if has_actual:
            y_actual = df_upload["churn"].values

        with st.spinner("🔄 Memproses prediksi..."):
            df_input = df_upload.drop(columns=["churn"], errors="ignore")
            df_proc  = preprocess(df_input, art)
            proba, pred, risk = predict(df_proc, art, THRESHOLD)

        df_result = df_upload.copy()
        df_result["churn_probability"] = (proba * 100).round(2)
        df_result["churn_prediction"]  = pred
        df_result["risk_label"]        = risk.astype(str)

        st.divider()
        st.subheader("📊 Ringkasan Hasil")

        n_churn  = pred.sum()
        n_total  = len(pred)
        n_high   = (risk == "🔴 High Risk").sum()
        n_medium = (risk == "🟡 Medium Risk").sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Pelanggan", f"{n_total:,}")
        m2.metric("Prediksi Churn", f"{n_churn:,}", f"{n_churn/n_total*100:.1f}%")
        m3.metric("🔴 High Risk", f"{n_high:,}")
        m4.metric("🟡 Medium Risk", f"{n_medium:,}")

        # Visualisasi distribusi prediksi
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Pie chart churn
        axes[0].pie([n_total - n_churn, n_churn],
                    labels=["Tidak Churn", "Churn"],
                    colors=["#2ecc71", "#e74c3c"],
                    autopct="%1.1f%%", startangle=90,
                    wedgeprops={"edgecolor": "white"})
        axes[0].set_title("Distribusi Prediksi Churn", fontweight="bold")

        # Distribusi probabilitas
        axes[1].hist(proba, bins=30, color="#3498db", edgecolor="white", alpha=0.85)
        axes[1].axvline(THRESHOLD, color="red", linestyle="--", linewidth=2, label=f"Threshold ({THRESHOLD})")
        axes[1].set_title("Distribusi Probabilitas Churn", fontweight="bold")
        axes[1].set_xlabel("Probabilitas")
        axes[1].legend()

        # Bar chart risk label
        risk_counts = pd.Series(risk.astype(str)).value_counts()
        colors_risk = {"🟢 Low Risk": "#2ecc71", "🟡 Medium Risk": "#f39c12", "🔴 High Risk": "#e74c3c"}
        bars = axes[2].bar(risk_counts.index, risk_counts.values,
                           color=[colors_risk.get(r, "#95a5a6") for r in risk_counts.index],
                           edgecolor="white")
        for bar, val in zip(bars, risk_counts.values):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                         str(val), ha="center", fontweight="bold")
        axes[2].set_title("Distribusi Risk Label", fontweight="bold")
        axes[2].set_ylabel("Jumlah Pelanggan")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Tabel hasil
        st.divider()
        st.subheader("📋 Tabel Hasil Prediksi")
        show_cols = ["customer_id", "country", "subscription_type", "total_spent",
                     "satisfaction_score", "churn_probability", "churn_prediction", "risk_label"]
        show_cols = [c for c in show_cols if c in df_result.columns]
        st.dataframe(df_result[show_cols], use_container_width=True)

        # Download
        csv_out = df_result.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Hasil Prediksi (.csv)",
            data=csv_out,
            file_name="hasil_prediksi_churn.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Akurasi jika ada label aktual
        if has_actual:
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(y_actual, pred)
            f1  = f1_score(y_actual, pred)
            st.divider()
            st.subheader("✅ Evaluasi (Label Aktual Tersedia)")
            e1, e2 = st.columns(2)
            e1.metric("Accuracy", f"{acc:.4f}")
            e2.metric("F1-Score", f"{f1:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# VISUALISASI MODEL
# ═════════════════════════════════════════════════════════════════════════════
elif menu == "📊 Visualisasi Model":
    st.title("📊 Visualisasi Model")
    st.divider()

    tab1, tab2 = st.tabs(["🔑 Feature Importance", "ℹ️ Informasi Model"])

    with tab1:
        st.subheader("🔑 Top 20 Feature Importance")
        importances = pd.Series(
            art["model"].feature_importances_,
            index=art["feature_order"]
        ).sort_values(ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = sns.color_palette("RdYlGn_r", 20)
        bars = ax.barh(importances.index[::-1], importances.values[::-1], color=colors[::-1])
        for bar, val in zip(bars, importances.values[::-1]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=9)
        ax.set_xlabel("Importance Score")
        ax.set_title("Top 20 Feature Importance — XGBoost Final Model", fontweight="bold", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.dataframe(
            pd.DataFrame({"Feature": importances.index, "Importance": importances.values.round(4)}),
            use_container_width=True
        )

    with tab2:
        st.subheader("ℹ️ Informasi Model")
        params = art["model"].get_params()
        info_df = pd.DataFrame({"Parameter": list(params.keys()), "Nilai": list(params.values())})
        st.dataframe(info_df, use_container_width=True)

        st.subheader("📋 Fitur yang Digunakan")
        feat_df = pd.DataFrame({"No": range(1, len(art["feature_order"])+1),
                                 "Nama Fitur": art["feature_order"]})
        st.dataframe(feat_df, use_container_width=True)
