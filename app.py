import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
import os
import base64

# 🔥 TAMBAHAN PENTING UNTUK HEIC
from pillow_heif import register_heif_opener
register_heif_opener()

st.set_page_config(
    page_title="🌿 Klasifikasi Rimpang Jamu Madura",
    layout="wide"
)

# =====================
# CUSTOM CSS
# =====================
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background-color: #0e1117 !important;
}

div.row-widget.stRadio > div {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# =====================
# LOAD MODEL (CACHED)
# =====================
@st.cache_resource
def load_models():
    feature_extractor = tf.keras.models.load_model("vgg19_feature_extractor_fix.h5")
    rf_model = joblib.load("model_random_forest_terbaik.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return feature_extractor, rf_model, label_encoder

feature_extractor, rf_model, lbl = load_models()

# =====================
# SIDEBAR LOGO + NAMA UNIVERSITAS
# =====================
with open("utm.png", "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode()

st.sidebar.markdown(
    f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 12px 6px;
    ">
        <img src="data:image/png;base64,{logo_base64}"
             style="width:48px; height:auto;" />
        <div style="line-height:1.2;">
            <div style="font-size:14px; font-weight:bold; color:white;">
                Universitas
            </div>
            <div style="font-size:14px; font-weight:bold; color:white;">
                Trunojoyo Madura
            </div>
        </div>
    </div>
    <hr style="border:0.5px solid #333;">
    """,
    unsafe_allow_html=True
)

# =====================
# SIDEBAR NAVIGASI
# =====================
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

if st.sidebar.button(
    "🏠 Home",
    use_container_width=True,
    type="primary" if st.session_state.current_page == "Home" else "secondary"
):
    st.session_state.current_page = "Home"
    st.rerun()

if st.sidebar.button(
    "📖 Deskripsi",
    use_container_width=True,
    type="primary" if st.session_state.current_page == "Deskripsi" else "secondary"
):
    st.session_state.current_page = "Deskripsi"
    st.rerun()

if st.sidebar.button(
    "🖼️ Klasifikasi",
    use_container_width=True,
    type="primary" if st.session_state.current_page == "Klasifikasi" else "secondary"
):
    st.session_state.current_page = "Klasifikasi"
    st.rerun()

# =====================
# KONTEN HALAMAN
# =====================
page = st.session_state.current_page

# ---------------------
# HOME
# ---------------------
if page == "Home":
    st.title("🌿 Aplikasi Klasifikasi Rimpang Jamu Madura")
    st.markdown("""
    Aplikasi ini dibuat berdasarkan skripsi:  
    **PERBANDINGAN TRANSFER LEARNING VGG-19 DENGAN FULLY CONNECTED LAYER DEFAULT DAN  
    VGG-19 DENGAN RANDOM FOREST CLASSIFIER UNTUK KLASIFIKASI CITRA RIMPANG JAMU MADURA**

    Aplikasi ini mengklasifikasikan lima jenis rimpang jamu Madura:
    - Temulawak  
    - Kencur  
    - Jahe  
    - Lengkuas  
    - Kunyit
    """)

# ---------------------
# DESKRIPSI (DENGAN GAMBAR OTOMATIS)
# ---------------------
elif page == "Deskripsi":
    st.title("📖 Deskripsi Rimpang Jamu Madura")
    st.markdown(
        "_Berikut adalah deskripsi visual dan karakteristik dari lima jenis rimpang yang umum digunakan dalam jamu tradisional Madura._"
    )
    st.markdown("---")

    classes = {
        "temulawak": {
            "latin": "Curcuma xanthorrhiza",
            "desc": [
                "Bentuk rimpang menggembung dengan ruas besar",
                "Kulit berwarna cokelat tua",
                "Bagian dalam oranye terang kekuningan",
                "Memiliki aroma khas dan rasa pahit"
            ]
        },
        "kencur": {
            "latin": "Kaempferia galanga",
            "desc": [
                "Ukuran kecil dan berbentuk pipih",
                "Kulit cokelat tua kehitaman dan kerut",
                "Daging berwarna putih berbintik",
                "Aroma sangat tajam"
            ]
        },
        "jahe": {
            "latin": "Zingiber officinale",
            "desc": [
                "Berbentuk seperti jari dengan ruas jelas",
                "Kulit cokelat muda hingga tua",
                "Daging berwarna kuning muda",
                "Aroma pedas dan hangat"
            ]
        },
        "lengkuas": {
            "latin": "Alpinia galanga",
            "desc": [
                "Ukuran rimpang besar dan keras",
                "Kulit cokelat kemerahan dan kasar",
                "Daging berwarna putih pucat",
                "Aroma tajam dan pedas"
            ]
        },
        "kunyit": {
            "latin": "Curcuma longa",
            "desc": [
                "Bentuk lurus bercabang teratur",
                "Kulit kuning kecokelatan",
                "Daging berwarna oranye cerah",
                "Aroma khas kunyit"
            ]
        }
    }

    extensions = ["HEIC", "heic", "jpg", "jpeg", "png"]

    for name, info in classes.items():
        col1, col2, col3 = st.columns([0.05, 0.4, 0.55])

        img_path = None
        for ext in extensions:
            path = f"{name}.{ext}"
            if os.path.exists(path):
                img_path = path
                break

        with col2:
            if img_path:
                img = Image.open(img_path).convert("RGB")
                # Resize image to smaller dimensions
                img.thumbnail((250, 250), Image.LANCZOS)
                st.image(img, caption=name.capitalize(), use_container_width=False)
            else:
                st.error(f"Gambar {name} tidak ditemukan")

        with col3:
            st.subheader(f"{name.capitalize()} (*{info['latin']}*)")
            for d in info["desc"]:
                st.markdown(f"- {d}")

        st.markdown("---")

# ---------------------
# KLASIFIKASI
# ---------------------
elif page == "Klasifikasi":
    st.title("🌿 Klasifikasi Rimpang Jamu")

    uploaded_file = st.file_uploader(
        "Upload gambar rimpang (jpg, jpeg, png, heic)",
        type=["jpg", "jpeg", "png", "heic", "HEIC"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_resized = img.resize((224, 224))
        st.image(img_resized, caption="Gambar yang diupload", width=300)

        with st.spinner("Sedang menganalisis gambar..."):
            img_array = np.array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            feat = feature_extractor.predict(img_array)
            if feat.ndim == 4:
                # Global Average Pooling: (1, H, W, C) -> (1, C)
                feat_flat = feat.mean(axis=(1, 2))
            else:
                feat_flat = feat.reshape(1, -1)
            pred_idx = rf_model.predict(feat_flat)[0]
            pred_label = lbl.inverse_transform([pred_idx])[0]

        st.success(f"🌱 **Prediksi: {pred_label}**")