import streamlit as st
import base64

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="SiagaGempa Jabar | Beranda",
    page_icon="🌍",
    layout="wide"
)

# --- FUNGSI UNTUK GAMBAR LATAR HERO SECTION ---
# Fungsi ini akan membaca file gambar dan mengubahnya menjadi format yang bisa dibaca CSS
@st.cache_data
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-attachment: fixed;
    }}
    /* Membuat kontainer di Streamlit sedikit transparan dan blur */
    .st-emotion-cache-16txtl3 {{
        background-color: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(5px);
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.05);
        padding: 1.5rem;
    }}
    /* Menghapus background header agar menyatu */
    .stHeader {{
        background-color: transparent !important;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# --- APLIKASI GAMBAR LATAR ---
# Pastikan Anda memiliki file 'background.png' di folder yang sama
try:
    set_background('background.png')
except FileNotFoundError:
    st.warning("File 'background.png' tidak ditemukan. Latar belakang tidak akan ditampilkan. Silakan unduh gambar dan letakkan di folder proyek Anda.")


# --- HERO SECTION ---
st.title("Selamat Datang di SiagaGempa Jabar 🌍")
st.header("Platform Prediksi Risiko Fisik dan Sosial Gempa Bumi di Jawa Barat")

st.markdown("""
Aplikasi ini dirancang untuk memberikan dua layanan utama:
- **Prediksi Risiko**: Menganalisis tingkat risiko suatu lokasi terhadap dampak gempa bumi berdasarkan data demografi, geografis, dan historis.
- **Informasi & Edukasi**: Menyediakan panduan praktis mengenai apa yang harus dilakukan sebelum, saat, dan sesudah gempa bumi terjadi.

**Silakan pilih halaman yang ingin Anda tuju dari menu di sebelah kiri.**
""")

st.info("Untuk memulai analisis, pilih halaman **'🛠️ Prediksi Risiko'** pada menu di samping.", icon="👈")

st.markdown("---")
st.write("Dibuat dengan ❤️ untuk Jawa Barat yang lebih tangguh.")
st.write("Update Terakhir: 4 Juli 2025")