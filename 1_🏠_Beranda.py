import streamlit as st
import pandas as pd
import numpy as np
import joblib # Untuk memuat model dan encoder
from sklearn.preprocessing import LabelEncoder
import os
import folium
from streamlit_folium import st_folium # Tetap perlu ini untuk menampilkan peta Folium

# --- Fungsi Haversine untuk Menghitung Jarak (Pengganti Geopandas Distance) ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371 # Radius bumi dalam kilometer
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

# --- Konfigurasi Awal Streamlit ---
st.set_page_config(
    page_title="Prediksi Kerentanan Gempa Jawa Barat",
    page_icon="🌍",
    layout="wide"
)

# --- Lokasi File Data & Model ---
DIR_DATA_FILES = "data" # Folder 'data' di root repo
MODEL_FILENAME = 'random_forest_model.pkl'
LABEL_ENCODER_FILENAME = 'label_encoder.pkl'
FEATURE_COLUMNS_FILENAME = 'feature_columns_model.pkl'

# Path ke file CSV yang sudah diproses (BUKAN GPKG LAGI!)
FINAL_GRID_DATA_PATH = os.path.join(DIR_DATA_FILES, 'final_grid_data_processed.csv')
GD_GEMPA_JABAR_PATH = os.path.join(DIR_DATA_FILES, 'gdf_gempa_jabar_processed.csv')
GD_POI_JABAR_PATH = os.path.join(DIR_DATA_FILES, 'gdf_poi_jabar_processed.csv')
GD_DEMOGRAFI_JABAR_CLEAN_PATH = os.path.join(DIR_DATA_FILES, 'gdf_demografi_jabar_clean_processed.csv')

# --- Fungsi Caching Data & Model ---
@st.cache_resource
def load_all_resources_no_geopandas():
    print("Memuat semua sumber daya (model, data, encoder) TANPA GEOPANDAS...")
    try:
        model = joblib.load(MODEL_FILENAME)
        label_encoder = joblib.load(LABEL_ENCODER_FILENAME)
        feature_cols_model = joblib.load(FEATURE_COLUMNS_FILENAME)

        # Muat DataFrames dari CSV (BUKAN GEOPANDAS LAGI)
        final_grid_data = pd.read_csv(FINAL_GRID_DATA_PATH)
        df_gempa_jabar = pd.read_csv(GD_GEMPA_JABAR_PATH)
        df_poi_jabar = pd.read_csv(GD_POI_JABAR_PATH)
        df_demografi_jabar_clean = pd.read_csv(GD_DEMOGRAFI_JABAR_CLEAN_PATH)
        
        return model, label_encoder, feature_cols_model, final_grid_data, \
               df_gempa_jabar, df_poi_jabar, df_demografi_jabar_clean
    except FileNotFoundError as e:
        st.error(f"ERROR: File sumber daya tidak ditemukan. Pastikan semua file ada di lokasi yang benar. {e}")
        st.stop()
    except Exception as e:
        st.error(f"ERROR: Gagal memuat sumber daya. {e}")
        st.stop()

model, label_encoder, feature_cols_model, final_grid_data, \
df_gempa_jabar, df_poi_jabar, df_demografi_jabar_clean = load_all_resources_no_geopandas()

# --- PARAMETER UNTUK FEATURE ENGINEERING (HARUS KONSISTEN DENGAN SAAT TRAINING!) ---
BUFFER_GEMPA_KM = 5    # Buffer gempa saat agregasi


# --- HELPER FUNCTION: Categorize POI (Pastikan sama persis dengan saat training!) ---
def categorize_poi(poi_type):
    poi_type_lower = str(poi_type).lower()
    if 'hospital' in poi_type_lower or 'clinic' in poi_type_lower:
        return 'Fasilitas Kesehatan'
    elif 'school' in poi_type_lower:
        return 'Sekolah'
    elif 'police' in poi_type_lower or 'townhall' in poi_type_lower:
        return 'Pemerintahan/Publik'
    elif 'library' in poi_type_lower or 'place_of_worship' in poi_type_lower or 'restaurant' in poi_type_lower:
        return 'Fasilitas Sosial/Publik Lain'
    else:
        return 'Bangunan Biasa' 

# --- FUNGSI UTAMA UNTUK FEATURE ENGINEERING & PREDIKSI SATU TITIK (TANPA GEOPANDAS) ---
def predict_vulnerability_for_point(
    user_lat, user_lon, 
    user_jumlah_kk, user_rasio_lp, 
    user_ada_rs, user_ada_sekolah, user_ada_pemerintahan, user_ada_bangunan_biasa, user_ada_fasos_lain
):
    """
    Mengambil input dari user, melakukan feature engineering (tanpa geopandas), dan memprediksi
    tingkat kerentanan/dampak untuk satu titik lokasi.
    """
    
    # 1. Dapatkan Fitur Demografi & Kemiskinan dari Kelurahan Terdekat ---
    # Mencari kelurahan terdekat berdasarkan centroid
    df_demografi_jabar_clean['distance_to_user'] = haversine_distance(
        user_lat, user_lon, df_demografi_jabar_clean['centroid_lat'], df_demografi_jabar_clean['centroid_lon']
    )
    closest_kelurahan = df_demografi_jabar_clean.loc[df_demografi_jabar_clean['distance_to_user'].idxmin()]
    
    demog_features_dict = {
        'jumlah_penduduk': closest_kelurahan['jumlah_penduduk'],
        'pria': closest_kelurahan['pria'],
        'wanita': closest_kelurahan['wanita'],
        'jumlah_produktif': closest_kelurahan['jumlah_produktif'],
        'jumlah_non_produktif': closest_kelurahan['jumlah_non_produktif'],
        'rasio_lp': closest_kelurahan['rasio_lp'],
        'rasio_produktif_nonproduktif': closest_kelurahan['rasio_produktif_nonproduktif'],
        'kepadatan_penduduk_kelurahan': closest_kelurahan['kepadatan_penduduk_kelurahan'],
        'jumlah_penduduk_miskin_kelurahan_estimasi': closest_kelurahan['jumlah_penduduk_miskin_kelurahan_estimasi']
    }
    

    # 2. Dapatkan Fitur Gempa dari Area Sekitar (Buffer)
    # Filter gempa dalam radius BUFFER_GEMPA_KM
    nearby_gempa = df_gempa_jabar[
        haversine_distance(user_lat, user_lon, df_gempa_jabar['latitude'], df_gempa_jabar['longitude']) <= BUFFER_GEMPA_KM
    ]

    if nearby_gempa.empty:
        gempa_features_dict = {
            'count_gempa': 0, 'max_mag': 0.0, 'avg_depth': 0.0
        }
    else:
        gempa_features_dict = {
            'count_gempa': len(nearby_gempa),
            'max_mag': nearby_gempa['mag'].max(),
            'avg_depth': nearby_gempa['depth'].mean()
        }


    # 3. Dapatkan Fitur POI dari Area Sekitar (Tanpa Buffer, langsung di titik)
    # Filter POI dalam radius kecil (misal 0.1km atau 100m)
    POI_SEARCH_RADIUS_KM = 0.1 # Radius 100 meter
    nearby_poi = df_poi_jabar[
        haversine_distance(user_lat, user_lon, df_poi_jabar['latitude'], df_poi_jabar['longitude']) <= POI_SEARCH_RADIUS_KM
    ]
    
    poi_counts = nearby_poi['category'].value_counts().to_dict()
    
    poi_features_dict = {}
    for category in ['Fasilitas Kesehatan', 'Sekolah', 'Pemerintahan/Publik', 'Fasilitas Sosial/Publik Lain', 'Bangunan Biasa']:
        col_name = f'count_poi_{category.replace(" ", "_").lower()}'
        poi_features_dict[col_name] = poi_counts.get(category, 0)


    # 4. Menggabungkan Input User Manual dengan Fitur yang Diperoleh
    all_features_dict = {
        **demog_features_dict,
        **gempa_features_dict,
        **poi_features_dict,
        # Fitur dari input manual user.
        'user_input_jumlah_kk': user_jumlah_kk, 
        'user_input_rasio_lp': user_rasio_lp, 
        'user_ada_rs': (1 if user_ada_rs == 'Ya' else 0),
        'user_ada_sekolah': (1 if user_ada_sekolah == 'Ya' else 0),
        'user_ada_pemerintahan': (1 if user_ada_pemerintahan == 'Ya' else 0),
        'user_ada_bangunan_biasa': (1 if user_ada_bangunan_biasa == 'Ya' else 0),
        'user_ada_fasos_lain': (1 if user_ada_fasos_lain == 'Ya' else 0),
    }
    
    # 5. Buat Feature Vector (X_new) dengan Urutan Kolom yang Konsisten
    X_new_series = pd.Series(all_features_dict)
    X_new = X_new_series.reindex(feature_cols_model, fill_value=0.0)

    # Ubah menjadi 2D array (1 baris) untuk model.predict()
    X_new_predict = X_new.values.reshape(1, -1)
    
    return X_new_predict

# --- FUNGSI PREDIKSI FINAL ---
def predict_vulnerability(
    user_lat, user_lon, 
    user_jumlah_kk, user_rasio_lp, 
    user_ada_rs, user_ada_sekolah, user_ada_pemerintahan, user_ada_bangunan_biasa, user_ada_fasos_lain
):
    """
    Melakukan prediksi kerentanan/dampak dan mengkonversi output ke label string.
    """
    X_new_predict = predict_vulnerability_for_point(
        user_lat, user_lon, 
        user_jumlah_kk, user_rasio_lp, 
        user_ada_rs, user_ada_sekolah, user_ada_pemerintahan, user_ada_bangunan_biasa, user_ada_fasos_lain
    )
    
    prediction_encoded = model.predict(X_new_predict)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0] # Konversi kembali ke string

    return prediction_label


# --- Judul dan Deskripsi Aplikasi ---
st.title("🌍 Prediksi Tingkat Kerentanan dan Dampak Potensial Gempa Bumi")
st.markdown("""
Aplikasi ini membantu Anda memprediksi tingkat kerentanan dan dampak potensial suatu lokasi terhadap gempa bumi, 
berdasarkan data geografis, demografi, dan fasilitas umum di sekitarnya.
Masukkan detail lokasi Anda di bawah ini.
""")

st.markdown("---") # Garis pemisah

# --- Input Pengguna ---
col1, col2 = st.columns([1, 1]) # Dua kolom, proporsi 1:1

with col1:
    st.header("📍 Detail Lokasi")
    st.write("Masukkan koordinat dan data demografi lokasi yang ingin Anda cek.")

    # Input Latitude
    latitude = st.number_input(
        "**Latitude (Garis Lintang)**",
        min_value=-10.0, # Batas untuk Indonesia
        max_value=10.0,  # Batas untuk Indonesia
        value=-6.9059,   # Contoh: Bandung
        format="%.6f",
        help="Garis lintang lokasi yang ingin diprediksi. Contoh: -6.9059 (Bandung)"
    )

    # Input Longitude
    longitude = st.number_input(
        "**Longitude (Garis Bujur)**",
        min_value=95.0,  # Batas untuk Indonesia
        max_value=141.0, # Batas untuk Indonesia
        value=107.6177,  # Contoh: Bandung
        format="%.6f",
        help="Garis bujur lokasi yang ingin diprediksi. Contoh: 107.6177 (Bandung)"
    )
    
    st.markdown("---")
    st.header("👨‍👩‍👧‍👦 Data Demografi di Lokasi (Estimasi)")
    st.write("Isi data berikut untuk lokasi spesifik Anda. Ini akan memperkaya prediksi.")

    # Input Jumlah KK (Ini adalah input langsung dari user, bukan dari demografi kelurahan)
    jumlah_kk_input = st.number_input(
        "**Jumlah Kepala Keluarga (KK) di lokasi**",
        min_value=0,
        value=50, # Default lebih kecil, karena ini estimasi di lokasi kecil
        step=5,
        help="Estimasi jumlah Kepala Keluarga di area yang Anda masukkan."
    )

    # Input Rasio Laki-laki / Perempuan
    rasio_laki_perempuan_input = st.slider(
        "**Rasio Laki-laki : Perempuan (L/P)**",
        min_value=0.5, # Contoh: 1 laki-laki untuk 2 perempuan
        max_value=2.0, # Contoh: 2 laki-laki untuk 1 perempuan
        value=1.0,     # Contoh: 1 laki-laki untuk 1 perempuan
        step=0.05,
        help="Rasio perbandingan jumlah laki-laki dan perempuan di lokasi Anda. Contoh: 1.0 untuk seimbang, 0.8 jika perempuan lebih banyak."
    )

with col2:
    st.header("🏥 Keberadaan Fasilitas Penting")
    st.write("Indikasikan apakah ada fasilitas penting berikut di sekitar lokasi Anda (dalam radius ~100m).")

    # Input Keberadaan POI
    ada_rumah_sakit_terdekat = st.radio(
        "**Ada Rumah Sakit/Klinik Terdekat?**",
        ('Tidak', 'Ya'), horizontal=True, key='rs_radio'
    )
    ada_sekolah_terdekat = st.radio(
        "**Ada Sekolah Terdekat?**",
        ('Tidak', 'Ya'), horizontal=True, key='sekolah_radio'
    )
    ada_kantor_pemerintahan_terdekat = st.radio(
        "**Ada Kantor Pemerintahan/Publik Terdekat?**",
        ('Tidak', 'Ya'), horizontal=True, key='pemerintahan_radio'
    )
    ada_fasos_lain_terdekat = st.radio(
        "**Ada Fasilitas Sosial/Publik Lain (Perpustakaan, Tempat Ibadah, Restoran) Terdekat?**",
        ('Tidak', 'Ya'), horizontal=True, key='fasos_radio'
    )
    ada_bangunan_biasa_terdekat = st.radio(
        "**Ada Bangunan Biasa Lainnya (Perumahan, Toko, Kantor Non-pemerintah) Terdekat?**",
        ('Tidak', 'Ya'), horizontal=True, key='bangunan_biasa_radio'
    )
    
    st.markdown("---")
    st.write("Tekan tombol di bawah untuk melihat hasil prediksi.")
    
    # Tombol Prediksi
    if st.button("✨ Prediksi Tingkat Kerentanan/Dampak", use_container_width=True, type="primary"):
        with st.spinner('Menganalisis data dan memprediksi...'):
            # Panggil fungsi prediksi
            predicted_level = predict_vulnerability(
                latitude, longitude, 
                jumlah_kk_input, rasio_laki_perempuan_input, 
                ada_rumah_sakit_terdekat, ada_sekolah_terdekat, ada_kantor_pemerintahan_terdekat, 
                ada_bangunan_biasa_terdekat, ada_fasos_lain_terdekat
            )
            
            # --- Tampilan Hasil Prediksi ---
            st.markdown("---")
            st.header("📊 Hasil Prediksi")
            
            if predicted_level == 'Tinggi':
                st.error(f"⚠️ **Tingkat Kerentanan/Dampak Potensial: {predicted_level}**")
                st.write("Lokasi ini memiliki indikasi kerentanan tinggi terhadap dampak gempa. Disarankan untuk mengambil langkah-langkah mitigasi dan persiapan. Perlu perhatian khusus!")
            elif predicted_level == 'Sedang':
                st.warning(f"🟧 **Tingkat Kerentanan/Dampak Potensial: {predicted_level}**")
                st.write("Lokasi ini memiliki indikasi kerentanan sedang. Tetap waspada dan pertimbangkan langkah antisipasi.")
            else: # Rendah
                st.success(f"✅ **Tingkat Kerentanan/Dampak Potensial: {predicted_level}**")
                st.write("Lokasi ini menunjukkan tingkat kerentanan yang relatif rendah. Namun, kewaspadaan tetap penting.")
            
            st.markdown("---")
            st.header("🗺️ Peta Lokasi Anda & Data Kontekstual")

            # --- Visualisasi Peta ---
            # Cari Kab/Kota dan Kelurahan dari titik yang diinput (opsional, untuk tampilan di peta)
            # Tanpa geopandas, ini harus dilakukan secara manual (cari terdekat dari CSV)
            
            # Mendapatkan info kelurahan (untuk display nama)
            df_demografi_jabar_clean['distance_to_user'] = haversine_distance(
                latitude, longitude, df_demografi_jabar_clean['centroid_lat'], df_demografi_jabar_clean['centroid_lon']
            )
            closest_kelurahan_for_display = df_demografi_jabar_clean.loc[df_demografi_jabar_clean['distance_to_user'].idxmin()]
            
            kab_name = closest_kelurahan_for_display['nama_kab'] if 'nama_kab' in closest_kelurahan_for_display else "Tidak Ditemukan"
            kec_name = closest_kelurahan_for_display['nama_kec'] if 'nama_kec' in closest_kelurahan_for_display else "Tidak Ditemukan"
            kel_name = closest_kelurahan_for_display['nama_kel'] if 'nama_kel' in closest_kelurahan_for_display else "Tidak Ditemukan"

            st.write(f"Lokasi di: **Kabupaten/Kota {kab_name}**, Kecamatan {kec_name}, Kelurahan {kel_name}")

            m = folium.Map(location=[latitude, longitude], zoom_start=12)
            
            # Marker lokasi input user
            folium.Marker(
                [latitude, longitude],
                tooltip="Lokasi Anda",
                popup=f"Lat: {latitude}, Lon: {longitude}<br>Prediksi: {predicted_level}",
                icon=folium.Icon(color="blue", icon="home")
            ).add_to(m)

            # Tambahkan marker gempa historis di sekitar lokasi user (opsional)
            # Filter gempa dalam radius visualisasi (misal 20km)
            nearby_gempa_map = df_gempa_jabar[
                haversine_distance(latitude, longitude, df_gempa_jabar['latitude'], df_gempa_jabar['longitude']) < 20
            ].copy() 

            for idx, row in nearby_gempa_map.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=row['mag'] * 1.5, 
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.6,
                    popup=f"Gempa M{row['mag']:.1f} ({row['depth']:.1f} km)"
                ).add_to(m)
            
            # Tambahkan marker POI di sekitar lokasi user (opsional)
            # Filter POI dalam radius visualisasi (misal 1km)
            nearby_poi_map = df_poi_jabar[
                haversine_distance(latitude, longitude, df_poi_jabar['latitude'], df_poi_jabar['longitude']) < 1
            ].copy() 

            for idx, row in nearby_poi_map.iterrows():
                color_icon = 'green'
                if row['category'] == 'Fasilitas Kesehatan': color_icon = 'darkred'
                elif row['category'] == 'Sekolah': color_icon = 'blue'
                elif row['category'] == 'Pemerintahan/Publik': color_icon = 'darkblue'
                elif row['category'] == 'Infrastruktur Kritis': color_icon = 'black'
                
                folium.Marker(
                    location=[row['latitude'], row['longitude']], # POI CSV tidak punya geometry, pakai lat/lon
                    tooltip=row['category'],
                    popup=f"POI: {row['category']}",
                    icon=folium.Icon(color=color_icon, icon="info-sign")
                ).add_to(m)


            st_folium(m, width=700, height=500)

# --- 7. Penjelasan (Opsional, di bagian bawah) ---
st.markdown("---")
st.caption("Disclaimer: Prediksi ini didasarkan pada model Machine Learning dan data yang tersedia. Hasil prediksi adalah estimasi dan bukan jaminan mutlak. Tingkat kerentanan ditentukan oleh kombinasi faktor bahaya gempa historis, demografi, dan keberadaan fasilitas di sekitar lokasi.")
st.caption("Dibuat untuk Final Project Bootcamp Data Science oleh [Nama Anda/Tim Anda].")
