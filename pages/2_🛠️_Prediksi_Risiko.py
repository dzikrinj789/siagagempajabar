import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import joblib
from sklearn.preprocessing import LabelEncoder
import os
import folium
from streamlit_folium import st_folium

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Alat Prediksi", page_icon="🛠️", layout="wide")


# --- CSS ---
st.markdown("""
<style>
    /* Memberi sedikit bayangan pada container agar ada pemisah visual */
    /* Ini menargetkan container default Streamlit saat menggunakan kolom */
    .st-emotion-cache-16txtl3 {
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.05);
        padding: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- JUDUL ---
st.title("🛠️ Alat Prediksi Risiko Fisik dan Sosial Gempa")
st.markdown("Pilih lokasi di peta, isi data estimasi, dan lihat tingkat potensi risiko terhadap dampak gempa.")


# --- DEFINISI LOKASI FILE & KONSTANTA ---
DIR_DATA_FILES = "data"
MODEL_FILENAME = 'random_forest_model.pkl'
LABEL_ENCODER_FILENAME = 'label_encoder.pkl'
FEATURE_COLUMNS_FILENAME = 'feature_columns_model.pkl'

GD_GEMPA_JABAR_PATH = os.path.join(DIR_DATA_FILES, 'gdf_gempa_jabar_processed.gpkg')
GD_POI_JABAR_PATH = os.path.join(DIR_DATA_FILES, 'gdf_poi_jabar_processed.gpkg')
GD_DEMOGRAFI_JABAR_CLEAN_PATH = os.path.join(DIR_DATA_FILES, 'gdf_demografi_jabar_clean_processed.gpkg')

BUFFER_GEMPA_KM = 10
BUFFER_POI_METER = 500

# --- FUNGSI-FUNGSI BANTUAN & PREDIKSI ---
@st.cache_resource
def load_all_resources():
    print("Memuat semua sumber daya...")
    try:
        model = joblib.load(MODEL_FILENAME)
        label_encoder = joblib.load(LABEL_ENCODER_FILENAME)
        feature_cols_model = joblib.load(FEATURE_COLUMNS_FILENAME)
        if hasattr(model, 'feature_names_in_'):
            model_expected_features = model.feature_names_in_.tolist()
        else:
            model_expected_features = feature_cols_model
        gdf_gempa_jabar = gpd.read_file(GD_GEMPA_JABAR_PATH)
        gdf_poi_jabar = gpd.read_file(GD_POI_JABAR_PATH)
        gdf_demografi_jabar_clean = gpd.read_file(GD_DEMOGRAFI_JABAR_CLEAN_PATH)
        for gdf in [gdf_gempa_jabar, gdf_poi_jabar, gdf_demografi_jabar_clean]:
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
        return model, label_encoder, model_expected_features, gdf_gempa_jabar, \
               gdf_poi_jabar, gdf_demografi_jabar_clean
    except FileNotFoundError as e:
        st.error(f"ERROR: File sumber daya tidak ditemukan: {e}. Pastikan path '{DIR_DATA_FILES}' dan file-filenya benar.")
        st.stop()
    except Exception as e:
        st.error(f"ERROR: Gagal memuat sumber daya: {e}")
        st.stop()

def predict_vulnerability_for_point(
    user_lat, user_lon,
    user_jumlah_kk, user_jumlah_laki, user_jumlah_perempuan,
    user_jumlah_anak, user_jumlah_lansia,
    user_ada_rs, user_ada_sekolah, user_ada_pemerintahan, user_ada_bangunan_biasa, user_ada_fasos_lain,
    model_expected_features, gdf_gempa, gdf_poi, gdf_demografi
):
    user_point = gpd.GeoDataFrame(geometry=[Point(user_lon, user_lat)], crs="EPSG:4326")
    user_point_proj = user_point.to_crs(epsg=3857)
    point_demog_sjoin = gpd.sjoin(user_point, gdf_demografi, how="inner", predicate='intersects')
    demog_features_dict = {}
    expected_demog_features_list = ['jumlah_penduduk', 'pria', 'wanita', 'jumlah_produktif', 'jumlah_non_produktif', 'rasio_lp', 'rasio_produktif_nonproduktif', 'kepadatan_penduduk_kelurahan', 'jumlah_penduduk_miskin_kelurahan_estimasi']
    if point_demog_sjoin.empty:
        for col in expected_demog_features_list: demog_features_dict[col] = 0.0
    else:
        demog_data = point_demog_sjoin.iloc[0]
        for col in expected_demog_features_list: demog_features_dict[col] = demog_data.get(col, 0.0)
    non_produktif_user_estimate = user_jumlah_anak + user_jumlah_lansia
    non_produktif_kelurahan = demog_features_dict.get('jumlah_non_produktif', 0)
    final_non_produktif = max(non_produktif_user_estimate, non_produktif_kelurahan)
    demog_features_dict['jumlah_non_produktif'] = final_non_produktif
    jumlah_produktif_kelurahan = demog_features_dict.get('jumlah_produktif', 0)
    if final_non_produktif > 0:
        demog_features_dict['rasio_produktif_nonproduktif'] = jumlah_produktif_kelurahan / final_non_produktif
    else:
        demog_features_dict['rasio_produktif_nonproduktif'] = 1.0
    buffer_gempa = gpd.GeoDataFrame(geometry=[user_point_proj.geometry.iloc[0].buffer(BUFFER_GEMPA_KM * 1000)], crs="EPSG:3857").to_crs(epsg=4326)
    nearby_gempa_sjoin = gpd.sjoin(gdf_gempa, buffer_gempa, how="inner", predicate='intersects')
    if nearby_gempa_sjoin.empty:
        gempa_features_dict = {'count_gempa': 0, 'max_mag': 0.0, 'avg_depth': 0.0}
    else:
        gempa_features_dict = {'count_gempa': len(nearby_gempa_sjoin), 'max_mag': nearby_gempa_sjoin['mag'].max(), 'avg_depth': nearby_gempa_sjoin['depth'].mean()}
    buffer_poi = gpd.GeoDataFrame(geometry=[user_point_proj.geometry.iloc[0].buffer(BUFFER_POI_METER)], crs="EPSG:3857").to_crs(epsg=4326)
    nearby_poi_sjoin = gpd.sjoin(gdf_poi, buffer_poi, how="inner", predicate='intersects')
    poi_counts = nearby_poi_sjoin['category'].value_counts().to_dict()
    poi_features_dict = {
        'count_poi_fasilitas_kesehatan': poi_counts.get('Fasilitas Kesehatan', 0), 'count_poi_sekolah': poi_counts.get('Sekolah', 0),
        'count_poi_pemerintahan/publik': poi_counts.get('Pemerintahan/Publik', 0), 'count_poi_fasilitas_sosial/publik_lain': poi_counts.get('Fasilitas Sosial/Publik Lain', 0),
        'count_poi_bangunan_biasa': poi_counts.get('Bangunan Biasa', 0)
    }
    if user_ada_rs == 'Ya': poi_features_dict['count_poi_fasilitas_kesehatan'] = max(1, poi_features_dict['count_poi_fasilitas_kesehatan'])
    if user_ada_sekolah == 'Ya': poi_features_dict['count_poi_sekolah'] = max(1, poi_features_dict['count_poi_sekolah'])
    if user_ada_pemerintahan == 'Ya': poi_features_dict['count_poi_pemerintahan/publik'] = max(1, poi_features_dict['count_poi_pemerintahan/publik'])
    if user_ada_fasos_lain == 'Ya': poi_features_dict['count_poi_fasilitas_sosial/publik_lain'] = max(1, poi_features_dict['count_poi_fasilitas_sosial/publik_lain'])
    if user_ada_bangunan_biasa == 'Ya': poi_features_dict['count_poi_bangunan_biasa'] = max(1, poi_features_dict['count_poi_bangunan_biasa'])
    rasio_lp_user = user_jumlah_laki / user_jumlah_perempuan if user_jumlah_perempuan > 0 else 1.0
    user_input_features = {'user_input_jumlah_kk': user_jumlah_kk, 'user_input_rasio_lp': rasio_lp_user}
    all_features_dict = {**demog_features_dict, **gempa_features_dict, **poi_features_dict, **user_input_features}
    X_new_series = pd.Series(all_features_dict)
    return pd.DataFrame([X_new_series]).reindex(columns=model_expected_features, fill_value=0.0)

def predict_vulnerability(X_new_predict, model, label_encoder):
    prediction_encoded = model.predict(X_new_predict)[0]
    return label_encoder.inverse_transform([prediction_encoded])[0]

# --- MUAT SUMBER DAYA & INISIALISASI STATE ---
model, label_encoder, model_expected_features, gdf_gempa_jabar, \
gdf_poi_jabar, gdf_demografi_jabar_clean = load_all_resources()

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.map_data = None
if 'confirmed_location' not in st.session_state:
    st.session_state.confirmed_location = None
if 'last_map_click' not in st.session_state:
    st.session_state.last_map_click = None

# --- TAMPILAN ANTARMUKA (UI) ---
st.markdown("---")
col1, col2 = st.columns(2, gap="large")

with col1:
    st.header("📍 1. Pilih Lokasi di Peta")
    CENTER_START = [-6.9175, 107.6191] # Koordinat Bandung
    map_input = folium.Map(location=CENTER_START, zoom_start=10, tiles="cartodbpositron")
    map_input.get_root().html.add_child(folium.Element("<style>.leaflet-container {cursor: crosshair;}</style>"))
    marker_location = None
    if st.session_state.last_map_click: marker_location = [st.session_state.last_map_click['lat'], st.session_state.last_map_click['lng']]
    elif st.session_state.confirmed_location: marker_location = [st.session_state.confirmed_location['lat'], st.session_state.confirmed_location['lng']]
    if marker_location:
        folium.Marker(location=marker_location, popup="Pilihan Anda", icon=folium.Icon(color='blue', icon='map-marker')).add_to(map_input)
        map_input.location, map_input.zoom_start = marker_location, 14
    map_output = st_folium(map_input, use_container_width=True, height=350)
    if map_output and map_output['last_clicked']: st.session_state.last_map_click = map_output['last_clicked']
    
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("✅ Konfirmasi Pilihan Titik", use_container_width=True):
            if st.session_state.last_map_click:
                st.session_state.confirmed_location = st.session_state.last_map_click
                st.success("Titik berhasil dikonfirmasi!")
            else:
                st.warning("Silakan klik sebuah titik di peta terlebih dahulu.")
    with btn_col2:
        if st.button("🔄 Reset Pilihan", use_container_width=True):
            st.session_state.confirmed_location = None
            st.session_state.last_map_click = None
            st.info("Pilihan lokasi direset.")
            st.rerun()

    if st.session_state.confirmed_location:
        st.markdown(f"**Titik Terkonfirmasi:** `{st.session_state.confirmed_location['lat']:.6f}, {st.session_state.confirmed_location['lng']:.6f}`")
    else:
        st.info("Belum ada titik lokasi yang dikonfirmasi.")

with col2:
    st.header("👨‍👩‍👧‍👦 2. Data Demografi di Lokasi (Estimasi)")
    jumlah_kk_input = st.number_input("**Jumlah Kepala Keluarga (KK) di lokasi**", min_value=0, value=50, step=5, help="Estimasi jumlah KK di sekitar titik lokasi.")
    sub_col1, sub_col2 = st.columns(2)
    jumlah_laki_input = sub_col1.number_input("**Estimasi Jumlah Laki-laki**", min_value=0, value=75, step=5)
    jumlah_perempuan_input = sub_col2.number_input("**Estimasi Jumlah Perempuan**", min_value=0, value=75, step=5)
    st.markdown("###### **Estimasi Kelompok Usia Rentan**")
    sub_col3, sub_col4 = st.columns(2)
    jumlah_anak_input = sub_col3.number_input("**Jumlah Anak-anak (<15 thn)**", min_value=0, value=20, step=2, help="Perkiraan jumlah anak-anak di sekitar lokasi.")
    jumlah_lansia_input = sub_col4.number_input("**Jumlah Lansia (>64 thn)**", min_value=0, value=10, step=2, help="Perkiraan jumlah lansia di sekitar lokasi.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("🏥 3. Konfirmasi Fasilitas Penting")
    st.write(f"Konfirmasi fasilitas dalam radius **{BUFFER_POI_METER} meter** di sekitar titik.")
    ada_rumah_sakit_terdekat = st.radio("**Ada Rumah Sakit/Klinik?**", ('Tidak', 'Ya'), horizontal=True)
    ada_sekolah_terdekat = st.radio("**Ada Sekolah?**", ('Tidak', 'Ya'), horizontal=True)
    ada_kantor_pemerintahan_terdekat = st.radio("**Ada Kantor Pemerintahan/Publik?**", ('Tidak', 'Ya'), horizontal=True)
    ada_fasos_lain_terdekat = st.radio("**Ada Fasilitas Sosial Lain?** (Tempat Ibadah, dll)", ('Tidak', 'Ya'), horizontal=True)
    ada_bangunan_biasa_terdekat = st.radio("**Ada Bangunan Biasa Lainnya?** (Toko, Rumah, dll)", ('Tidak', 'Ya'), horizontal=True)
    
    st.markdown("---")
    st.header("✨ 4. Mulai Prediksi")
    if st.button("Prediksi Tingkat Risiko Fisik dan Sosial", use_container_width=True, type="primary", key="prediksi_button"):
        if st.session_state.confirmed_location is None:
            st.warning("⚠️ **Harap pilih dan konfirmasi lokasi di peta terlebih dahulu!**")
        else:
            with st.spinner('Menganalisis data dan memprediksi...'):
                lat, lon = st.session_state.confirmed_location['lat'], st.session_state.confirmed_location['lng']
                X_new = predict_vulnerability_for_point(
                    lat, lon, 
                    jumlah_kk_input, jumlah_laki_input, jumlah_perempuan_input,
                    jumlah_anak_input, jumlah_lansia_input,
                    ada_rumah_sakit_terdekat, ada_sekolah_terdekat, ada_kantor_pemerintahan_terdekat,
                    ada_bangunan_biasa_terdekat, ada_fasos_lain_terdekat,
                    model_expected_features, gdf_gempa_jabar, gdf_poi_jabar, gdf_demografi_jabar_clean
                )
                predicted_level = predict_vulnerability(X_new, model, label_encoder)
                st.session_state.prediction_made = True
                st.session_state.predicted_level = predicted_level
                st.session_state.map_data = {'latitude': lat, 'longitude': lon}

# --- HASIL PREDIKSI ---
if st.session_state.prediction_made:
    predicted_level = st.session_state.predicted_level
    latitude = st.session_state.map_data['latitude']
    longitude = st.session_state.map_data['longitude']
    user_point_gdf = gpd.GeoDataFrame(geometry=[Point(longitude, latitude)], crs="EPSG:4326")
    
    st.markdown("---")
    st.header("📊 Hasil Prediksi")
    if predicted_level == 'Tinggi':
        st.error(f"**Tingkat Risiko: TINGGI**")
    elif predicted_level == 'Sedang':
        st.warning(f"**Tingkat Risiko: SEDANG**")
    else:
        st.success(f"**Tingkat Risiko: RENDAH**")

    st.markdown("---")
    st.header("💡 Mengapa Tingkat Risiko Ini?")
    if predicted_level == 'Tinggi':
        st.markdown("""
        Lokasi dengan tingkat risiko **TINGGI** biasanya memiliki satu atau lebih karakteristik yang meningkatkan kerentanan:
        * **Kepadatan Penduduk Tinggi:** Area padat penduduk meningkatkan potensi jumlah korban jiwa dan skala kerusakan fisik.
        * **Dekat dengan Zona Gempa Aktif:** Lokasi ini mungkin sangat dekat dengan riwayat gempa berkekuatan tinggi atau berada di zona sesar aktif, meningkatkan kemungkinan guncangan hebat.
        * **Keterbatasan Akses Fasilitas:** Mungkin ada kekurangan fasilitas kesehatan yang memadai, sekolah yang aman, atau infrastruktur penting lainnya yang dapat menghambat upaya penyelamatan dan pemulihan.
        * **Demografi Rentan:** Tingginya proporsi kelompok rentan (anak-anak, lansia) atau rasio produktif-nonproduktif yang tidak seimbang dapat memperburuk dampak sosial bencana.
        
        **Saran:** Tindakan mitigasi yang agresif sangat diperlukan. Ini termasuk evaluasi struktur bangunan secara menyeluruh, pengembangan sistem peringatan dini, pelatihan darurat reguler, dan penguatan infrastruktur kritis. Prioritaskan pembangunan komunitas yang tangguh dan program edukasi bencana yang intensif.
        """)
    elif predicted_level == 'Sedang':
        st.markdown("""
        Lokasi dengan tingkat risiko **SEDANG** mungkin menunjukkan campuran faktor-faktor, seperti:
        * **Kepadatan Penduduk Sedang:** Ada populasi yang cukup signifikan, yang berarti potensi dampak bisa moderat.
        * **Kedekatan dengan Area Gempa:** Mungkin ada riwayat gempa dengan magnitudo sedang atau lokasi berada dalam jangkauan dampak gempa yang lebih luas, meskipun tidak selalu di pusatnya.
        * **Ketersediaan Fasilitas:** Fasilitas penting mungkin ada, tetapi distribusinya atau kapasitasnya bisa terbatas dibandingkan dengan kebutuhan saat bencana.
        * **Variabilitas Demografi:** Ada kemungkinan variasi dalam struktur demografi yang memerlukan perhatian lebih dalam perencanaan respons.

        **Saran:** Direkomendasikan untuk meningkatkan kesiapsiagaan komunitas, mengembangkan rencana evakuasi yang lebih rinci, dan mendorong penguatan bangunan. Pelatihan darurat dan simulasi evakuasi sangat dianjurkan.
        """)
    else: # Rendah
        st.markdown("""
        Lokasi dengan tingkat risiko **RENDAH** umumnya memiliki kombinasi karakteristik berikut:
        * **Kepadatan Penduduk Rendah:** Area ini cenderung memiliki jumlah penduduk yang lebih sedikit, mengurangi potensi dampak korban jiwa dan kerusakan bangunan.
        * **Jauh dari Pusat Gempa Aktif:** Lokasi ini relatif jauh dari riwayat pusat gempa signifikan, sehingga kemungkinan terdampak langsung oleh guncangan kuat lebih kecil.
        * **Fasilitas Penting Memadai:** Akses ke fasilitas kesehatan, sekolah, dan infrastruktur publik yang kuat dapat mendukung respons cepat dan pemulihan jika terjadi bencana.
        * **Demografi Stabil:** Rasio usia produktif dan non-produktif yang seimbang, serta jumlah kepala keluarga yang terkelola, menunjukkan kapasitas adaptasi komunitas yang lebih baik.

        **Saran:** Meskipun risikonya rendah, tetap penting untuk memiliki rencana darurat dasar, mengetahui jalur evakuasi, dan memastikan bangunan memenuhi standar keselamatan gempa.
        """)

    st.markdown("---")
    st.header("🗺️ Peta Lokasi Anda & Data Kontekstual")
    kab_name, kec_name, kel_name = "Tidak Terdeteksi", "Tidak Terdeteksi", "Tidak Terdeteksi"
    kelurahan_info = None
    if all(col in gdf_demografi_jabar_clean.columns for col in ['nama_kab', 'nama_kec', 'nama_kel']):
        sjoin_result = gpd.sjoin(user_point_gdf, gdf_demografi_jabar_clean[['geometry', 'nama_kab', 'nama_kec', 'nama_kel']], how="inner", predicate='intersects')
        if not sjoin_result.empty:
            kelurahan_info = sjoin_result
            kab_name = kelurahan_info.iloc[0]['nama_kab']
            kec_name = kelurahan_info.iloc[0]['nama_kec']
            kel_name = kelurahan_info.iloc[0]['nama_kel']
    st.write(f"**Lokasi Administratif (Estimasi):** {kel_name}, {kec_name}, {kab_name}")
    
    m_results = folium.Map(location=[latitude, longitude], zoom_start=15, tiles="cartodbpositron")
    
    if kelurahan_info is not None:
        style = {'fillColor': '#3186cc', 'color': '#3186cc', 'weight': 2, 'fillOpacity': 0.2}
        folium.GeoJson(
            kelurahan_info.geometry, style_function=lambda x: style,
            tooltip=f"<b>Kelurahan: {kel_name}</b>", name="Batas Kelurahan"
        ).add_to(m_results)
        
    user_point_proj = user_point_gdf.to_crs(epsg=3857)
    gdf_gempa_proj = gdf_gempa_jabar.to_crs(epsg=3857)
    gempa_buffer = user_point_proj.geometry.iloc[0].buffer(20000)
    nearby_gempa_map = gdf_gempa_jabar[gdf_gempa_proj.geometry.intersects(gempa_buffer)]
    gempa_group = folium.FeatureGroup(name="Gempa Terdekat (Radius 20 km)", show=True).add_to(m_results)
    if not nearby_gempa_map.empty:
        for _, row in nearby_gempa_map.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']], radius=row['mag'] * 1.5,
                color='red', fill=True, fill_color='darkred', fill_opacity=0.6,
                tooltip=f"Gempa M{row['mag']:.1f}",
                popup=f"<b>Magnitudo: {row['mag']:.1f}</b><br>Kedalaman: {row['depth']:.1f} km"
            ).add_to(gempa_group)
            
    gdf_poi_proj = gdf_poi_jabar.to_crs(epsg=3857)
    poi_buffer = user_point_proj.geometry.iloc[0].buffer(1000)
    nearby_poi_map = gdf_poi_jabar[gdf_poi_proj.geometry.intersects(poi_buffer)]
    poi_group = folium.FeatureGroup(name="Fasilitas Umum Terdekat (Radius 1 km)", show=True).add_to(m_results)
    if not nearby_poi_map.empty:
        poi_icon_map = {
            'Fasilitas Kesehatan': {'color': 'red', 'icon': 'plus-sign'},
            'Sekolah': {'color': 'blue', 'icon': 'education'},
            'Pemerintahan/Publik': {'color': 'darkblue', 'icon': 'bank'},
            'Fasilitas Sosial/Publik Lain': {'color': 'green', 'icon': 'tree-conifer'},
            'Bangunan Biasa': {'color': 'gray', 'icon': 'home'}
        }
        for _, poi in nearby_poi_map.iterrows():
            category = poi['category']
            icon_style = poi_icon_map.get(category, {'color': 'purple', 'icon': 'info-sign'})
            folium.Marker(
                location=[poi.geometry.y, poi.geometry.x], tooltip=category,
                icon=folium.Icon(color=icon_style['color'], icon=icon_style['icon'], prefix='glyphicon')
            ).add_to(poi_group)
            
    folium.Marker(
        [latitude, longitude], tooltip="Lokasi Anda",
        popup=f"<b>Prediksi: {predicted_level}</b>",
        icon=folium.Icon(color="orange", icon="star")
    ).add_to(m_results)
    folium.LayerControl().add_to(m_results)
    
    st_folium(m_results, use_container_width=True, height=500, returned_objects=[])
    
    st.markdown("---")
    st.write("**Catatan Peta Kontekstual:**")
    if nearby_gempa_map.empty:
        st.info("Tidak ditemukan data gempa signifikan (M>4) dalam radius 20 km.")
    if nearby_poi_map.empty:
        st.info("Tidak ditemukan data Fasilitas Umum (POI) dalam radius 1 km dari lokasi terpilih.")
    if kelurahan_info is None:
        st.info("Informasi batas wilayah kelurahan tidak tersedia untuk titik lokasi ini.")
