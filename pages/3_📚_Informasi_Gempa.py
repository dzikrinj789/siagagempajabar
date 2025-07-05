import streamlit as st
import pandas as pd

st.set_page_config(page_title="Informasi Gempa", page_icon="📚", layout="wide")

st.title("📚 Panduan Kesiapsiagaan Gempa Bumi")
st.markdown("Mengetahui apa yang harus dilakukan dapat menyelamatkan nyawa. Berikut adalah panduan praktis untuk menghadapi gempa bumi.")

st.markdown("---")

# --- KONTEN INFORMASI ---

with st.expander("❓ **Mengenal Skala Magnitudo (SR)**", expanded=True):
    st.write("""
    Skala Magnitudo (sering disebut Skala Richter/SR) mengukur kekuatan atau energi yang dilepaskan oleh gempa. Skala ini bersifat logaritmik, artinya setiap kenaikan satu angka berarti kekuatan gempa sekitar 32 kali lebih besar.
    """)
    skala_data = {
        "Magnitudo": ["< 4.0", "4.0 - 4.9", "5.0 - 5.9", "6.0 - 6.9", "7.0 - 7.9", "8.0 >"],
        "Efek yang Dirasakan": [
            "Umumnya tidak dirasakan, tapi terekam oleh seismograf.",
            "Dirasakan oleh banyak orang, jendela bergetar, benda ringan bisa jatuh.",
            "Kerusakan ringan pada bangunan yang tidak kokoh.",
            "Dapat menyebabkan kerusakan pada area yang cukup luas.",
            "Kerusakan serius dalam area yang sangat luas.",
            "Kerusakan masif, dapat menghancurkan komunitas di dekat pusat gempa."
        ]
    }
    st.table(pd.DataFrame(skala_data))

with st.expander("💡 **Apa yang Harus Dilakukan SEBELUM Gempa?**"):
    st.markdown("""
    - **Kenali Lingkungan Anda**: Ketahui di mana letak pintu keluar, tangga darurat, dan tempat aman di dalam dan di luar gedung/rumah.
    - **Amankan Benda Berat**: Letakkan benda besar dan berat di rak bawah. Gantung bingkai foto atau cermin jauh dari tempat tidur atau sofa.
    - **Siapkan Tas Siaga Bencana (TSB)**: Isi tas dengan persediaan darurat seperti air minum, makanan ringan, P3K, senter, peluit, dan obat-obatan pribadi.
    - **Lakukan Latihan**: Lakukan latihan evakuasi secara rutin bersama keluarga atau rekan kerja. Praktikkan teknik "Berlindung, Lindungi Kepala, Bertahan" (Drop, Cover, Hold On).
    """)

with st.expander("🚨 **Apa yang Harus Dilakukan SAAT Gempa?**"):
    st.markdown("""
    - **Jika di Dalam Ruangan**:
        - **JANGAN KELUAR!** Tetap di dalam.
        - Segera berlindung di bawah meja yang kokoh. Lindungi kepala dan leher dengan lengan Anda.
        - Jauhi jendela, kaca, dan benda-benda yang mungkin jatuh.
        - **Jangan gunakan lift.**
    - **Jika di Luar Ruangan**:
        - Cari tempat terbuka yang jauh dari gedung, pohon, tiang listrik, dan papan reklame.
        - Merunduk dan lindungi kepala Anda.
    - **Jika di Dalam Kendaraan**:
        - Menepi dan hentikan kendaraan di tempat yang aman (jauhi jembatan, terowongan, atau rambu besar).
        - Tetap di dalam mobil sampai guncangan berhenti.
    """)

with st.expander("🩹 **Apa yang Harus Dilakukan SETELAH Gempa?**"):
    st.markdown("""
    - **Periksa Diri dan Sekitar**: Pastikan Anda dan orang di sekitar tidak ada yang terluka. Berikan pertolongan pertama jika mampu.
    - **Waspada Gempa Susulan**: Gempa susulan bisa terjadi kapan saja. Tetap waspada.
    - **Periksa Potensi Bahaya**: Periksa kebocoran gas (jika tercium bau gas, segera keluar), kerusakan listrik, dan kerusakan struktur bangunan.
    - **Ikuti Informasi Resmi**: Dengarkan informasi dari sumber yang terpercaya seperti BMKG atau BPBD. Jangan mudah percaya pada hoaks.
    - **Evakuasi jika Diperlukan**: Jika bangunan rusak parah atau berada di area rawan tsunami (jika gempa terjadi di laut), segera evakuasi ke tempat yang lebih tinggi dan aman.
    """)

with st.expander("🗺️ **Sesar Aktif Utama di Jawa Barat**"):
    st.write("Jawa Barat dilintasi oleh beberapa sesar (patahan) aktif yang menjadi sumber gempa bumi. Mengenali sesar di dekat Anda dapat meningkatkan kewaspadaan.")
    
    # Data sesar (contoh)
    data_sesar = {
        'Sesar': ['Sesar Lembang', 'Sesar Cimandiri', 'Sesar Baribis', 'Sesar Garsela (Garut Selatan)'],
        'Perkiraan Panjang': ['~30 km', '~100 km', '~100 km', '~42 km'],
        'Lokasi Melintas (Contoh)': ['Bandung Utara, Lembang', 'Sukabumi, Cianjur', 'Subang, Purwakarta, Karawang', 'Garut Selatan, Bandung Selatan'],
        'Potensi Magnitudo': ['6.5 - 7.0 M', '7.0 M', '6.5 - 7.0 M', '6.5 - 7.0 M']
    }
    st.table(pd.DataFrame(data_sesar))
    st.warning("Data di atas adalah perkiraan untuk tujuan edukasi. Selalu merujuk pada data resmi dari lembaga geologi dan BMKG.")


st.markdown("---")
st.info("Informasi di halaman ini bersifat umum. Selalu ikuti arahan dari otoritas lokal di wilayah Anda.")