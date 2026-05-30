import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Page config

st.set_page_config(
    page_title="SPK Rekomendasi Game Steam Menggunakan Metode SAW",
    page_icon=" ",
    layout="wide",
)

# Custom CSS

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.main {
    background:linear-gradient(135deg,#9CBBFF,#3163E0,#1e293b);
    color: white;
}

h1, h2, h3 {
    color: #f8fafc !important;
}

[data-testid="stSidebar"] {
    background:linear-gradient(135deg,#9CBBFF,#3163E0,#1e293b);
    backdrop-filter: blur(10px);
    border-right: 1px solid #334155;
}

.stMetric {
    background:linear-gradient(135deg,#9CBBFF,#3163E0,#1e293b);
    padding: 15px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}

.block-container {
    background:linear-gradient(135deg,#3163E0,#1e293b,#9CBBFF);
    padding-top: 2rem;
}

div[data-baseweb="tab-list"] {
    gap: 12px;
}

button[data-baseweb="tab"] {
    background:linear-gradient(135deg,#9CBBFF,#3163E0,#1e293b);
    border-radius: 10px;
    color: white;
    padding: 10px 18px;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background:linear-gradient(135deg,#9CBBFF,#3163E0,#1e293b);
}

input {
    border-radius: 12px !important;
}

</style>
""", unsafe_allow_html=True)

# Hero Section

st.markdown("""
<div style='padding:30px;border-radius:25px;
background:linear-gradient(135deg,#9CBBFF,#3163E0,#1e293b);
box-shadow:0 10px 30px rgba(0,0,0,0.4);
color:white;text-align:center;margin-bottom:25px;'>

<h1 style='font-size:42px;'>Steam Game Recommendation System</h1>
<p style='font-size:18px;'>
Sistem Pendukung Keputusan menggunakan metode <b>Simple Additive Weighting (SAW)</b>
</p>

</div>
""", unsafe_allow_html=True)

# Load & cache data

@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("merged_data.csv")

    important_cols = [
        "Original Price",
        "Recent Reviews Number",
        "All Reviews Number",
        "Supported Languages",
        "Game Features",
    ]
    df = df.dropna(subset=important_cols)

    df["Original Price"] = (
        df["Original Price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.strip()
        .str.replace("Free to Play", "0", regex=False)
        .str.replace("Free", "0", regex=False)
    )

    df["Recent Reviews Number"] = (
        df["Recent Reviews Number"]
        .astype(str)
        .str.extract(r"of the ([\d,]+)")[0]
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    df["All Reviews Number"] = (
        df["All Reviews Number"]
        .astype(str)
        .str.extract(r"of the ([\d,]+)")[0]
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    numeric_cols = ["Original Price", "Recent Reviews Number", "All Reviews Number"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)
    df = df[df["Original Price"] >= 0]

    df["Language Count"] = df["Supported Languages"].apply(
        lambda x: len(str(x).split(","))
    )

    df["Feature Count"] = df["Game Features"].apply(
        lambda x: len(str(x).split(","))
    )

    df = df[df["All Reviews Number"] >= 100000]

    df["Recent Reviews Number"] = np.log1p(df["Recent Reviews Number"])
    df["All Reviews Number"] = np.log1p(df["All Reviews Number"])

    return df.reset_index(drop=True)


df = load_and_preprocess()

# Dashboard Statistics

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Game", f"{len(df):,}")

with col2:
    st.metric("Harga Rata-rata", f"${df['Original Price'].mean():.2f}")

with col3:
    st.metric("Avg Language", f"{df['Language Count'].mean():.1f}")

with col4:
    st.metric("Avg Review", f"{df['All Reviews Number'].mean():.2f}")

with st.expander("Tentang Metode SAW"):
    st.write("""
    Simple Additive Weighting (SAW) merupakan metode SPK
    yang melakukan normalisasi setiap kriteria kemudian
    menjumlahkannya berdasarkan bobot.

    Semakin tinggi skor SAW maka semakin direkomendasikan.
    """)

# Sidebar

with st.sidebar:
    st.header("Pengaturan")
    st.subheader("Bobot Kriteria")
    st.caption("Total bobot harus = 1.00")

    w1 = st.slider("Original Price (Cost)",    0.0, 1.0, 0.20, 0.05)
    w2 = st.slider("Recent Reviews",           0.0, 1.0, 0.25, 0.05)
    w3 = st.slider("All Reviews",              0.0, 1.0, 0.30, 0.05)
    w4 = st.slider("Language Count",          0.0, 1.0, 0.10, 0.05)
    w5 = st.slider("Feature Count",           0.0, 1.0, 0.15, 0.05)

    total_w = round(w1 + w2 + w3 + w4 + w5, 2)

    if total_w != 1.0:
        st.warning(f"Total bobot saat ini: {total_w}")
    else:
        st.success(f"Total bobot: {total_w}")

    top_n = st.number_input(
        "Jumlah Top Game",
        min_value=5,
        max_value=150,
        value=10,
        step=5,
    )

# SAW Calculation

criteria = [
    "Original Price",
    "Recent Reviews Number",
    "All Reviews Number",
    "Language Count",
    "Feature Count",
]

X = df[criteria].copy()
normalized = pd.DataFrame(index=X.index)

normalized["C1"] = 1 - (X["Original Price"] / X["Original Price"].max()) if X["Original Price"].max() > 0 else 1.0
normalized["C2"] = X["Recent Reviews Number"] / X["Recent Reviews Number"].max()
normalized["C3"] = X["All Reviews Number"] / X["All Reviews Number"].max()
normalized["C4"] = X["Language Count"] / X["Language Count"].max()
normalized["C5"] = X["Feature Count"] / X["Feature Count"].max()

weights_array = np.array([w1, w2, w3, w4, w5])
scores = normalized.dot(weights_array)

df["SAW Score"] = scores
ranking = df.sort_values("SAW Score", ascending=False)

top = ranking[[
    "Title", "Original Price", "Recent Reviews Number",
    "All Reviews Number", "Language Count",
    "Feature Count", "SAW Score",
]].head(top_n).reset_index(drop=True)

# Tabs

tab1, tab2, tab3, tab4 = st.tabs([
    "Hasil",
    "Visualisasi",
    "Normalisasi",
    "Dataset",
])

# Tab 1

with tab1:

    st.subheader(f"Top {top_n} Game")

    cards = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    colors = ["#f59e0b", "#94a3b8", "#b45309"]

    for i, card in enumerate(cards):
        if i < len(top):
            row = top.iloc[i]

            with card:
                st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, {colors[i]}, #1e293b);
                    padding:25px;
                    border-radius:22px;
                    color:white;
                    min-height:220px;
                    box-shadow:0 6px 18px rgba(0,0,0,0.35);
                '>

                <h2>{medals[i]} Rank {i+1}</h2>
                <h3>{row['Title'][:35]}</h3>

                <hr>

                <p>Score: <b>{row['SAW Score']:.4f}</b></p>
                <p>Price: ${row['Original Price']:.2f}</p>
                <p>Languages: {row['Language Count']}</p>

                </div>
                """, unsafe_allow_html=True)

    st.divider()

    st.dataframe(top, use_container_width=True)

# Tab 2

with tab2:

    # --- Chart 1: SAW Score Bar Chart (existing) ---
    st.subheader("SAW Score Ranking")

    fig1, ax1 = plt.subplots(figsize=(10, max(5, top_n * 0.55)))
    fig1.patch.set_facecolor('#111827')
    ax1.set_facecolor('#111827')

    bar_colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(top)))
    bars = ax1.barh(top["Title"], top["SAW Score"], color=bar_colors)

    ax1.tick_params(colors='white')
    ax1.xaxis.label.set_color('white')
    ax1.yaxis.label.set_color('white')
    ax1.title.set_color('white')
    ax1.set_title(f"Top {top_n} Steam Games – SAW Score", fontsize=15, fontweight='bold')

    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{width:.4f}', va='center', color='white', fontsize=8)

    st.pyplot(fig1)

    st.divider()

    # --- Chart 2: Scatter Plot – Price vs All Reviews ---
    st.subheader("Harga vs Jumlah Review (Top Game)")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig2.patch.set_facecolor('#111827')
    ax2.set_facecolor('#111827')

    scatter_colors = plt.cm.plasma(
        (top["SAW Score"] - top["SAW Score"].min()) /
        (top["SAW Score"].max() - top["SAW Score"].min() + 1e-9)
    )

    sc = ax2.scatter(
        top["Original Price"],
        top["All Reviews Number"],
        c=top["SAW Score"],
        cmap="plasma",
        s=top["Feature Count"] * 20,
        alpha=0.85,
        edgecolors='white',
        linewidths=0.5,
    )

    for _, row in top.iterrows():
        ax2.annotate(
            row["Title"][:18],
            (row["Original Price"], row["All Reviews Number"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            color='white',
            alpha=0.85,
        )

    cbar = fig2.colorbar(sc, ax=ax2)
    cbar.set_label("SAW Score", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    ax2.set_xlabel("Original Price ($)", color='white', fontsize=11)
    ax2.set_ylabel("All Reviews Number (log)", color='white', fontsize=11)
    ax2.set_title("Harga vs Review – ukuran titik = Feature Count", fontsize=13, fontweight='bold', color='white')
    ax2.tick_params(colors='white')

    for spine in ax2.spines.values():
        spine.set_edgecolor('#334155')

    st.pyplot(fig2)
    st.caption("Warna titik menunjukkan SAW Score (lebih terang = lebih tinggi). Ukuran titik menunjukkan jumlah fitur.")

    st.divider()

    # --- Chart 3: Radar Chart – Profil Kriteria Top 5 ---
    st.subheader("Profil Kriteria – Top 5 Game")

    top5 = top.head(5).copy()
    criteria_labels = ["Price\n(Cost)", "Recent\nReviews", "All\nReviews", "Language\nCount", "Feature\nCount"]
    norm_cols = ["C1", "C2", "C3", "C4", "C5"]
    top5_norm = normalized.loc[top5.index, norm_cols].values

    N = len(criteria_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig3, ax3 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig3.patch.set_facecolor('#111827')
    ax3.set_facecolor('#111827')

    radar_colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(top5)))

    for i, (_, row) in enumerate(top5.iterrows()):
        values = top5_norm[i].tolist()
        values += values[:1]
        ax3.plot(angles, values, color=radar_colors[i], linewidth=2, linestyle='solid')
        ax3.fill(angles, values, color=radar_colors[i], alpha=0.15)

    ax3.set_thetagrids(np.degrees(angles[:-1]), criteria_labels, color='white', fontsize=10)
    ax3.tick_params(colors='white')
    ax3.yaxis.set_tick_params(labelcolor='white')
    ax3.set_ylim(0, 1)

    for spine in ax3.spines.values():
        spine.set_edgecolor('#334155')

    ax3.set_title("Profil Normalisasi – Top 5 Game", color='white', fontsize=14, fontweight='bold', pad=20)

    legend_labels = [row["Title"][:25] for _, row in top5.iterrows()]
    legend = ax3.legend(legend_labels, loc='upper right', bbox_to_anchor=(1.35, 1.15),
                        fontsize=8, framealpha=0.3, labelcolor='white',
                        facecolor='#1e293b', edgecolor='#334155')

    st.pyplot(fig3)
    st.caption("Radar chart menampilkan nilai normalisasi setiap kriteria untuk 5 game teratas.")

    st.divider()

    # --- Chart 4: Distribusi Harga (Histogram) ---
    st.subheader("Distribusi Harga & Feature Count Dataset")

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))
    fig4.patch.set_facecolor('#111827')

    for ax_ in (ax4a, ax4b):
        ax_.set_facecolor('#111827')
        ax_.tick_params(colors='white')
        ax_.xaxis.label.set_color('white')
        ax_.yaxis.label.set_color('white')
        ax_.title.set_color('white')
        for spine in ax_.spines.values():
            spine.set_edgecolor('#334155')

    # Histogram harga
    n_bins = 15
    price_vals = df["Original Price"]
    counts, bin_edges = np.histogram(price_vals, bins=n_bins)
    bar_colors_hist = plt.cm.plasma(np.linspace(0.2, 0.9, n_bins))
    ax4a.bar(bin_edges[:-1], counts, width=np.diff(bin_edges),
             color=bar_colors_hist, edgecolor='#1e293b', align='edge')
    ax4a.set_xlabel("Original Price ($)", fontsize=11)
    ax4a.set_ylabel("Jumlah Game", fontsize=11)
    ax4a.set_title("Distribusi Harga Game", fontsize=13, fontweight='bold')

    # Bar chart Feature Count distribusi
    feat_counts = df["Feature Count"].value_counts().sort_index()
    fc_colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(feat_counts)))
    ax4b.bar(feat_counts.index.astype(str), feat_counts.values, color=fc_colors, edgecolor='#1e293b')
    ax4b.set_xlabel("Jumlah Fitur", fontsize=11)
    ax4b.set_ylabel("Jumlah Game", fontsize=11)
    ax4b.set_title("Distribusi Feature Count", fontsize=13, fontweight='bold')
    ax4b.tick_params(axis='x', rotation=45)

    fig4.tight_layout(pad=3)
    st.pyplot(fig4)
    st.caption("Kiri: sebaran harga seluruh game dalam dataset. Kanan: sebaran jumlah fitur yang dimiliki tiap game.")

# Tab 3

with tab3:

    st.subheader("Matriks Normalisasi")

    # Ambil index dari ranking top N, lalu ambil baris normalisasi sesuai urutan ranking
    top_indices = ranking.head(top_n).index
    norm_display = normalized.loc[top_indices].copy()
    norm_display.insert(0, "Title", df.loc[top_indices, "Title"].values)
    norm_display = norm_display.reset_index(drop=True)

    numeric_cols_norm = [c for c in norm_display.columns if c != "Title"]

    st.dataframe(
        norm_display.style
            .format("{:.4f}", subset=numeric_cols_norm)
            .background_gradient(cmap="viridis", subset=numeric_cols_norm),
        use_container_width=True,
    )

# Tab 4

with tab4:

    st.subheader("Dataset")

    search = st.text_input("Cari game:")

    display_raw = df[[
        "Title", "Original Price",
        "Recent Reviews Number",
        "All Reviews Number",
        "Language Count",
        "Feature Count",
    ]]

    if search:
        display_raw = display_raw[
            display_raw["Title"].str.contains(search, case=False, na=False)
        ]

    st.dataframe(display_raw, use_container_width=True)

# Footer

st.divider()
st.caption("SPK Rekomendasi Game Steam • Streamlit • Metode SAW - Oleh kelompok 10")
st.caption("Fahmi Firdaus (123240055) | Hendri Prasetyo (123240066)")