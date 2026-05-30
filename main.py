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
        max_value=104,
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

normalized["C1"] = X["Original Price"].min() / X["Original Price"]
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
    col1, col2 = st.columns(2)
    with col1:

        st.subheader("Distribusi Bobot")

        fig_pie, ax_pie = plt.subplots(figsize=(5,5))

        labels = [
            "Price",
            "Recent",
            "All Review",
            "Language",
            "Feature"
        ]

        weights = [w1, w2, w3, w4, w5]

        ax_pie.pie(
            weights,
            labels=labels,
            autopct="%1.0f%%",
            startangle=90
        )

        st.pyplot(fig_pie)

    with col2:

        st.subheader("Bobot Kriteria")

        weight_df = pd.DataFrame({
            "Kriteria": [
                "Price",
                "Recent Reviews",
                "All Reviews",
                "Languages",
                "Features"
            ],
            "Bobot": weights
        })

        fig_weight, ax_weight = plt.subplots(figsize=(6,5))

        ax_weight.barh(
            weight_df["Kriteria"],
            weight_df["Bobot"]
        )

        ax_weight.set_xlabel("Bobot")

        st.pyplot(fig_weight)

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.55)))

    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(top)))

    bars = ax.barh(top["Title"], top["SAW Score"], color=colors)

    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

    ax.set_title(
        f"Top {top_n} Steam Games",
        fontsize=15,
        fontweight='bold'
    )

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.002,
            bar.get_y() + bar.get_height()/2,
            f'{width:.4f}',
            va='center',
            color='white'
        )

    st.pyplot(fig)

# Tab 3

with tab3:

    st.subheader("Matriks Normalisasi")

    norm_display = normalized.head(20).copy()

    st.dataframe(
        norm_display.style.format("{:.4f}").background_gradient(cmap="viridis"),
        use_container_width=True,
    )

# Tab 4

with tab4:

    st.subheader("Dataset")

    search = st.text_input("🔍 Cari game:")

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