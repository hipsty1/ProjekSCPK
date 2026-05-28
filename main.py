import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="SPK Rekomendasi Game Steam",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 SPK Rekomendasi Game Steam")
st.caption("Metode Simple Additive Weighting (SAW)")


@st.cache_data
def load_data():

    df = pd.read_csv("merged_data.csv")

    cols = [
        "Original Price",
        "Recent Reviews Number",
        "All Reviews Number",
        "Supported Languages",
        "Game Features"
    ]

    df = df.dropna(subset=cols)

    df["Original Price"] = (
        df["Original Price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(
            r"(?i).*free.*",
            "0",
            regex=True
        )
    )

    df["Recent Reviews Number"] = (
        df["Recent Reviews Number"]
        .astype(str)
        .str.extract(r'of the ([\d,]+)')[0]
        .str.replace(",", "", regex=False)
    )

    df["All Reviews Number"] = (
        df["All Reviews Number"]
        .astype(str)
        .str.extract(r'of the ([\d,]+)')[0]
        .str.replace(",", "", regex=False)
    )

    for col in [
        "Original Price",
        "Recent Reviews Number",
        "All Reviews Number"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna()

    df["Language Count"] = (
        df["Supported Languages"]
        .fillna("")
        .apply(
            lambda x: len(str(x).split(","))
        )
    )

    df["Feature Count"] = (
        df["Game Features"]
        .fillna("")
        .apply(
            lambda x: len(str(x).split(","))
        )
    )

    df["Recent Reviews Number"] = np.log1p(
        df["Recent Reviews Number"]
    )

    df["All Reviews Number"] = np.log1p(
        df["All Reviews Number"]
    )

    return df


df = load_data()

st.sidebar.header("Pengaturan")

criteria = {}

if st.sidebar.checkbox("Harga", value=True):
    criteria["Original Price"] = st.sidebar.slider(
        "Bobot Harga",
        0.0, 1.0, 0.20, 0.05
    )

if st.sidebar.checkbox("Review Terbaru", value=True):
    criteria["Recent Reviews Number"] = st.sidebar.slider(
        "Bobot Review Terbaru",
        0.0, 1.0, 0.25, 0.05
    )

if st.sidebar.checkbox("Total Review", value=True):
    criteria["All Reviews Number"] = st.sidebar.slider(
        "Bobot Total Review",
        0.0, 1.0, 0.30, 0.05
    )

if st.sidebar.checkbox("Jumlah Bahasa", value=True):
    criteria["Language Count"] = st.sidebar.slider(
        "Bobot Jumlah Bahasa",
        0.0, 1.0, 0.10, 0.05
    )

if st.sidebar.checkbox("Jumlah Fitur", value=True):
    criteria["Feature Count"] = st.sidebar.slider(
        "Bobot Jumlah Fitur",
        0.0, 1.0, 0.15, 0.05
    )

free_only = st.sidebar.checkbox(
    "Hanya Game Gratis"
)

review_limit = np.log1p(100000)

if free_only:

    df = df[
        (df["Original Price"] == 0)
        &
        (
            df["All Reviews Number"]
            >= review_limit
        )
    ]

else:

    df = df[
        df["All Reviews Number"]
        >= review_limit
    ]

if len(df) == 0:
    st.warning(
        "Tidak ada data yang sesuai filter."
    )
    st.stop()

if len(criteria) == 0:
    st.warning(
        "Pilih minimal satu kriteria."
    )
    st.stop()

total_weight = sum(criteria.values())

if total_weight == 0:
    st.warning(
        "Total bobot tidak boleh nol."
    )
    st.stop()

weights = {
    k: v / total_weight
    for k, v in criteria.items()
}

X = df[
    list(weights.keys())
].copy()

normalized = pd.DataFrame(
    index=X.index
)

for col in X.columns:

    if col == "Original Price":

        if X[col].nunique() == 1:

            normalized[col] = 1.0

        else:

            normalized[col] = (
                X[col].max() - X[col]
            ) / (
                X[col].max() - X[col].min()
            )

    else:

        max_val = X[col].max()

        if max_val == 0:

            normalized[col] = 1.0

        else:

            normalized[col] = (
                X[col] / max_val
            )

normalized = normalized.fillna(1)

scores = np.zeros(
    len(normalized)
)

for col, weight in weights.items():

    scores += (
        normalized[col]
        * weight
    )

df_saw = df.copy()

df_saw["SAW Score"] = scores

ranking = (
    df_saw
    .sort_values(
        by="SAW Score",
        ascending=False
    )
    .reset_index(drop=True)
)

ranking["Rank"] = (
    ranking.index + 1
)

best = ranking.iloc[0]

st.success(
    f"""
🥇 GAME TERBAIK

{best['Title']}

SAW Score : {best['SAW Score']:.4f}
"""
)

m1, m2 = st.columns(2)

with m1:
    st.metric(
        "Jumlah Game",
        len(df)
    )

with m2:
    st.metric(
        "Kriteria Aktif",
        len(weights)
    )

st.subheader("Dataset")

st.dataframe(
    df,
    height=350,
    use_container_width=True
)

left, right = st.columns(2)

with left:

    st.subheader("Normalisasi")

    st.dataframe(
        normalized.round(4),
        height=350,
        use_container_width=True
    )

with right:

    st.subheader("Ranking")

    st.dataframe(
        ranking[
            [
                "Rank",
                "Title",
                "Original Price",
                "SAW Score"
            ]
        ],
        height=350,
        use_container_width=True
    )

st.subheader("🏅 Podium")

podium = ranking.head(3)

c1, c2, c3 = st.columns(3)

with c1:
    st.info(
        f"🥇 {podium.iloc[0]['Title']}\n\nSkor: {podium.iloc[0]['SAW Score']:.4f}"
    )

if len(podium) > 1:
    with c2:
        st.info(
            f"🥈 {podium.iloc[1]['Title']}\n\nSkor: {podium.iloc[1]['SAW Score']:.4f}"
        )

if len(podium) > 2:
    with c3:
        st.info(
            f"🥉 {podium.iloc[2]['Title']}\n\nSkor: {podium.iloc[2]['SAW Score']:.4f}"
        )

st.subheader("Top 10 Rekomendasi")

top10 = ranking.head(10)

st.dataframe(
    top10[
        [
            "Rank",
            "Title",
            "Original Price",
            "SAW Score"
        ]
    ],
    use_container_width=True
)

fig = px.bar(
    top10,
    x="SAW Score",
    y="Title",
    orientation="h",
    text="SAW Score"
)

fig.update_layout(
    height=600,
    yaxis={
        "categoryorder":
        "total ascending"
    }
)

st.plotly_chart(
    fig,
    use_container_width=True
)