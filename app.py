import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💼",
    layout="wide",
)

# ── Theme toggle (sidebar) ────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

with st.sidebar:
    st.title("⚙️ Settings")
    chosen = st.radio(
        "Theme",
        options=["Dark", "Light"],
        index=0 if st.session_state.theme == "Dark" else 1,
        horizontal=True,
    )
    st.session_state.theme = chosen

DARK = st.session_state.theme == "Dark"

# ── Theme colour palettes ─────────────────────────────────────────────────────
if DARK:
    PAGE_BG      = "#0e1117"
    CARD_BG      = "#1e2130"
    TEXT_COLOR   = "#ffffff"
    MUTED_COLOR  = "#a0aab4"
    BORDER_COLOR = "#2e3250"
    # matplotlib
    FIG_BG       = "#0e1117"
    AX_BG        = "#161b2e"
    LABEL_COLOR  = "#e0e0e0"
    GRID_COLOR   = "#2a2f4a"
    SPINE_COLOR  = "#3a3f5a"
    LEGEND_FACE  = "#1e2130"
    LEGEND_EDGE  = "#3a3f5a"
    SCATTER_CLR  = "#4a9eed"
    LINE_CLR     = "#ff4b6e"
    PRED_CLR     = "#00e676"
    PRED_EDGE    = "#ffffff"
else:
    PAGE_BG      = "#f5f7fa"
    CARD_BG      = "#ffffff"
    TEXT_COLOR   = "#1a1a2e"
    MUTED_COLOR  = "#5a6a7a"
    BORDER_COLOR = "#dce3ed"
    # matplotlib
    FIG_BG       = "#f5f7fa"
    AX_BG        = "#ffffff"
    LABEL_COLOR  = "#1a1a2e"
    GRID_COLOR   = "#dce3ed"
    SPINE_COLOR  = "#c0c8d4"
    LEGEND_FACE  = "#ffffff"
    LEGEND_EDGE  = "#c0c8d4"
    SCATTER_CLR  = "#1a6fbd"
    LINE_CLR     = "#d62839"
    PRED_CLR     = "#007a3d"
    PRED_EDGE    = "#1a1a2e"

# ── Inject CSS for page & metric cards ───────────────────────────────────────
st.markdown(
    f"""
    <style>
        /* Main page background */
        .stApp {{
            background-color: {PAGE_BG};
            color: {TEXT_COLOR};
        }}
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {CARD_BG};
            border-right: 1px solid {BORDER_COLOR};
        }}
        /* Metric cards */
        [data-testid="stMetric"] {{
            background-color: {CARD_BG};
            border: 1px solid {BORDER_COLOR};
            border-radius: 10px;
            padding: 14px 18px;
        }}
        [data-testid="stMetricLabel"] {{
            color: {MUTED_COLOR} !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {TEXT_COLOR} !important;
        }}
        /* Dividers */
        hr {{
            border-color: {BORDER_COLOR};
        }}
        /* General text */
        h1, h2, h3, h4, p, label, .stMarkdown {{
            color: {TEXT_COLOR} !important;
        }}
        .stCaption, .stCaption p {{
            color: {MUTED_COLOR} !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load data & train model (cached) ─────────────────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv("Salary_Data.csv")
    X = df[["YearsExperience"]]
    y = df["Salary"]
    model = LinearRegression()
    model.fit(X, y)
    return model, df

model, df = load_model()

X         = df[["YearsExperience"]]
y         = df["Salary"]
r2        = model.score(X, y)
coef      = model.coef_[0]
intercept = model.intercept_

# ── Page title ────────────────────────────────────────────────────────────────
st.title("💼 Salary Prediction Dashboard")
st.caption("Trained on 30 employee records — Simple Linear Regression")
st.divider()

# ── Two-column layout ─────────────────────────────────────────────────────────
col_chart, col_panel = st.columns([3, 2])

# ── Right panel: Metrics + Predictor ─────────────────────────────────────────
with col_panel:
    # Metrics Panel
    st.header("📊 Model Overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score",     f"{r2:.4f}")
    m2.metric("Slope ($/yr)", f"${coef:,.2f}")
    m3.metric("Intercept",    f"${intercept:,.2f}")
    st.info(f"**Formula:** Salary = {coef:,.2f} × Years + {intercept:,.2f}")

    st.divider()

    # Prediction Input & Output
    st.header("🔮 Predict a Salary")
    years = st.slider(
        "Years of Experience",
        min_value=0.0, max_value=15.0,
        value=5.0, step=0.1,
    )
    years = st.number_input(
        "Or type an exact value",
        min_value=0.0, max_value=50.0,
        value=float(years), step=0.1,
    )

    predicted = model.predict(
        pd.DataFrame([[years]], columns=["YearsExperience"])
    )[0]
    predicted = max(predicted, 0.0)

    st.divider()
    st.metric(
        label=f"Predicted Salary for {years:.1f} yr(s)",
        value=f"${predicted:,.2f}",
    )

# ── Left panel: Scatter plot + regression line ────────────────────────────────
with col_chart:
    st.header("📈 Data & Regression Line")

    x_min  = df["YearsExperience"].min()
    x_max  = df["YearsExperience"].max()
    x_line = np.linspace(x_min, x_max, 200)
    y_line = model.predict(
        pd.DataFrame(x_line, columns=["YearsExperience"])
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)

    # Actual data points
    ax.scatter(
        df["YearsExperience"], df["Salary"],
        color=SCATTER_CLR, s=70, zorder=3, label="Actual Data",
    )

    # Regression line
    ax.plot(
        x_line, y_line,
        color=LINE_CLR, linewidth=2.5, label="Regression Line",
    )

    # Predicted point highlight
    ax.scatter(
        [years], [predicted],
        color=PRED_CLR, s=130, zorder=5,
        edgecolors=PRED_EDGE, linewidths=1.5,
        label=f"Your Input ({years:.1f} yrs → ${predicted:,.0f})",
    )

    ax.set_xlabel("Years of Experience", color=LABEL_COLOR, fontsize=12)
    ax.set_ylabel("Salary ($)", color=LABEL_COLOR, fontsize=12)
    ax.set_title(
        "Experience vs Salary — Training Data & Regression Line",
        color=LABEL_COLOR, fontsize=13, pad=12,
    )
    ax.tick_params(colors=LABEL_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.6)
    ax.legend(
        facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE,
        labelcolor=LABEL_COLOR,
    )
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, _: f"${val:,.0f}")
    )

    st.pyplot(fig)
    plt.close(fig)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: Linear Regression | Library: scikit-learn | Data: Salary_Data.csv (30 records)")
