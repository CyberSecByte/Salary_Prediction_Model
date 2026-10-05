# 💼 Salary Prediction Dashboard

An interactive web application that predicts employee salary from years of experience, built with **Python**, **scikit-learn**, and **Streamlit**.

---

## 📁 Project Structure

```
Salary Prediction Model/
├── app.py                        # Main Streamlit application
├── Salary_Data.csv               # Training dataset (30 records)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
python -m streamlit run app.py
```

### 3. Open in browser

Streamlit will print a local URL — open it in your browser:

```
Local URL:   http://localhost:8501
```

---

## 📊 The Dataset — `Salary_Data.csv`

| Column | Description |
|---|---|
| `YearsExperience` | Years of work experience (1.1 – 10.5) |
| `Salary` | Annual salary in USD ($37,731 – $122,391) |

- **30 records** of real employee data
- A clear positive linear relationship: salary rises consistently with experience
- This makes it a textbook dataset for **Simple Linear Regression**

---

## 🤖 The Model — How Predictions Work

The app uses **Simple Linear Regression** from `scikit-learn`.

### What is Linear Regression?

Linear Regression finds the best-fit straight line through the data points. The line is described by the equation:

```
Salary = Slope × YearsExperience + Intercept
```

| Term | What it means | Value (this model) |
|---|---|---|
| **Slope** | How much salary increases per year of experience | ~$9,450 / yr |
| **Intercept** | The estimated base salary at 0 years of experience | ~$25,792 |
| **R² Score** | How well the model fits the data (0 = bad, 1 = perfect) | **0.9570** |

An R² of **0.9570** means the model explains **95.7%** of the variation in salary — an excellent fit for a single-variable model.

### Training flow

```
Salary_Data.csv
      │
      ▼
  pandas.read_csv()         ← load data into a DataFrame
      │
      ▼
  X = [[YearsExperience]]   ← feature (input)
  y = [Salary]              ← target (output)
      │
      ▼
  LinearRegression.fit(X, y)  ← find best slope & intercept
      │
      ▼
  model.predict([[years]])   ← produce a salary estimate
```

The model is trained once at startup and cached with `@st.cache_resource` so it is not retrained on every widget interaction.

---

## 🖥️ Dashboard Layout

The app is split into two columns:

### Left column — 📈 Chart

- **Scatter plot** of all 30 training data points (blue dots)
- **Regression line** drawn through the data (red line)
- **Your predicted point** highlighted as a green dot that moves in real time as you adjust the input

### Right column — Controls & Stats

#### 📊 Model Overview
Three metric cards showing:
- **R² Score** — overall model accuracy
- **Slope ($/yr)** — salary increase per additional year of experience
- **Intercept** — estimated base salary

A formula banner shows the full equation, e.g.:
> `Salary = 9,449.96 × Years + 25,792.20`

#### 🔮 Predict a Salary
- **Slider** — drag from 0 to 15 years of experience
- **Number input** — type any value up to 50 years for precise control
- Both controls stay in sync
- The **Predicted Salary** metric updates instantly below

---

## 🌗 Dark / Light Theme

A theme toggle lives in the **sidebar** (click `>` in the top-left corner to open it).

| | Dark | Light |
|---|---|---|
| Background | Deep navy `#0e1117` | Light grey `#f5f7fa` |
| Cards | `#1e2130` | White `#ffffff` |
| Scatter dots | Blue `#4a9eed` | Dark blue `#1a6fbd` |
| Regression line | Red-pink `#ff4b6e` | Crimson `#d62839` |
| Predicted point | Lime green `#00e676` | Forest green `#007a3d` |

Switching theme re-renders the entire page **including the matplotlib chart**, which applies a matching colour palette for backgrounds, axes, grid lines, and labels.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework — widgets, layout, theming |
| `pandas` | Load and manipulate the CSV dataset |
| `scikit-learn` | `LinearRegression` model training and prediction |
| `numpy` | Generate smooth regression line coordinates |
| `matplotlib` | Render the scatter plot and regression line chart |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 💡 How the Theme Works (Technical)

Streamlit does not expose a runtime API to switch between its built-in dark/light themes. Instead, this app:

1. Stores the user's choice in `st.session_state.theme`
2. Defines **two complete colour palettes** as Python variables (`DARK` / `LIGHT` blocks)
3. Injects CSS targeting Streamlit's internal `data-testid` selectors via `st.markdown(..., unsafe_allow_html=True)` to override page background, sidebar, metric cards, and text colours
4. Passes the palette variables directly into `matplotlib` figure and axes properties so the chart matches the selected theme

Because Streamlit re-runs the entire script on every widget change, toggling the radio button causes a full re-render with the new palette applied everywhere.

---

*Model: Linear Regression | Library: scikit-learn | Data: Salary_Data.csv (30 records)*
