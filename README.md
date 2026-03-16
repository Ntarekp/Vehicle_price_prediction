# 🚗 Vehicle Insights Dashboard

A full-stack Machine Learning web application built with **Django** that predicts vehicle prices, classifies owner income levels, and segments customers into meaningful groups — all visualised through an interactive dashboard.

---

## 📸 Project Overview

This project was built as part of a Machine Learning course assignment. It demonstrates:

- **Exploratory Data Analysis (EDA)** with an interactive Rwanda client distribution map
- **Regression** — predicting vehicle selling prices using Random Forest
- **Classification** — predicting owner income category using Random Forest
- **Clustering** — segmenting customers into groups (Budget → Ultra Premium) using KMeans with a Silhouette Score above 0.9 and CV ≤ 15% per class

---

## 🗂️ Project Structure

```
Price_prediction/
│
├── config/
│   ├── settings.py          # Django project settings
│   └── urls.py              # Root URL configuration
│
├── dummy-data/
│   ├── vehicles_ml_dataset.csv      # Main dataset
│   └── rwanda_districts.geojson     # Rwanda map boundaries
│
├── model_generators/
│   ├── regression/
│   │   ├── train_regression.py      # Train price prediction model
│   │   └── regression_model.pkl     # Saved model (generated after training)
│   │
│   ├── classification/
│   │   ├── train_classifier.py      # Train income category model
│   │   └── classification_model.pkl # Saved model (generated after training)
│   │
│   └── clustering/
│       ├── train_cluster.py         # Train customer segmentation model
│       ├── clustering_model.pkl     # Original model (generated after training)
│       ├── clustering_model_refined.pkl  # Refined model (generated after training)
│       └── qt_scaler.pkl            # QuantileTransformer scaler
│
├── predictor/
│   ├── views.py             # Django view functions (request handling)
│   ├── urls.py              # App URL routing
│   ├── data_exploration.py  # EDA functions + interactive Rwanda map
│   └── templates/predictor/
│       ├── index.html                   # EDA page
│       ├── regression_analysis.html     # Regression page
│       ├── classification_analysis.html # Classification page
│       └── clustering_analysis.html     # Clustering page
│
├── manage.py                # Django management tool
├── requirements.txt         # All Python dependencies
└── README.md                # This file
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Ntarekp/Vehicle_price_prediction.git
cd Vehicle_price_prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate
```

### 3. Install all dependencies

```bash
pip install -r requirements.txt
```

### 4. Train all machine learning models

Run each training script once before starting the server. This generates the `.pkl` model files:

```bash
python model_generators/regression/train_regression.py
python model_generators/classification/train_classifier.py
python model_generators/clustering/train_cluster.py
```

> ⏱️ The clustering script takes approximately 2 minutes due to the grid search for optimal CV ≤ 15% per class.

### 5. Run the Django development server

```bash
python manage.py runserver
```

### 6. Open in your browser

```
http://127.0.0.1:8000/data_exploration/
```

---

## 📦 Requirements

All dependencies are listed in `requirements.txt`:

```
django
pandas
numpy
scikit-learn
joblib
folium
branca
geopandas
matplotlib
plotly
```

Install them all at once with:

```bash
pip install -r requirements.txt
```

---

## 🌐 Pages & URLs

| Page | URL | Description |
|------|-----|-------------|
| Exploratory Data Analysis | `/data_exploration/` | Dataset preview, statistics, interactive Rwanda map |
| Regression Analysis | `/regression_analysis/` | Predict vehicle selling price |
| Classification Analysis | `/classification_analysis/` | Predict owner income category |
| Clustering Analysis | `/clustering_analysis/` | Segment customers + CV per class table |

---

## 🤖 Machine Learning Models

### 1. Regression — Selling Price Prediction
- **Algorithm:** Random Forest Regressor
- **Features:** Year, Kilometers Driven, Seating Capacity, Owner Income
- **Target:** Selling Price
- **Metric:** R² Score

### 2. Classification — Income Category Prediction
- **Algorithm:** Random Forest Classifier
- **Features:** Year, Kilometers Driven, Seating Capacity, Owner Income
- **Target:** Income Level (Low / Medium / High)
- **Metric:** Accuracy %

### 3. Clustering — Customer Segmentation
- **Algorithm:** KMeans
- **Features:** Estimated Income, Selling Price
- **Segments:** Budget, Economy, Standard, Mid-Range, Comfort, Premium, Ultra Premium
- **Metrics:** Silhouette Score > 0.9, CV ≤ 15% per class enforced

---

## 🗺️ Interactive Rwanda Map

The EDA page features a fully interactive choropleth map built with **Folium**:

- Hover over any district → see district name, province, and number of clients
- Click any district → popup with full details
- Colour scale from yellow (few clients) → dark red (many clients)
- Free tile background — no API key needed

---

## 📊 Clustering — CV ≤ 15% Per Class Explained

The clustering model is refined using a grid search that enforces a **Coefficient of Variation (CV) of ≤ 15%** for both income and price within every customer segment:

```
CV (%) = (Standard Deviation ÷ Mean) × 100
```

This guarantees that customers inside each segment are genuinely similar to each other — not just mathematically grouped. The model searches over:

- Multiple IQR clip multipliers (tight outlier removal)
- Multiple values of k (number of clusters)
- Picks the configuration with the highest Silhouette Score that still meets the CV requirement

---

## ➕ How to Add a New Page

1. Create a new HTML template in `predictor/templates/predictor/`
2. Add a new view function in `predictor/views.py`
3. Add a new URL path in `predictor/urls.py`
4. Add a navigation link in the sidebar of all existing templates

---

## 🚀 Running on GitHub Codespaces

1. Open this repository on GitHub
2. Click the green **`<> Code`** button
3. Click **`Codespaces`** tab
4. Click **`Create codespace on main`**
5. In the terminal that opens, run:

```bash
pip install -r requirements.txt
python model_generators/regression/train_regression.py
python model_generators/classification/train_classifier.py
python model_generators/clustering/train_cluster.py
python manage.py runserver 0.0.0.0:8000
```

6. Codespaces will show a popup — click **`Open in Browser`**

---

## ⚠️ Known Limitations

- The clustering grid search takes ~2 minutes on first run
- The dataset is synthetic (dummy data) — results are for demonstration purposes
- Django `DEBUG=True` is set — change to `False` before any real deployment
- Models must be retrained if the dataset changes
- No user authentication is implemented

---

## 👨‍🎓 Academic Context

This project was developed as part of a Machine Learning course covering:

- Supervised Learning (Regression & Classification)
- Unsupervised Learning (Clustering)
- Data Preprocessing (Scaling, Outlier Removal, Distribution Transformation)
- Web Application Development with Django
- Geospatial Data Visualisation

---

## 📁 Dataset Columns

| Column | Description |
|--------|-------------|
| `client_name` | Name of the vehicle owner |
| `district` | Rwanda district of the client |
| `year` | Vehicle manufacture year |
| `kilometers_driven` | Total KM driven |
| `seating_capacity` | Number of seats |
| `estimated_income` | Owner's estimated income |
| `selling_price` | Vehicle selling price |
| `income_level` | Income category (Low/Medium/High) |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3 | Core programming language |
| Django | Web framework |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning models |
| Joblib | Model serialisation |
| Folium | Interactive maps |
| Branca | Colour scales for maps |
| GeoPandas | GeoJSON processing |
| Matplotlib | Static charts |
| Plotly | Interactive charts |
| Bootstrap 5 | Frontend styling |

---

*Built with ❤️ as part of a Machine Learning course project*