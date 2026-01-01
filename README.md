# 🧠 ML Data Processing Toolkit

An interactive **Streamlit-based Machine Learning application** for **data exploration, preprocessing, and classification**.  
This project provides a structured workflow to process datasets, train ML models, and evaluate performance with a user-friendly multi-page web interface.

---

## 🚀 Features

- 📊 **Data Viewer**  
  Explore datasets and inspect raw data interactively.

- ⚙️ **Pre-processing**  
  Apply smoothing, filtering, baseline correction, and other preprocessing steps.

- 🧠 **Classification**  
  Train ML classification models and evaluate performance using **cross-validation**.

- 🖥️ **Multi-page Streamlit App**  
  Easy navigation between **Home**, **Data Viewer**, **Pre-processing**, and **Classification** pages using `st-pages`.

---

## 🗂️ Project Structure

ml/
├── configs/
│ └── config.py # Configuration (paths, constants, assets)
│
├── src/
│ ├── streamlit_app/ # Streamlit multi-page application
│ │ ├── index.py # Main Streamlit entry point
│ │ ├── st_data.py # Data viewer page
│ │ ├── st_preprocessing.py # Pre-processing logic & UI
│ │ └── st_classification.py # Classification & evaluation
│ │
│ ├── data/ # Data handling modules
│ ├── models/ # ML models & training logic
│ └── utils/ # Helper and utility functions
│
├── requirements.txt # Python dependencies
├── pyproject.toml # Project metadata
├── setup.py # Package setup
├── README.md # Project documentation
└── .gitignore # Ignored files




Avoid putting it inline like your first attempt — GitHub will just treat it as normal text.

If you want, I can rewrite the entire README with this properly formatted tree so it looks clean and professional on GitHub.

Do you want me to do that?

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/nenosoft131/ml.git
cd ml


python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
****

