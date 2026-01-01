🧠 ML Data Processing Toolkit

An interactive Streamlit-based Machine Learning application for data exploration, preprocessing, and classification.
This project demonstrates a structured, end-to-end ML workflow with a user-friendly multi-page web interface.

🚀 Features

📊 Data Viewer

Load and explore datasets interactively

Inspect raw data before processing

⚙️ Pre-processing

Apply common preprocessing techniques:

Smoothing

Filtering

Baseline correction

🧠 Classification

Train machine learning classification models

Evaluate model performance using cross-validation

🖥️ Multi-page Streamlit App

Clear navigation using st-pages

Central home dashboard

🗂️ Project Structure
ml/
│
├── configs/
│   └── config.py
│
├── src/
│   ├── streamlit_app/
│   │   ├── index.py
│   │   ├── st_data.py
│   │   ├── st_preprocessing.py
│   │   └── st_classification.py
│   │
│   ├── data/
│   ├── models/
│   └── utils/
│
├── requirements.txt
├── pyproject.toml
├── setup.py
├── README.md
└── .gitignore

⚙️ Installation
git clone https://github.com/nenosoft131/ml.git
cd ml
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

▶️ Run the Application
streamlit run src/streamlit_app/index.py


Open in your browser:

http://localhost:8501

🛠️ Technologies Used

Python

Streamlit

scikit-learn

NumPy & Pandas

Pillow (PIL)

st-pages

🎯 Use Cases

Machine learning prototyping

Data preprocessing workflows

Educational ML demonstrations

Rapid experimentation with classification models

📌 Future Enhancements

Advanced ML models

Dataset upload via UI

Model persistence

Experiment tracking

Docker deployment

🤝 Contributing

Contributions are welcome.
Please open an issue or submit a pull request.

📄 License

MIT License

👤 Author

NenoSoft131
GitHub: https://github.com/nenosoft131
