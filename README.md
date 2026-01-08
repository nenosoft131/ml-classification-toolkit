### Classification Toolkit – Streamlit ML Application

## Overview

This project is a Streamlit-based machine learning classification toolkit designed to support data exploration, preprocessing, model training, and classification through an interactive web interface.

It provides an end-to-end workflow for experimenting with classification models, combining a modular backend architecture with an easy-to-use Streamlit frontend. The toolkit is suitable for rapid prototyping, research workflows, and applied machine learning tasks.

### Features

##  📊 Interactive Data Exploration

  - Visualize raw and processed data
  - Inspect features and labels
  - Custom plotting utilities

## 🧹 Data Preprocessing Pipeline

- Reusable preprocessing steps
- Configurable pipelines
- Dataset abstraction for consistency

## 🤖 Classification Models

- Neural network–based classifiers
- Support Vector Machine (SVM) training
- Scripted classification workflows

## 🧩 Modular & Extensible Design
- Clear separation of concerns
- Easy to add new models or preprocessing steps
- Shared utilities and constants

## 🌐 Streamlit Web Application

- Upload and explore datasets
- Run preprocessing and training interactively
- Visualize classification results

## 🛠️ Installation

Follow these steps to set up and run the application locally:

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run streamlit_app/app.py


## Configuration

- Global constants are defined in constants.py
- Dataset handling logic is centralized in common_dataset.py
- Plotting behavior can be customized via utils/plotting.py
- Pipelines can be extended inside the pipeline/ directory

## Acknowledgments

Built with:
- Streamlit for interactive ML apps
- scikit-learn for classical ML models
- PyTorch/TensorFlow for neural networks




