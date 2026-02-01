# Steel Faults Classification

This project aims to automate the detection and classification of steel plate faults using machine learning. Based on the "Steel Plates Faults" dataset, we classify defects into 7 distinct categories: Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, and Other_Faults.

## 🚀 Features

- **Exploratory Data Analysis (EDA)**: Comprehensive analysis of features, correlations, and distributions.
- **Model Benchmarking**: Comparison of 5 models: Logistic Regression, Random Forest, XGBoost, LightGBM, and AdaBoost.
- **Hyperparameter Tuning**: Optimized XGBoost model using GridSearchCV.
- **Flask API**: High-performance API for real-time predictions.
- **Responsive Dashboard**: A modern web interface for manual data entry and prediction visualization.
- **Dockerized**: Fully containerized for local development and cloud deployment.

## 📁 Project Structure

```text
├── data/
│   └── steel_plates_faults_original_dataset.csv  # Dataset file
├── notebooks/
│   └── eda_and_modeling.ipynb                  # EDA & Model Comparison
├── templates/
│   └── index.html                              # Frontend UI
├── app.py                                      # Flask API
├── train.py                                    # Model training script
├── test.py                                     # API test script
├── model.bin                                   # Saved model & metadata
├── Dockerfile                                  # Container configuration
├── Pipfile                                     # Dependency management
├── Pipfile.lock                                # Locked dependencies
└── README.md                                   # This file
```

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.12
- `pipenv` (Install via `pip install pipenv`)

### 2. Environment Setup
Clone the repository and install dependencies:
```bash
pipenv install
```

### 3. Run Locally
Activate the virtual environment:
```bash
pipenv shell
```
Start the Flask server:
```bash
python app.py
```
Open your browser and navigate to `http://localhost:9696`.

## 📈 Model Performance

The project explores multiple tree-based algorithms. **XGBoost** achieved the best performance after tuning:
- **Baseline Accuracy**: ~79.4%
- **Final Accuracy**: ~79.6% (Tuned)

## 🐳 Docker Deployment

To build and run the application using Docker:

```bash
# Build the image
docker build -t steel-faults .

# Run the container
docker run -p 9696:9696 steel-faults
```

## ☁️ Cloud Deployment

The application is ready for deployment on platforms like **Render**, **Heroku**, or **AWS/Google Cloud (Kubernetes)**.
- **Port**: 9696
- **Runtime**: Docker
- **Entrypoint**: `gunicorn --bind 0.0.0.0:9696 app:app`

---
*Created as part of the ML Zoomcamp 2025 Capstone Project.*
