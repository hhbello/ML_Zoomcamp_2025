# Steel Faults Classification

## Problem Description
Steel is one of the most critical materials in modern infrastructure, manufacturing, and construction. Ensuring the quality of steel plates is essential, as defects can compromise structural integrity, safety, and economic efficiency. During production, steel plates may develop faults such as cracks, scratches, or inclusions, which are often difficult to detect and classify accurately using traditional inspection methods. Manual inspection is time-consuming, prone to human error, and costly for large-scale operations.

## The Objective
The goal of this project is to build a machine learning model that can automatically classify steel plate faults into their respective categories based on sensor and inspection data. By providing accurate fault detection and classification, steel manufacturers can: 
- Improve quality control and reduce defective output.
- Minimize production downtime by identifying issues early.
- Lower inspection costs through automation.
- Enhance safety and reliability in downstream applications such as construction and automotive manufacturing.

## Dataset Description
The analysis and modeling are based on industrial inspection data collected from steel plate manufacturing processes. The dataset captures multiple sensor readings and process parameters that are used to identify and classify different types of faults in steel plates.

### Steel Plate Faults Data
  **Shape:** 1,941 records

  **Features:** 
   - `X_Minimum, X_Maximum`: Minimum and maximum x-dimension measurements of the plate.
   - `Y_Minimum, Y_Maximum`: Minimum and maximum y-dimension measurements of the plate.
   - `Pixels_Areas`: Total pixel area of the plate region.
   - `X_Perimeter, Y_Perimeter`: Perimeter measurements along x and y axes.
   - `Sum_of_Luminosity`: Aggregate brightness value of the plate image.
   - `Minimum_of_Luminosity, Maximum_of_Luminosity`: Range of brightness values.
   - `Length_of_Conveyer`: Conveyor belt length during inspection.
   - `TypeOfSteel_A300, TypeOfSteel_A400`: Steel grade indicators.
   - Additional geometric and sensor-derived features describing plate dimensions and inspection conditions.


  **Target Variable:** Fault type classification (multi-class). The dataset includes 7 distinct fault categories such as:
   - `Pastry`
   - `Z_Scratch`
   - `K_Scatch`
   - `Stains`
   - `Dirtiness`
   - `Bumps`
   - `Other_Faults`

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

## Exploratory Data Analysis (EDA)
### Distribution of Fault Types
The bar chart shows the frequency of each fault category in the steel dataset. Other_Faults is the most common, with nearly 700 occurrences, followed by Bumps and K_Scratch around 400 each. Z_Scratch and Pastry appear moderately often, while Stains and Dirtiness are relatively rare, both under 100.

This imbalance highlights that the dataset is skewed toward certain fault types, which is important to consider during model training and evaluation. It suggests the need for techniques such as class weighting or resampling to ensure the model performs well across all categories.

<img width="1009" height="611" alt="image" src="https://github.com/user-attachments/assets/59e310ef-19ad-485f-845e-ad6a01e99eb4" />

### Feature Correlation
Strong positive correlations (dark blue) are visible between geometric attributes such as X_Maximum, X_Perimeter, and Pixels_Areas, reflecting their natural dependence on plate size. Conversely, some features show weak or negative correlations (dark red), indicating they capture distinct aspects of the inspection process.

<img width="1353" height="1148" alt="image" src="https://github.com/user-attachments/assets/2a4f71ef-79a2-49f7-a34b-647f7baa7ae5" />

### Feature Importance
The horizontal bar chart highlights the features that contributed most to the Random Forest model’s predictions. Length_of_Conveyer, LogOfAreas, and Log_X_Index emerge as the top three drivers, each with relatively high importance scores. Other influential variables include Steel_Plate_Thickness, Pixels_Areas, and Sum_of_Luminosity, which capture geometric and brightness characteristics of the steel plates.

<img width="1152" height="861" alt="image" src="https://github.com/user-attachments/assets/ce407847-693f-41c5-b024-e3e156de8cbb" />


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
  <img width="1160" height="707" alt="image" src="https://github.com/user-attachments/assets/26875b98-9f11-4d92-a222-dad86073477f" />


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
*Created as part of the ML Zoomcamp 2025 Capstone Project 3.*
