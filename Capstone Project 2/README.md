<img width="1389" height="489" alt="image" src="https://github.com/user-attachments/assets/e415f2af-06d4-4a58-be26-9c8d43c59364" /># Solar Power Generation Prediction

## Problem Description
Solar energy is a cornerstone of renewable energy transitions world-wide. However, solar power generation is inherently variable, depending heavily on weather conditions such as temperature, irradiation, and cloud cover. This inconsistency poses significant challenges for power grid operators who must balance electricity supply and demand in real-time.

### The Objective
The goal of this project is to develop a predictive model that can accurately forecast the **AC Power** output of a solar plant based on weather sensor data. By providing reliable predictions, plant operators can:
- Improve grid stability and energy distribution planning.
- Optimize maintenance schedules for inverters and solar modules.
- Enhance the economic viability of solar energy projects.

## Dataset Description
The analysis and modeling are based on two primary datasets collected over a 34-day period with 15-minute intervals from a power plant in india.

### 1. Generation Data
- **Shape**: 68,778 records
- **Features**:
  - `DATE_TIME`: Date and time for each observation.
  - `DC_POWER`: Direct Current power (kW).
  - `AC_POWER`: Alternating Current power (kW) - **Target Variable**.
  - `DAILY_YIELD`: Cumulative yield for the day.
  - `TOTAL_YIELD`: Total cumulative yield for the inverter.

### 2. Weather Sensor Data
- **Shape**: 3,182 records
- **Features**:
  - `AMBIENT_TEMPERATURE`: Air temperature around the plant.
  - `MODULE_TEMPERATURE`: Temperature recorded at the solar panel module.
  - `IRRADIATION`: Amount of solar radiation hitting the panels.

## Modeling and Evaluation
Multiple models were compared:
1. Linear Regression (Baseline)
2. Random Forest Regressor
3. XGBoost Regressor (Gradient Boosting)
4. Deep Neural Network (Keras/TensorFlow)

The Linear Regressor fits a straight line to predict outcomes based on input features; simple and interpretable.
Random Forest Regressor uses many decision trees and averages their predictions; reduces overfitting and handles non‑linear data well.

XGBoost Regressor builds trees sequentially, correcting errors step by step; highly efficient and powerful for structured data.

Deep Neural Network (Keras/TensorFlow) builds layers of interconnected neurons to learn complex patterns; excels at capturing non‑linear relationships in large datasets.

### **Hyperparameter Tuning and Model Selection:**
All the models had their parameters tweaked slightly from their base form besides the linear regression model. 

The Random Forest Model had the same performance with the base model and tuned model, showing no signs of improvement from it's base score which was already a very good.

The base DNN model had the best performance of the neural networks with R<sup>2</sup> 0.985 and RMSE 54.88kW but did not perform as well as the tree-based models.
<img width="1389" height="489" alt="image" src="https://github.com/user-attachments/assets/97a9fcbe-e65c-4f7a-b458-ac6284f54311" />

The DNN with dropouts was the most unstable when tracking the training and validation loss with R<sup>2</sup> 0.9729 and RMSE 64.64kW.
<img width="1389" height="489" alt="image" src="https://github.com/user-attachments/assets/f5e153ee-0b4a-4031-a689-a75c201bfdba" />

The DNN without dropouts had the best perfomance of the three variations with R<sup>2</sup> 0.9834 and RMSE 50.69kW. Without the dropouts, the model might tend to over fit to the training data.
<img width="1389" height="489" alt="image" src="https://github.com/user-attachments/assets/e30e9693-00e8-44e6-9f43-51397cc6e27d" />


Comparing all the models analyzed, the tuned XGB Model had a slightly better performance than the random forest model and was selected as the best model with and R<sup>2</sup> score of 0.9865.

<img width="1389" height="590" alt="image" src="https://github.com/user-attachments/assets/21135f59-b51b-413c-b23e-58039431a26b" />



## Technical Stack
- **Data Engineering**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`, `xgboost`
- **Deployment**: `Flask`, `Gunicorn`, `Docker`


## How to Run

### Local Setup (using Pipenv)
1. Ensure you have `pipenv` installed (`pip install pipenv`).
2. Install dependencies:
   ```bash
   pipenv install
   ```
3. Run the training script:
   ```bash
   pipenv run python train.py
   ```
4. Start the prediction service:
   ```bash
   pipenv run python app.py
   ```
   ***Note:** The keras/tensorflow dependencies are very large, and are only used in the notebook to evaluate the model performance, as such it can be excluded from the pipfile, and all cells that do not require       the tensorflow will run.*

### Docker Deployment
1. Build the image:
   ```bash
   docker build -t solar-prediction .
   ```
2. Run the container:
   ```bash
   docker run -it --rm -p 9696:9696 solar-prediction
   ```
   *Note: Or use `docker compose up --build` if docker-compose is configured.*

---

## Repository Structure
- `train.py`: Script for data processing, model training, and saving.
- `app.py`: Flask application providing the prediction API.
- `predict_test.py`: Client script for verifying the API.
- `data/`: Contains the raw generation and weather sensor CSVs.
- `Dockerfile`: Containerization configuration.
- `Pipfile`/`Pipfile.lock`: Dependency management.
