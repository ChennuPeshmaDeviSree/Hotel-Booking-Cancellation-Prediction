
# Hotel Booking Cancellation Prediction

## Project Overview
This project predicts whether a hotel booking will be cancelled or not using Machine Learning.

A Logistic Regression model is used to predict booking cancellation based on customer and booking-related details.

The project includes a Streamlit frontend and a FastAPI backend for real-time predictions.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression
- Streamlit
- FastAPI
- Uvicorn
- Joblib

## Dataset
- Dataset: Hotel Booking Demand Dataset
- Source: Kaggle
- Original dataset contains 119,390 records and 32 columns.

Dataset Link:
https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

## Machine Learning Workflow
1. Data Loading
2. Data Cleaning
3. Duplicate Removal
4. Handling Missing Values
5. Feature and Target Separation
6. Train-Test Split
7. Data Preprocessing
8. Logistic Regression Model Training
9. Model Evaluation
10. Saving the ML Pipeline

## Project Structure

```text
Hotel-Booking-Cancellation-Prediction/
│
├── app.py
├── streamlit_app.py
├── train.py
├── hotel_bookings_pipeline.pkl
├── requirements.txt
└── README.md
```

## How to Run the Project

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to Project Folder

```bash
cd Hotel-Booking-Cancellation-Prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run FastAPI Backend

```bash
python -m uvicorn app:app --reload --port 8000
```

### 5. Run Streamlit Frontend

Open another terminal and execute:

```bash
python -m streamlit run streamlit_app.py
```

## Application Features
- Enter hotel booking details.
- Predict booking cancellation.
- Display cancellation probability.
- Streamlit-based user interface.
- FastAPI-based prediction backend.

## Model
- Algorithm: Logistic Regression
- Target Variable: is_canceled
- 0: Not Cancelled
- 1: Cancelled

## Author
Peshma Devi Sree Chennu
