# Hotel Booking Cancellation Prediction

## Project Overview

This project predicts whether a hotel booking will be cancelled or not using Machine Learning.

A **Random Forest Classifier** is used to predict booking cancellations based on customer and booking-related details.

The project includes a Streamlit frontend and a FastAPI backend to provide real-time predictions.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Classifier
- Streamlit
- FastAPI
- Uvicorn
- Joblib

## Dataset

- **Dataset:** Hotel Booking Demand Dataset
- **Source:** Kaggle
- **Original Dataset:** 119,390 records and 32 columns
- **Dataset Link:** https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

**Note:** The dataset is not included in this repository. Download `hotel_bookings.csv` from Kaggle and place it in the project folder before training the model.

## Machine Learning Workflow

1. Data Loading
2. Data Cleaning
3. Duplicate Removal
4. Handling Missing Values
5. Feature and Target Separation
6. Train-Test Split
7. Data Preprocessing
8. Random Forest Model Training
9. Model Evaluation
10. Saving the ML Pipeline

## Project Structure

```text
Hotel-Booking-Cancellation-Prediction/
│
├── train.py
├── app.py
├── streamlit_app.py
├── requirements.txt
├── cancellation_distribution.png
├── confusion_matrix.png
├── README.md
└── .gitignoregit status
```

The trained model file (`hotel_bookings_pipeline.pkl`) is generated locally by running `train.py`. It is not included in this repository.

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/ChennuPeshmaDeviSree/Hotel-Booking-Cancellation-Prediction.git
```

### 2. Navigate to the Project Folder

```bash
cd Hotel-Booking-Cancellation-Prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the Hotel Booking Demand dataset from Kaggle:

https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

Place the downloaded `hotel_bookings.csv` file in the project folder.

### 5. Train the Model

Run the following command:

```bash
python train.py
```

This trains the Random Forest model and generates the `hotel_bookings_pipeline.pkl` file.

### 6. Run the FastAPI Backend

```bash
python -m uvicorn app:app --reload --port 8000
```

The FastAPI backend will run at:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

### 7. Run the Streamlit Frontend

Open another terminal and execute:

```bash
python -m streamlit run streamlit_app.py
```

The Streamlit application will open in your browser.

## Application Features

- Enter hotel booking details.
- Predict whether a booking will be cancelled.
- Display cancellation probability.
- Interactive Streamlit user interface.
- FastAPI-based prediction backend.

## Model Details

- **Algorithm:** Random Forest Classifier
- **Problem Type:** Binary Classification
- **Target Variable:** `is_canceled`
- **0:** Not Cancelled
- **1:** Cancelled

## Author

**Peshma Devi Sree Chennu**