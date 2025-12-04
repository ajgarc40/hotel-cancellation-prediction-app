# Hotel Cancellation Prediction App

Streamlit application and ML pipeline for predicting hotel booking cancellations for a hotel partner, created as a final project for CIS 508 - Machine Learning in Business.

## 1. Business Problem

Hotels lose money and operational efficiency when guests cancel late or never show.  

This project builds a model that predicts the probability that an upcoming booking will be cancelled, so the hotel can:

- Overbook intelligently when appropriate
- Prioritize outreach to high risk reservations
- Adjust pricing or policies for risky segments

Target variable:  
`is_canceled` equals 1 if the booking was cancelled, 0 otherwise.

## 2. Data

Dataset: `hotel_bookings.csv` (Kaggle style hotel booking dataset)

Examples of features used:

- Booking info: lead time, arrival date, nights, stays in weekend nights, stays in week nights
- Guest profile: number of adults, children, babies, repeated guest flag
- Channel and market: distribution channel, market segment, deposit type
- Historical behavior: previous cancellations, previous bookings not cancelled
- Pricing and allocation: average daily rate, required car parking spaces, reserved room type

Basic cleaning and preprocessing happen inside the modeling notebook (Databricks) and are mirrored in the training script.

## 3. Modeling Approach

This project is framed as a binary classification task.

Models explored in Databricks and logged with MLflow include:

- Logistic regression (baseline)
- Tree based models (Decision Tree, Random Forest, Gradient Boosting, XGBoost)
- k Nearest Neighbors
- Naive Bayes
- Support Vector Machine
- Neural network
- Ensemble methods

For each model, a small hyperparameter grid was defined and evaluated using F1 score and ROC AUC on a validation or test split.  

The final deployed model is a tuned tree based model that performed well on F1 and AUC and is practical for deployment.

## 4. Streamlit Application

The Streamlit app (`app.py`) provides a simple interface where a hotel manager can:

- Enter booking details (lead time, party size, stay length, ADR, etc)
- Submit the information to the model
- See:
  - Predicted probability of cancellation
  - A human readable label such as “Likely to cancel” or “Likely to show”
  - A short explanation of the prediction

The app loads a trained model that is created by the training script below.

## 5. Training Script

`train_app_model_local.py`:

- Loads `hotel_bookings.csv`
- Applies the same preprocessing used in the notebook (feature engineering, encoding, scaling, train or test split)
- Trains the chosen “best” model for deployment
- Saves the trained pipeline to a local file (for example `model.pkl`) that is loaded by the Streamlit app

Note: the large model artifact is not stored in this repository because of GitHub’s file size limits. It is generated locally by running the training script.

## 6. How to Run Locally

### 6.1. Clone the repo

```bash
git clone git@github.com:ajgarc40/hotel-cancellation-prediction-app.git
cd hotel-cancellation-prediction-app
