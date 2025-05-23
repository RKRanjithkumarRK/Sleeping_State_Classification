# Sleeping_State_Classification
This project utilizes machine learning to predict diseases based on user-input symptoms. It employs a trained model that analyzes symptom patterns and predicts the most likely disease.
💤 Sleep Stage Classification Using Machine Learning
This project aims to classify human sleep stages using machine learning algorithms. It is designed to support healthcare and research by automating the detection of different sleep stages, which can help in diagnosing sleep disorders.

🔍 Project Overview
Sleep is vital for human health, and detecting disorders early can lead to better treatment. This project uses data collected from individuals (such as sleep patterns, biometrics, and daily habits) to train multiple machine learning models for classifying sleep stages or detecting potential sleep disorders.

The project includes:

A backend model training script (main.py)

A fully functional interactive web app using Streamlit (app.py)

A downloadable trained model (best_sleep_disorder_model.pkl)

Prediction results (model_predictions.csv)

⚙️ Technologies & Tools Used
Programming Language: Python

Libraries:

Data Handling: pandas, numpy

Visualization: matplotlib, seaborn

Machine Learning Models:

Logistic Regression

Decision Tree

Random Forest

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Naive Bayes

CatBoost

Evaluation: accuracy_score, confusion_matrix, roc_curve, classification_report, auc

Model Persistence: joblib

Deployment: Streamlit for the web-based user interface

🧠 Features
Upload Your Dataset: CSV upload with required target column

Automatic Preprocessing: Missing value removal, categorical encoding, and scaling

Multiple Models: Trains and evaluates multiple classifiers

Performance Metrics: Accuracy, classification report, confusion matrix, and ROC curve for each model

Model Selection: Best-performing model saved automatically

Result Export: Download the predictions as a CSV file

Web Interface: Easy-to-use Streamlit app for demonstration or deployment

🗂 File Descriptions
File	Description
main.py	Trains multiple models on the sleep dataset and selects the best one
app.py	Streamlit web app for interactive use with dataset upload and results display
best_sleep_disorder_model.pkl	Best model saved using joblib
model_predictions.csv	Output predictions from each model including the actual values

🚀 How to Run
Run Locally:
bash
Copy
Edit
# Install required packages
pip install pandas numpy scikit-learn catboost matplotlib seaborn streamlit

# Run the Streamlit app
streamlit run app.py
📊 Example Use Case
A user uploads a dataset with patient records, the system processes it and applies all trained models. The user can view detailed performance metrics and download prediction results. The best model is saved for future deployment or API integration.

📌 Future Enhancements
Integration with real-time health tracking devices

Use of deep learning for improved accuracy

Automatic detection of abnormal sleep patterns

REST API for deployment in clinical apps

👤 Author
Your Name

B.Tech in Artificial Intelligence and Data Science

Email: [your_email@example.com]

GitHub: [your_github_profile_link]
