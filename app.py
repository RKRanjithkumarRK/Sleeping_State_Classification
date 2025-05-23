import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import joblib

st.title('Sleep Disorder Detection')

uploaded_file = st.file_uploader('Upload your dataset (CSV file)', type='csv')

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if 'target' not in data.columns:
        st.error('The dataset must contain a "target" column.')
    else:
        st.write('### First 5 rows of the dataset:')
        st.write(data.head())

        # Handling missing values
        data = data.dropna()

        # Encode categorical columns if present
        label_encoders = {}
        for column in data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le

        # Define features (X) and target (y)
        X = data.drop('target', axis=1)
        y = data['target']

        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Define models
        models = {
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'SVM': SVC(probability=True),
            'KNN': KNeighborsClassifier(),
            'CatBoost': CatBoostClassifier(verbose=0),
            'Naive Bayes': GaussianNB()
        }

        st.write('### Predictions from All Models')
        results = pd.DataFrame(X)
        results['Actual'] = y

        for model_name, model in models.items():
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            accuracy = accuracy_score(y, y_pred) * 100
            st.write(f'{model_name} Accuracy: {accuracy:.2f}%')
            st.write(f'Classification Report for {model_name}:')
            st.text(classification_report(y, y_pred))

            # Confusion Matrix
            cm = confusion_matrix(y, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{model_name} - Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)

            # ROC Curve
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_scaled)[:, 1]
                fpr, tpr, _ = roc_curve(y, y_prob)
                auc_score = auc(fpr, tpr)

                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_title('ROC Curve')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.legend()
                st.pyplot(fig)

            results[f'{model_name}_Prediction'] = y_pred

        # Save results to CSV file
        results.to_csv('model_predictions.csv', index=False)
        st.download_button(label='Download Prediction Results', data=results.to_csv(index=False), file_name='model_predictions.csv', mime='text/csv')

        # Save the best model
        best_model_name = max(models, key=lambda name: accuracy_score(y, models[name].predict(X_scaled)))
        best_model = models[best_model_name]
        joblib.dump(best_model, 'best_sleep_disorder_model.pkl')

        st.write(f'Best model saved as "best_sleep_disorder_model.pkl" ({best_model_name})')
