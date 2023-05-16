
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load the trained model
model = joblib.load('regex_prova_input.pkl')

# Define the input columns
input_columns = [ 'rm', 'tax', 'ptratio', 'lstat','mdev']

# Function to make predictions
def make_prediction(features):
    prediction = model.predict([features])[0]
    return round(prediction, 2)

# Function to load the file
def load_file(file):
    if file.type == 'application/vnd.ms-excel':
        df = pd.read_excel(file)
    elif file.type == 'text/csv':
        df = pd.read_csv(file)
    else:
        df = None
    return df

# Function to preprocess the data
def preprocess_data(df):
    df.replace(to_replace='?', value=np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.astype(float)
    return df

# Function to calculate evaluation metrics
def calculate_metrics(true_values, predicted_values):
    r2score = r2_score(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    rmse = mean_squared_error(true_values, predicted_values, squared=False)
    return r2score, mae, mse, rmse

# Main function
def main():
    st.title("Previsione Prezzo Immobili")
    st.header("Quali caratteristiche ha la casa?")
    input_values = []
    for col in input_columns:
        value = st.number_input(f"{col}:", step=0.1)
        input_values.append(value)

    if st.button("Simulazione Previsione"):
        input_data = pd.DataFrame([input_values], columns=input_columns)
        prediction = make_prediction(input_data.values[0])
        st.success(f"Il prezzo previsto è: {prediction}")

    st.header("Carica il file con i dati immobiliari")
    uploaded_file = st.file_uploader("Trascina qui il file", type=["xlsx", "csv"])

    if uploaded_file is not None:
        df = load_file(uploaded_file)
        if df is not None:
            if 'medv' in df.columns:
                df.rename(columns={'medv': 'price'}, inplace=True)
            df = preprocess_data(df)
            st.subheader("Dati immobiliari")
            st.dataframe(df.head())
            y_true = df['price']
            X = df[input_columns]
            y_pred = model.predict(X)
            r2score, mae, mse, rmse = calculate_metrics(y_true, y_pred)
            st.subheader("Metriche di valutazione")
            st.text(f"R2 Score: {round(r2score, 4)}")
            st.text(f"MAE: {round(mae, 2)}")
            st.text(f"MSE: {round(mse, 2)}")
            st.text(f"RMSE: {round(rmse, 2)}")
        else:
            st.error("Errore durante il caricamento del file")

if __name__ == '__main__':
    main()
