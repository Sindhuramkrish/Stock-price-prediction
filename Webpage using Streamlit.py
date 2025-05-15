import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Title
st.title("Stock Price Analyzer & LSTM Predictor")

# User Input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")

# Load Data
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start='2014-01-01', end='2024-12-31')
    df.reset_index(inplace=True)
    return df

if ticker:
    df = load_data(ticker)
    df['MA100'] = df['Open'].rolling(100).mean()
    df['MA200'] = df['Open'].rolling(200).mean()

    # Plot - Opening Price
    st.subheader("Opening Price Over Time")
    fig1, ax1 = plt.subplots()
    ax1.plot(df['Date'], df['Open'], label='Opening Price', color='blue')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()
    st.pyplot(fig1)

    # Plot - 100 MA
    st.subheader("Opening Price with 100-Day Moving Average")
    fig2, ax2 = plt.subplots()
    ax2.plot(df['Date'], df['Open'], label='Open', color='blue')
    ax2.plot(df['Date'], df['MA100'], label='100 MA', color='red')
    ax2.legend()
    st.pyplot(fig2)

    # Plot - 200 MA
    st.subheader("Opening Price with 200-Day Moving Average")
    fig3, ax3 = plt.subplots()
    ax3.plot(df['Date'], df['Open'], label='Open', color='blue')
    ax3.plot(df['Date'], df['MA200'], label='200 MA', color='green')
    ax3.legend()
    st.pyplot(fig3)

    # Plot - 100 & 200 MA
    st.subheader("Opening Price with 100 & 200-Day Moving Averages")
    fig4, ax4 = plt.subplots()
    ax4.plot(df['Date'], df['Open'], label='Open', color='blue')
    ax4.plot(df['Date'], df['MA100'], label='100 MA', color='red')
    ax4.plot(df['Date'], df['MA200'], label='200 MA', color='green')
    ax4.legend()
    st.pyplot(fig4)

    # LSTM Prediction
    if st.button("Run LSTM Prediction"):
        try:
            model = load_model("lstm_model.h5")
            scaler = joblib.load("scaler.save")

            data = df[['Open']].values
            scaled_data = scaler.transform(data)

            sequence_length = 60
            X = []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i, 0])
            X = np.array(X)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            predictions = model.predict(X)
            predicted_prices = scaler.inverse_transform(predictions)

            actual_prices = data[sequence_length:]

            # Plot
            st.subheader("LSTM Prediction vs Actual Opening Price")
            fig5, ax5 = plt.subplots()
            ax5.plot(actual_prices, label='Actual Price', color='blue')
            ax5.plot(predicted_prices, label='Predicted Price', color='red')
            ax5.legend()
            st.pyplot(fig5)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
