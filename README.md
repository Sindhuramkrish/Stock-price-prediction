Real-Time Stock Price Prediction using LSTM

Overview:

This project predicts the opening price of stock using an LSTM model. We collected, cleaned, and visualized historical stock data from 2014 to 2024, and trained an LSTM to forecast future prices.

Tools and Libraries:

Python
Jupyter Notebook
yfinance, pandas, numpy, matplotlib, seaborn
scikit-learn, tensorflow


Workflow

1. Data Collection: Downloaded AAPL stock data using yfinance.
2. Preprocessing: Cleaned data, removed unwanted columns, handled missing values.
3. Visualization: Plotted trends and moving averages (100-day, 200-day).
4. Outlier Detection: Identified using IQR and Z-score methods.
5. Normalization: Scaled 'Open' prices between 0 and 1.
6. Model Building: Trained an LSTM model with two layers and dropout.
7. Evaluation: Compared actual vs predicted prices using RMSE and MAE.



How to Run:

 Install required libraries.
 pip install numpy pandas matplotlib seaborn scipy scikit-learn tensorflow yfinance
 
 Then run the Jupyter Notebook file.

Results:

Good prediction accuracy with low RMSE and MAE.

Predicted prices closely follow actual prices.
