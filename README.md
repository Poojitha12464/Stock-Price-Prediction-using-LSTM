# Stock-Price-Prediction-using-LSTM
📈 Stock Price Prediction with LSTM (PyTorch)
A time series forecasting project that uses a Long Short-Term Memory (LSTM) neural network to predict stock closing prices. Built with PyTorch and trained on historical data fetched via yfinance.

🗂️ Project Structure
stock_lstm/
├── data.py            # Data download and preprocessing
├── model.py           # LSTM model architecture
├── train.py           # Training loop with early stopping
├── evaluate.py        # Metrics and visualizations
├── predict.py         # Future price forecasting
├── best_model.pth     # Saved model weights (generated after training)
├── requirements.txt   # Python dependencies
└── README.md

⚙️ Installation
1. Clone the repo
bashgit clone https://github.com/your-username/stock-lstm.git
cd stock-lstm
2. Create a virtual environment (recommended)
bashpython -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
3. Install dependencies
bashpip install -r requirements.txt
requirements.txt
torch
yfinance
pandas
numpy
scikit-learn
matplotlib

🚀 Quick Start
Run each script in order:
bashpython data.py        # Download and preprocess data
python train.py       # Train the LSTM model
python evaluate.py    # Evaluate on test set and plot results
python predict.py     # Generate 30-day future forecast

📋 Pipeline Overview
Raw Prices → Normalize [0,1] → Sliding Windows → LSTM → Predict → Evaluate → Forecast
StepScriptDescription1data.pyDownload 5 years of stock data via yfinance2data.pyScale prices to [0, 1] using MinMaxScaler3data.pyCreate 60-day sliding window sequences4model.pyDefine 2-layer stacked LSTM + Linear head5train.pyTrain with Adam optimizer, MSE loss, LR scheduler6evaluate.pyCompute RMSE/MAE, plot actual vs predicted7predict.pyAutoregressive 30-day future forecast

🧠 Model Architecture
Input: (batch_size, 60, 1)        ← 60 days, 1 feature (close price)
         ↓
LSTM Layer 1  (hidden=64, dropout=0.2)
         ↓
LSTM Layer 2  (hidden=64)
         ↓
Last hidden state: (batch_size, 64)
         ↓
Linear Layer: 64 → 1
         ↓
Output: predicted next-day price
Key Hyperparameters
ParameterDefaultDescriptionseq_len60Number of past days used as inputhidden_size64LSTM memory capacity per cellnum_layers2Stacked LSTM depthdropout0.2Regularization between layersbatch_size32Samples per gradient updatelearning_rate1e-3Adam optimizer LRepochs50Max training epochs

📊 Data Split
SplitProportionPurposeTrain70%Fit model weightsValidation15%Tune hyperparameters, early stoppingTest15%Final unbiased evaluation

⚠️ Data is split chronologically — never shuffled. Shuffling leaks future data into training.


📉 Training
The training loop includes:

Adam optimizer with a learning rate of 1e-3
ReduceLROnPlateau scheduler — halves LR if val loss plateaus for 5 epochs
Gradient clipping (max_norm=1.0) to prevent exploding gradients
Best model checkpoint — saves weights at lowest validation loss

Epoch  10 | Train Loss: 0.004821 | Val Loss: 0.005103
Epoch  20 | Train Loss: 0.002314 | Val Loss: 0.002891
Epoch  30 | Train Loss: 0.001205 | Val Loss: 0.001674
...

📈 Evaluation Metrics
MetricFormulaMeaningRMSE√(mean((actual - pred)²))Penalizes large errors moreMAEmean(|actual - pred|)Average dollar error
A MAE of $2–5 on a $150 stock (~1–3%) is a solid baseline for this architecture.

🔮 Future Forecast
predict.py performs autoregressive prediction — each predicted price is fed back as the next input:
[day1 ... day60] → predict day61
[day2 ... day61] → predict day62
...

⚠️ Uncertainty compounds over time. Forecast accuracy degrades beyond ~5–10 days. For production use, consider retraining daily and adding confidence intervals.


🛠️ Customisation
Change the stock ticker
python# data.py
df = yf.download("TSLA", start="2019-01-01", end="2024-01-01")
Add more features (multi-variate LSTM)
python# data.py — include volume and price columns
df = df[['Close', 'Volume', 'High', 'Low']]   # input_size becomes 4

# model.py — update input_size
model = StockLSTM(input_size=4)
Predict returns instead of raw prices
python# data.py — use % daily change (more stationary)
df['Return'] = df['Close'].pct_change().dropna()
Tune the sequence length
pythonSEQ_LEN = 120    # look back 120 days instead of 60

📦 Dependencies
LibraryVersionUsetorch≥ 2.0LSTM model, trainingyfinance≥ 0.2Stock data downloadpandas≥ 2.0Data handlingnumpy≥ 1.24Numerical opsscikit-learn≥ 1.3MinMaxScaler, metricsmatplotlib≥ 3.7Plotting

⚠️ Disclaimer
This project is for educational purposes only. Stock price prediction is inherently uncertain. Do not use model outputs for real investment decisions.
