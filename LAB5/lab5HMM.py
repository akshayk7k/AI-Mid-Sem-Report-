import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import scale

def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def calculate_daily_returns(data):
    data['Returns'] = data['Adj Close'].pct_change().dropna()
    return data[1:]

def fit_hmm(returns, n_states=2):
    returns = returns.reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
    model.fit(returns)
    hidden_states = model.predict(returns)
    return model, hidden_states

def analyze_hidden_states(model):
    means = model.means_
    variances = [np.diag(cov) for cov in model.covars_]
    for i, (mean, var) in enumerate(zip(means, variances)):
        print(f"Hidden State {i + 1}:")
        print(f"Mean Return: {mean[0]:.4f}")
        print(f"Variance: {var[0]:.4f}")
        print()

def plot_hidden_states(data, hidden_states):
    fig, ax = plt.subplots(2, figsize=(10, 6))
    ax[0].plot(data.index, data['Adj Close'], label='Adjusted Close Price')
    ax[0].set_title('Stock Price')
    ax[0].set_ylabel('Price')
    ax[0].legend()
    ax[1].plot(data.index, hidden_states, label='Hidden States', color='orange')
    ax[1].set_title('Hidden States (Market Regimes)')
    ax[1].set_ylabel('Hidden State')
    ax[1].legend()
    plt.tight_layout()
    plt.show()

def plot_transition_matrix(model):
    transition_matrix = model.transmat_
    print("Transition Matrix:")
    print(transition_matrix)

def predict_future_state(model, returns):
    latest_return = returns[-1].reshape(-1, 1)
    predicted_state = model.predict(latest_return)[0]
    print(f"Predicted Future State: {predicted_state}")
    return predicted_state

if __name__ == "__main__":
    ticker = 'AAPL'
    start_date = '2010-01-01'
    end_date = '2023-01-01'
    data = download_data(ticker, start_date, end_date)
    data = calculate_daily_returns(data)
    returns = data['Returns'].dropna().values
    model, hidden_states = fit_hmm(returns, n_states=2)
    analyze_hidden_states(model)
    plot_hidden_states(data, hidden_states)
    plot_transition_matrix(model)
    predict_future_state(model, returns)
