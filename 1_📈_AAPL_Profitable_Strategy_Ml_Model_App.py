
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from tvDatafeed import TvDatafeed, Interval
import yfinance as yf 
import warnings
import logging

plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')

st.set_page_config(
  page_title='SVC Model Strategy App AAPL Data',
  page_icon='📈'
)

st.title('SVC Strategy Buy and Selling App on AAPL Stock Data')
st.sidebar.markdown('# Main Page App')

@st.cache(allow_output_mutation=True)
def get_data():
    #logging.basicConfig(level=logging.DEBUG)
    #tv = TvDatafeed()
    #data = tv.get_hist(symbol='AAPL', exchange='NASDAQ', interval=Interval.in_daily,n_bars = 300)
    aapl = yf.download('AAPL', '2021-11-30')
    aapl = aapl.drop(['Adj Close'], axis=1) 
    return appl

data = get_data()
target = data.iloc[[-1]]
data2 = data[:-1]

# PREPROCESSING DATA
# indepent variables
data2['high-low'] = data2['high'] - data2['low']
data2['open-close'] = data2['open'] - data2['close']

X = data2[['open-close', 'high-low', 'close']]
# store the target variable, if tomorrows close price is greater than todays close price then put 1 else put 0
y = np.where(data2.close.shift(-1) > data2.close, 1, 0)

# get percentage to split the data (90% train 10% test)
percentage_split = 0.9
row = int(data2.shape[0] * percentage_split)

# creating train data set and test data set
X_train = X[:row]
y_train = y[:row]

X_test = X[row:]
y_test = y[row:]

# models creation SVC
from sklearn.svm import SVC

model = SVC()

#train model
model.fit(X_train[['open-close', 'high-low']], y_train)

# make models predictions
data2['pred'] = model.predict(X[['open-close', 'high-low']])

# strategy creation buy and hold 
# calculate daily returns
data2['return'] = data2['close'].pct_change(1)

#calculate the strategy returns
data2['stra_return'] = data2['pred'].shift(1) * data2['return']

# calculate the cumulative returns
data2['cum_ret'] = data2['return'].cumsum()
# calculation of the strategy cumulative reutnrs
data2['cum_stra'] = data2['stra_return'].cumsum()

# visualize and data show

two_plots = plt.figure(figsize=(16,8))
plt.title('Returns')
plt.plot(data2['cum_ret'], color='orange', label='Stock Returns')
plt.plot(data2['cum_stra'], color='purple', label='Strategy Returns')
plt.legend()
st.pyplot(two_plots)

#Processing today's predictions
target['high-low'] = target['high'] - target['low']
target['open-close'] = target['open'] - target['close']

X_input = target[['open-close', 'high-low', 'close']]

target['pred'] = model.predict(X_input[['open-close', 'high-low']])

st.write('# Today\'s model prediction is: {}'.format(target.iloc[0, 8]))

# MODEL INSTRUCTIONS
st.title('Strategy Instructions')

data = {'Model Prediction = 1':['Buy the asset'],
        'Model Prediction = 0':['Sell the asset']}
df = pd.DataFrame(data)
st.table(df)

# GENERAL INFO
st.write('## Model buy/sell Strategy\'s performance is: {:.2f} %'.format(data2['cum_stra'][-1]*100))
st.write('### Simple buy/hold Strategy\'s performance is: {:.2f} %'.format(data2['cum_ret'][-1]*100))

st.write("""
   ## AAPL Closing Price
""")

st.line_chart(data2['close'])

st.write("""
  ## AAPL Asset Volume
""")

st.line_chart(data2['volume'])
