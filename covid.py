import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
from fbprophet import Prophet
from fbprophet.plot import plot_plotly

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA


df = pd.read_csv("https://covid19.who.int/WHO-COVID-19-global-data.csv", usecols = ['Date_reported', 'Country', 'New_cases'])
new_df = df.loc[df['Country'] == 'Thailand'].sort_values(by='Date_reported', ascending=False)
st.write(new_df)

fig = px.line(new_df, x="Date_reported", y="New_cases")
fig.layout.update(title_text=" Talk With Covid-19 News",xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# df_train = new_df[['Date_reported', 'New_cases']]
# df_train = df_train.rename(columns={"Date_reported": "ds", "New_cases": "y"})

# m = Prophet()
# m.fit(df_train)
# future = m.make_future_dataframe(periods=7)
# forecast = m.predict(future)

# st.subheader('Forecast data')
# st.write(forecast.tail())

# st.write('forecast data')
# fig1 = plot_plotly(m, forecast)
# st.plotly_chart(fig1)

# st.write('forecast components')
# fig2 = m.plot_components(forecast)
# st.write(fig2)

#decomposition = seasonal_decompose(df_log)
model = ARIMA(new_df, order=(3,1,1))
results = model.fit(disp=-1)
    
#forecasting data in the future
prediction_ARIMA_diff = pd.Series(results.fittedvalues, copy = True)
prediction_ARIMA_diff_cumsum = prediction_ARIMA_diff.cumsum() 
prediction_ARIMA_log = pd.Series(new_df['New_cases'].iloc[0], index = new_df.index)
prediction_ARIMA_log = prediction_ARIMA_log.add(prediction_ARIMA_diff_cumsum, fill_value=0)
prediction_ARIMA = np.exp(prediction_ARIMA_log)

#วาดกราฟ
fig2 = plt.figure()
plt.plot(df,color = 'blue', label = 'Original')
plt.plot(prediction_ARIMA,color = 'red', label = 'Rolling mean')
plt.legend(loc = 'best')
plt.title('Prediction')
st.plotly_chart(fig2)