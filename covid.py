import pandas as pd
import streamlit as st
import plotly.express as px
from fbprophet import Prophet
from fbprophet.plot import plot_plotly

df = pd.read_csv("https://covid19.who.int/WHO-COVID-19-global-data.csv", usecols = ['Date_reported', 'Country', 'New_cases'])
new_df = df.loc[df['Country'] == 'Thailand'].sort_values(by='Date_reported', ascending=False)
st.write(new_df)

fig = px.line(new_df, x="Date_reported", y="New_cases")
fig.layout.update(title_text=" Talk With Covid-19 News",xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

df_train = new_df[['Date_reported', 'New_cases']]
df_train = df_train.rename(columns={"Date_reported": "ds", "New_cases": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=7)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)


st.write('Acc = 0.8042')
