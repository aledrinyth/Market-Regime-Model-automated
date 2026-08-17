import plotly.io as pio
import streamlit as st

st.set_page_config(layout="wide")
st.title("Market Regime Model Dashboard")

# Streamlit accesses the file locally from the repo folder
fig = pio.read_json("data/chart.json")

st.plotly_chart(fig, use_container_width=True)