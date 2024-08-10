import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import datetime
import matplotlib.pyplot as plt
import scipy.optimize as sco
import numpy as np


st.set_page_config(layout="wide", page_title="App", page_icon="👋")
st.write(
    """
# A Simple Yet Useful Fin-App
"""
)
st.sidebar.success("Select a tool above.")
