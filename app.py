import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.write(
    """ 
        # Google Stock Price App

        shown are the stock ***closing price*** and ***volume*** of Google!
    """         
)


google_df = yf.download('GOOGL', start='2020-01-01', end='2021-01-01')


st.write(
    '''
        ### closing price
    '''    
)
st.line_chart(google_df['Close'])

st.write( 
    '''
        ### Volume price
    '''
)
st.line_chart(google_df['Volume'])

st.write(
    """ 
        # Apple Stock Price App

        shown are the stock ***closing price*** and ***volume*** of Apple!
    """         
)


Apple_df = yf.download('AAPL', start='2020-01-01', end='2021-01-01')


st.write(
    '''
        ### closing price
    '''    
)
st.line_chart(Apple_df['Close'])

st.write( 
    '''
        ### Volume price
    '''
)
st.line_chart(Apple_df['Volume'])
