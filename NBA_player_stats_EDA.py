import pandas as pd
import base64
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.title("NBA Player Stats Explorer")
st.write(
    """
        This app performs simple webscraping of NBA player stats data!    
        
        Used libraries: pandas, streamlit, base64, seaborn, matplotlib
        
        Data source: [Basketball-reference.com](https://www.basketball-reference.com/).
    """
)
# web scraping


st.sidebar.header('User input')
selected_year = st.sidebar.selectbox("Year", list(reversed(range(1950, 2024))))


@st.cache
def load_data(year):
    url = (
        "https://www.basketball-reference.com/leagues/NBA_"
        + str(year)
        + "_per_game.html"
    )
    html = pd.read_html(url, header=0)
    df = html[0]
    raw = df.drop(df[df.Age == "Age"].index)  # Deletes repeating headers in content
    raw = raw.fillna(0)
    playerstats = raw.drop(["Rk"], axis=1)
    return playerstats


playerstats = load_data(selected_year)

teams = sorted(playerstats.Tm.unique())
position = sorted(playerstats.Pos.unique())

selected_team = st.sidebar.multiselect("Team", teams, teams)
selected_pos = st.sidebar.multiselect("Position", position, position)

#filtering data based on user input
st.header("Display Player Stats of Selected Team(s)")
selected_data = playerstats[ (playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

row, cols = selected_data.shape
st.write("Table contains " + str(row) + " Row and " + str(cols) + " Columns")
st.write(selected_data)



def donwload_file(data):
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown(donwload_file(selected_data), unsafe_allow_html=True)


# create a heatmap

button = st.button("Generate Heatmap")

if button:
    selected_data.to_csv('selected_data.csv', index=False)
    df = pd.read_csv('selected_data.csv')
    
    data= pd.DataFrame()
    
    for col in df.columns:
        if df[col].dtype != 'object':
            data[col] = df[col]
            
    corr = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=False)
    
    st.pyplot()    