from collections import defaultdict
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st


photo = Image.open("/home/ahmed/Ai/streamlit-for-data-science-deployment/dna-logo.jpg")

st.image(photo, use_column_width=True)
st.write("***")

st.header("DNA Nucleotide Count Web App")
st.write(
    "This is a simple DNA nucleotide count web app to count the nucleotide composition in a given DNA sequence"
)

st.write("***")

sequence = """>DNA Query 2\nGAACACGTGGAGGCAAACAGGAAGGTGAAGAAGAACTTATCCTATCAGGACGGAAGGTCCTGTGCTCGGG\nATCTTCCAGACGTCGCGACTCTAAATTGCCCCCTCTGAGGTCAAGGAACACAAGATGGTTTTGGAAATGC\nTGAACCCGATACATTATAACATCACCAGCATCGTGCCTGAAGCCATGCCTGCTGCCACCATGCCAGTCCT"""
st.header("Please enter the DNA sequence in the text area below:")
sequence = st.text_area("DNA Input Sequence", sequence, height=150)
# reformatting the sequence

sequence = sequence.splitlines()[1:]
sequence = "".join(sequence)

st.write("***")
st.header("Input DNA Sequence")
st.write("The DNA sequence you entered:\n", sequence)

st.write("***")


def count_dna(text):
    count = {
        "A": text.count("A"),
        "T": text.count("T"),
        "G": text.count("G"),
        "C": text.count("C"),
    }

    return count


dna = count_dna(sequence)

st.write("***")
st.write("## DNA Nucleotide Count")
st.subheader('1. Print text')
st.write('There are  ' + str(dna['A']) + ' adenine (A)')
st.write('There are  ' + str(dna['T']) + ' thymine (T)')
st.write('There are  ' + str(dna['G']) + ' guanine (G)')
st.write('There are  ' + str(dna['C']) + ' cytosine (C)')



st.subheader('2. Print dataframe')

df = pd.DataFrame().from_dict(dna, orient='index')
df = df.rename({0: 'count'}, axis='columns')
df.reset_index(inplace=True)
df = df.rename(columns={'index': 'nucleotide'})
df 
st.write("***")
st.header("barchar for DNA Sequence")

fig = plt.figure(figsize=(4, 2))
sns.barplot(x='nucleotide', y='count', data=df, width=0.3, palette='viridis')
st.pyplot(fig=fig)