import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors


# function to calculate molecular descriptors
def AromaticProportion(m):
    aromatic_atoms = [
        m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())
    ]
    aa_count = []
    for i in aromatic_atoms:
        if i == True:
            aa_count.append(1)
    AromaticAtom = sum(aa_count)
    HeavyAtom = Descriptors.HeavyAtomCount(m)
    AR = AromaticAtom / HeavyAtom
    return AR


# function to generate molecular descriptors
def generate(smiles, verbose=False):

    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData = np.arange(1, 1)
    i = 0
    for mol in moldata:

        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
        desc_AromaticProportion = AromaticProportion(mol)

        row = np.array(
            [desc_MolLogP, desc_MolWt, desc_NumRotatableBonds, desc_AromaticProportion]
        )

        if i == 0:
            baseData = row
        else:
            baseData = np.vstack([baseData, row])
        i = i + 1

    columnNames = ["MolLogP", "MolWt", "NumRotatableBonds", "AromaticProportion"]
    descriptors = pd.DataFrame(data=baseData, columns=columnNames)

    return descriptors


# APP

image = Image.open("solubility-logo.jpg")

st.image(image, use_column_width=True)
st.write(
    """
        # Molecular Solubility Prediction Web App
        This app predicts the **Solubility (LogS)** values of molecules!
        Data obtained from the John S. Delaney. [ESOL: Estimating Aqueous Solubility Directly from Molecular Structure](https://pubs.acs.org/doi/10.1021/ci034243x). ***J. Chem. Inf. Comput. Sci.*** 2004, 44, 3, 1000-1005.
        ***
    """
)


st.sidebar.header('User Input Features')
def user_input_features():
    smiles = st.sidebar.text_area("SMILES input", "CCCC\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    return smiles

SMILES = user_input_features()
SMILES = 'C\n' + SMILES
SMILES = SMILES.split('\n')
st.header('Input SMILES')
SMILES[1:] # Skips the dummy first item 

st.header('Computed descriptors')
X = generate(SMILES)
X[1:] # Skips the dummy first item


model = pickle.load(open("solubility_model.pkl", "rb"))
prediction = model.predict(X)
prediction[1:] # Skips the dummy first item
