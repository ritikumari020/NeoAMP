# Web app
import streamlit as st
from annotated_text import annotated_text
from millify import millify
from dotenv import load_dotenv
import plotly.express as px
import openai  # OpenAI API
import os
import pickle
from PIL import Image
import zipfile
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Load API key securely
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
if OPENAI_API_KEY:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    st.error("Error: OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable.")

# Function to query OpenAI for disease prediction
def predict_disease_from_peptide(peptide_sequence):
    if not OPENAI_API_KEY:
        return "Error: OpenAI API key is missing."

    prompt = f"Given the antimicrobial peptide '{peptide_sequence}', return only the name of the disease or condition it is known to treat. If no known disease is found, return a common treatable disease by antimicrobial peptides. Respond in just a few words."

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],  # Removed the system role
            max_tokens=15,  # Limit token count to ensure a short response
            temperature=0.5  # Reduce randomness for more focused answers
        )
        return response.choices[0].message.content.strip()  # Strip any extra spaces or formatting

    except openai.OpenAIError as e:
        return f"OpenAI API Error: {e}"
    except Exception as e:
        return f"Unexpected Error: {e}"


# General options
im = Image.open("favicon.ico")
st.set_page_config(
    page_title="AMPredST",
    page_icon=im,
    layout="wide",
)

# Attach customized CSS style
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Set the session state to store the peptide sequence
if 'peptide_input' not in st.session_state:
    st.session_state.peptide_input = ''

# Input peptide
st.sidebar.subheader('Input peptide sequence')

def insert_active_peptide_example():
    st.session_state.peptide_input = 'LLNQELLLNPTHQIYPVA'

def insert_inactive_peptide_example():
    st.session_state.peptide_input = 'KSAGYDVGLAGNIGNSLALQVAETPHEYYV'

def clear_peptide():
    st.session_state.peptide_input = ''

peptide_seq = st.sidebar.text_input('Enter peptide sequence', st.session_state.peptide_input, key='peptide_input', help='Be sure to enter a valid peptide sequence')

st.sidebar.button('Example of an active AMP', on_click=insert_active_peptide_example)
st.sidebar.button('Example of an inactive peptide', on_click=insert_inactive_peptide_example)
st.sidebar.button('Clear input', on_click=clear_peptide)

if st.session_state.peptide_input:
    st.subheader('⚛️ Input Peptide:')
    st.info(peptide_seq)

    # Predict disease for the peptide
    st.subheader('🧬 Predicted Disease(s):')
    disease_prediction = predict_disease_from_peptide(peptide_seq)
    st.success(disease_prediction)

    # General properties of the peptide
    st.subheader('General properties of the peptide')
    analysed_seq = ProteinAnalysis(peptide_seq)
    mol_weight = analysed_seq.molecular_weight()
    aromaticity = analysed_seq.aromaticity()
    instability_index = analysed_seq.instability_index()
    isoelectric_point = analysed_seq.isoelectric_point()
    charge_at_pH = analysed_seq.charge_at_pH(7.0)
    gravy = analysed_seq.gravy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Molecular weight (kDa)", millify(mol_weight/1000, precision=3))
    col2.metric("Aromaticity", millify(aromaticity, precision=3))
    col3.metric("Isoelectric point", millify(isoelectric_point, precision=3))

    col4, col5, col6 = st.columns(3)
    col4.metric("Instability index", millify(instability_index, precision=3))
    col5.metric("Charge at pH 7", millify(charge_at_pH, precision=3))
    col6.metric("Gravy index", millify(gravy, precision=3))

    # Bar plot of the amino acid composition
    count_amino_acids = analysed_seq.count_amino_acids()
    st.subheader('Amino Acid Composition')

    df_amino_acids = pd.DataFrame.from_dict(count_amino_acids, orient='index', columns=['count'])
    df_amino_acids['aminoacid'] = df_amino_acids.index
    df_amino_acids['count'] = df_amino_acids['count'].astype(int)

    plot = px.bar(df_amino_acids, y='count', x='aminoacid',
                  text_auto='.2s', labels={"count": "Count", "aminoacid": "Amino Acid"})
    plot.update_traces(textfont_size=12, textangle=0, textposition="outside", showlegend=False)
    st.plotly_chart(plot)

    # Load the best ML model
    zip_file_name = 'ExtraTreesClassifier_maxdepth50_nestimators200.zip'
    model_file = 'ExtraTreesClassifier_maxdepth50_nestimators200.bin'

    @st.cache_resource
    def unzip_model(zip_file_name):
        with zipfile.ZipFile(zip_file_name, 'r') as file:
            file.extractall()

    @st.cache_resource
    def load_model(model_file):
        with open(model_file, 'rb') as f_in:
            return pickle.load(f_in)

    unzip_model(zip_file_name)
    model = load_model(model_file)

    # Predict AMP activity
    amino_acids_percent = {key: value * 100 for key, value in analysed_seq.get_amino_acids_percent().items()}
    df_amino_acids_percent = pd.DataFrame.from_dict(amino_acids_percent, orient='index', columns=['frequencies']).T
    y_pred = model.predict_proba(df_amino_acids_percent)[0, 1]
    active = y_pred >= 0.5

    st.subheader('Prediction of AMP Activity using Machine Learning')
    if active:
        annotated_text(("The input molecule is an active AMP", "", "#b6d7a8"))
        annotated_text("Probability of antimicrobial activity: ", (f"{y_pred}", "", "#b6d7a8"))
    else:
        annotated_text(("The input molecule is not an active AMP", "", "#ea9999"))
        annotated_text("Probability of antimicrobial activity: ", (f"{y_pred}", "", "#ea9999"))

    # Overview
    st.subheader('Peptide Overview')
    st.write('AMPs are studied for various therapeutic applications, including antibiotic-resistant infections and immune modulation.')
