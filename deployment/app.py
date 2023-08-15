import sys
sys.path.append('../src/')

from libs import *
from config import *
from data_processing import *
import streamlit as st

#@st.cache()

model = keras.models.load_model(path_model)
scaler = joblib.load(path_scaler)
with open('../data/raw/meta.json', 'r') as f:
    meta = json.load(f)
features = meta['data_cols'] + meta['error_cols'] + meta['maint_cols']
targets = ['failure', 'maint', 'error', 'anomaly']
params = dict(
    window = 24,
    timerange = 8760,
    split = 7320,
    IDs = 100,
)

def training(url_data, url_label, url_meta, **kwargs):
    '''
    1/ Merge all data from urls
    2/ scale and transform data to prepare for training
    3/ save processed file (if needed)
    4/ 
    '''
    # merge all data from url
    X_train, y_train = process(url_data, url_label, url_meta, **kwargs)


def check_input(input):
    pass


def main():
    st.title("Predictive maintenance Home Page")
    #url_data = st.input_text('url of data: ')
    #url_label = st.input_text('url of label: ')
    #url_meta = st.input_text('url of meta:')


    st.sidebar.success("You are currently viewing main page")


if __name__ == "__main__":
    main()