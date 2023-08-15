import streamlit as st
import time
import numpy as np


def training(url_data, url_label, url_meta, **kwargs):
    '''
    1/ Merge all data from urls
    2/ scale and transform data to prepare for training
    3/ save processed file (if needed)
    4/ 
    '''
    # merge all data from url
    X_train, y_train = process(url_data, url_label, url_meta, **kwargs)


def train_main():
    st.title("Train data")
    url_data = st.text_input('url of data: ')
    url_label = st.text_input('url of label: ')
    url_meta = st.text_input('url of meta:')

    st.write('url for data: ', url_data)
    st.write('url for label: ', url_label)
    st.write('url for meta: ', url_meta)


if __name__ == '__main__':
    train_main()