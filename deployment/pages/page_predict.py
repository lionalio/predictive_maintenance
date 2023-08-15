import sys
sys.path.append('../../src/')

from libs import *
from config import *
from data_processing import *
import streamlit as st

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


def prediction(input_data, features, label):
    X = data_transform(
        input_data, features, label, 
        scaler, params['window'], is_label=False
        )
    preds = model.predict(X)
    classes = np.argmax(preds, axis = 1)
    #classes = model.predict(X, verbose=1, batch_size=128)
    for i in range(X.shape[0]):
        status = 'OK' if classes[i] == 0 else 'Need reparing'
        st.write('predicting machine {} status: {}'.format(i, status))
    #st.write(classes)


def predict_main():
    st.title("Predict from input file")
    #url_data = st.input_text('url of data: ')
    #url_label = st.input_text('url of label: ')
    #url_meta = st.input_text('url of meta:')


    uploaded_file = st.file_uploader("Choose a historical file")
    if uploaded_file is not None:
        input = pd.read_csv(uploaded_file)
        #st.write(input)
        prediction(input, features, targets[0])


if __name__ == '__main__':
    predict_main()