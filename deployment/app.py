import sys
sys.path.insert(0, '../src/')

from libs import *
from data_processing import *
import streamlit as st



#@st.cache()

model = keras.models.load_model('../models/transformer.keras')
scaler = joblib.load('../models/scaler.gz')
with open('../data/raw/meta.json', 'r') as f:
    meta = json.load(f)
features = meta['data_cols'] + meta['error_cols'] + meta['maint_cols']
targets = ['failure', 'maint', 'error', 'anomaly']

window = 24

def prediction(input_data, features, label):
    X = data_transform(
        input_data, features, label, 
        scaler, window, is_label=False
        )
    #preds = model.predict(X)
    #classes = np.argmax(preds, axis = 1)
    classes = model.predict_classes(X, verbose=1, batch_size=128)
    st.write(classes)


def main():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        input = pd.read_csv(uploaded_file)
        #st.write(input)
        prediction(input, features, targets[0])

if __name__ == "__main__":
    main()
