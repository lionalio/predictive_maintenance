import os

PATH_RAW_DATA = '../data/raw/'
PATH_PROCESSED_DATA = '../data/processed/'
PATH_MODEL = '../models/'

name_scaler = 'scaler.gz'
path_scaler = os.path.join(PATH_MODEL, name_scaler)

name_model = 'transformer.keras'
path_model = os.path.join(PATH_MODEL, name_model)

path_proc_X_train = os.path.join(PATH_PROCESSED_DATA, 'X_train.pkl')
path_proc_X_test = os.path.join(PATH_PROCESSED_DATA, 'X_test.pkl')
path_proc_y_train = os.path.join(PATH_PROCESSED_DATA, 'y_train.pkl')
path_proc_y_test = os.path.join(PATH_PROCESSED_DATA, 'y_test.pkl')