from libs import *
from config import *
from read_data import *


def create_train_test(X, y, ids, split, timeline):
    # Create a train/test data for each machineID
    # First group data for each machine ID
    # For each machine there are 8761 records! 
    # For the sake of simplicity and convenience, we want to take 8760 recs 
    # So that's why we need to modify our loop a bit
    X_by_ids, y_by_ids = [], []
    start, stop = 0, timeline
    for i in range(ids):
        X_by_ids.append(X.iloc[start:stop])
        y_by_ids.append(y.iloc[start:stop])
        # next iteration will increase by 1 to skip the last record
        start = stop + 1 
        stop = start + timeline

    # Train test split for each machineID
    X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = [], [], [], []
    for i in range(ids):
        X_train_tmp.append(X_by_ids[i].iloc[0:split:])
        X_test_tmp.append(X_by_ids[i].iloc[split:timeline:])
        y_train_tmp.append(y_by_ids[i].iloc[0:split:])
        y_test_tmp.append(y_by_ids[i].iloc[split:timeline:])

    # Let's concat them all together!
    X_train = pd.concat(X_train_tmp)
    X_test = pd.concat(X_test_tmp)
    y_train = pd.concat(y_train_tmp)
    y_test = pd.concat(y_test_tmp)

    return X_train, X_test, y_train, y_test


def get_scaling(X_train, features, namesave):
    scaler = MinMaxScaler()
    scaler.fit(X_train[features])
    joblib.dump(scaler, namesave)

    return scaler
    

def feature_engineering(X, window):
    data_matrix = X.values
    n_elements = data_matrix.shape[0]
    # Iterate over two lists in parallel.
    # For example id1 have 192 rows and sequence_length is equal to 50
    # so zip iterate over two following list of numbers (0,142),(50,192)
    # 0 50 -> from row 0 to row 50
    # 1 51 -> from row 1 to row 51
    # 2 52 -> from row 2 to row 52
    # ...
    # 141 191 -> from row 141 to 191
    for start, stop in zip(range(0, n_elements-window), range(window, n_elements)):
        yield data_matrix[start:stop, :]


def labeling(y, window, label):
    data_matrix = y[label].values
    num_elements = data_matrix.shape[0]
    # I have to remove the first seq_length labels
    # because for one id the first sequence of seq_length size have as target
    # the last label (the previus ones are discarded).
    # All the next id's sequences will have associated step by step one label as target. 
    return data_matrix[window:num_elements, :]


def data_transform(data, features, label, scaler, window, is_label=False):
    if not is_label:
        data[features] = scaler.transform(data[features])
        gen_data = [
            list(feature_engineering(data[data['machineID'] == idx], window)) 
            for idx in data['machineID'].unique()
            ]
    else:
        gen_data =[
            list(labeling(data[data['machineID'] == idx], window, [label]))
            for idx in data['machineID'].unique()
        ]

    data_final = np.concatenate(list(gen_data)).astype(np.float32)

    return data_final


def process_raw_data(url_data, url_label, url_meta, **kwargs):
    X, y, features, targets = merge_data(url_data, url_label, url_meta)

    X_train, X_test, y_train, y_test = create_train_test(
        X, y, kwargs['IDs'], kwargs['split'], kwargs['timerange']
        )
    
    scaler = get_scaling(X_train, features, '../models/scaler.gz')
    
    #X_train_final = data_transform(
    #    X_train, features, targets[0],
    #    scaler, window, is_label=False
    #)
    #y_train_final = data_transform(
    #    y_train, features, targets[0],
    #    scaler, window, is_label=True
    #)
    #X_test_final = data_transform(
    #    X_test, features, targets[0],
    #    scaler, window, is_label=False
    #)
    #y_test_final = data_transform(
    #    y_test, features, targets[0],
    #    scaler, window, is_label=True
    #)

    with open(path_proc_X_train, 'wb') as f1:
        pkl.dump(X_train, f1)
    with open(path_proc_X_test, 'wb') as f2:
        pkl.dump(X_test, f2)
    with open(path_proc_y_train, 'wb') as f3:
        pkl.dump(y_train, f3)
    with open(path_proc_y_test, 'wb') as f4:
        pkl.dump(y_test, f4)

    return X_train_final, y_train_final


if __name__ == '__main__':
    url_meta = 'https://s3.us-west-1.amazonaws.com/aitomatic.us/pmfp-data/metadata.json'
    url_data = 'https://s3.us-west-1.amazonaws.com/aitomatic.us/pmfp-data/iot_pmfp_data.feather'
    url_label = 'https://s3.us-west-1.amazonaws.com/aitomatic.us/pmfp-data/iot_pmfp_labels.feather'

    X, y, features, targets = merge_data(url_data, url_label, url_meta)

    window = 24
    timerange = 8760
    split = 7320
    IDs = 100
    X_train, X_test, y_train, y_test = create_train_test(X, y, IDs, split, timerange)
    print(y_train[targets[0]].value_counts())
    print(y_test[targets[0]].value_counts())
    X_train.to_csv('../data/raw/X_train.csv', index=False)
    X_test.to_csv('../data/raw/X_test.csv', index=False)
    y_train.to_csv('../data/raw/y_train.csv', index=False)
    y_test.to_csv('../data/raw/y_test.csv', index=False)
    

    scaler = get_scaling(X_train, features, '../models/scaler.gz')
    X_train_final = data_transform(
        X_train, features, targets[0],
        scaler, window, is_label=False
    )
    y_train_final = data_transform(
        y_train, features, targets[0],
        scaler, window, is_label=True
    )
    X_test_final = data_transform(
        X_test, features, targets[0],
        scaler, window, is_label=False
    )
    y_test_final = data_transform(
        y_test, features, targets[0],
        scaler, window, is_label=True
    )

    print(X.shape)
    print(y.shape)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print('Saving: ')
    
    with open('../data/processed/X_train1.pkl', 'wb') as f1:
        pkl.dump(X_train_final, f1)
    with open('../data/processed/X_test1.pkl', 'wb') as f2:
        pkl.dump(X_test_final, f2)
    with open('../data/processed/y_train1.pkl', 'wb') as f3:
        pkl.dump(y_train_final, f3)
    with open('../data/processed/y_test1.pkl', 'wb') as f4:
        pkl.dump(y_test_final, f4)

    