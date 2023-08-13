from libs import *


def read_data_from_url(url):
    resp = requests.get(url, stream=True)
    resp.raw.decode_content = True
    mem_fh = io.BytesIO(resp.raw.read())
    df = pd.read_feather(mem_fh)

    return df


def read_metadata_from_url(url):
    response = requests.get(url)
    data_json = response.json()

    return data_json


def read_data_from_local(path):
    return pd.read_feather(path)


def merge_data(url_data, url_label, url_meta):
    data = read_data_from_url(url_data)
    label = read_data_from_url(url_label)
    meta = read_metadata_from_url(url_meta)

    with open('../data/raw/meta.json', 'w') as f:
        json.dump(meta, f)

    tmp = data[[meta['time_col']] + meta['data_cols'] + meta['label_cols']]

    all_data = tmp.merge(label, on=['machineID', 'datetime'], how='inner')
    features = meta['data_cols'] + meta['error_cols'] + meta['maint_cols']
    targets = ['failure', 'maint', 'error', 'anomaly']

    # X still use machineID
    X = all_data[all_data.columns.difference(targets)]
    X.drop(['datetime', 'model'], axis=1, inplace=True)
    y = all_data[['machineID'] + targets]

    return X, y, features, targets


if __name__ == '__main__':
    url_meta = 'https://s3.us-west-1.amazonaws.com/aitomatic.us/pmfp-data/metadata.json'
    print(read_metadata_from_url(url_meta))
    url_data = 'https://s3.us-west-1.amazonaws.com/aitomatic.us/pmfp-data/iot_pmfp_data.feather'
    data = read_data_from_url(url_data)
    url_label = 'https://s3.us-west-1.amazonaws.com/aitomatic.us/pmfp-data/iot_pmfp_labels.feather'
    label = read_data_from_url(url_label)

    print(data.columns)
    print(data.shape)
    print(label.columns)
    print(label.shape)

    all_data = data.merge(label, on=['machineID', 'datetime'], how='inner')
    print(all_data.iloc[0:8762])

    print(all_data.columns)
