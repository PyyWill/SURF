import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import MinMaxScaler

def load_data(data_path, n_train_hours, batch_size_train=8192*4, batch_size_test=64):
    dataset = read_csv(data_path, header=0, index_col=0)
    def parse_complex_series(series):
        return np.array([np.fromstring(s[1:-1], sep=' ', dtype=complex) for s in series])
    
    demand_mainbus_real = np.stack([parse_complex_series(dataset['demand_mainbus']).real[:, i] for i in range(3)], axis=1)
    demand_mainbus_imag = np.stack([parse_complex_series(dataset['demand_mainbus']).imag[:, i] for i in range(3)], axis=1)
    demand_schlinger_real = np.stack([parse_complex_series(dataset['demand_schlinger']).real[:, i] for i in range(3)], axis=1)
    demand_schlinger_imag = np.stack([parse_complex_series(dataset['demand_schlinger']).imag[:, i] for i in range(3)], axis=1)
    demand_resnick_real = np.stack([parse_complex_series(dataset['demand_resnick']).real[:, i] for i in range(3)], axis=1)
    demand_resnick_imag = np.stack([parse_complex_series(dataset['demand_resnick']).imag[:, i] for i in range(3)], axis=1)
    demand_beckman_real = np.stack([parse_complex_series(dataset['demand_beckman']).real[:, i] for i in range(3)], axis=1)
    demand_beckman_imag = np.stack([parse_complex_series(dataset['demand_beckman']).imag[:, i] for i in range(3)], axis=1)
    demand_braun_real = np.stack([parse_complex_series(dataset['demand_braun']).real[:, i] for i in range(3)], axis=1)
    demand_braun_imag = np.stack([parse_complex_series(dataset['demand_braun']).imag[:, i] for i in range(3)], axis=1)

    input_features = np.hstack([
        demand_mainbus_real,  
        demand_mainbus_imag,  
        demand_schlinger_real,
        demand_schlinger_imag,
        demand_resnick_real,
        demand_resnick_imag,
        demand_beckman_real,
        demand_beckman_imag,
        demand_braun_real,
        demand_braun_imag,
        dataset[['temp', 'humidity', 'windspeed', 'solarradiation']].values 
    ]) # 3*10+4
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_input = scaler.fit_transform(input_features)
    
    n_in = 1
    n_out = 1

    columns_to_drop = []
    for i in range(n_out):
        start_col = (n_in + i) * 34 + 30
        end_col = (n_in + i) * 34 + 33
        columns_to_drop.extend(range(start_col, end_col + 1))

    reframed = series_to_supervised(scaled_input, n_in=n_in, n_out=n_out)
    reframed.drop(reframed.columns[columns_to_drop], axis=1, inplace=True)
    
    output_dim = 30*n_out
    train = reframed.values[:n_train_hours, :]
    test = reframed.values[n_train_hours:, :]
    train_X, train_y = train[:, output_dim:], train[:, :output_dim]
    test_X, test_y = test[:, output_dim:], test[:, :output_dim]
    
    train_X = torch.tensor(train_X).float().unsqueeze(1)
    train_y = torch.tensor(train_y).float()
    test_X = torch.tensor(test_X).float().unsqueeze(1)
    test_y = torch.tensor(test_y).float()

    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    
    return train_loader, test_loader, scaler, train_X, train_y, test_X, test_y, n_in, n_out


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg
