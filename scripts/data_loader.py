import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import MinMaxScaler

def load_data(data_path, n_train_hours, batch_size_train=2, batch_size_test=64):
    dataset = read_csv(data_path, header=0, index_col=0)
    
    def parse_complex_series(series):
        # Parse the string format: "[real1;imag1;real2;imag2;real3;imag3]"
        parsed_list = []
        for i, s in enumerate(series):
            # Remove brackets and split by comma
            values = s[1:-1].split(';')
            # Convert to float
            float_values = [float(v) for v in values]
            parsed_list.append(float_values)
        
        # Convert to numpy array
        parsed = np.array(parsed_list)
        complex_parsed = parsed[:, 0::2] + 1j * parsed[:, 1::2]
        
        return complex_parsed
    
    # Parse all demand variables
    demand_vars = ['demand_mainbus', 'demand_broad', 'demand_schlinger', 'demand_resnick', 'demand_beckman', 'demand_braun']
    demand_data = {}
    
    # Parse other variables
    for var in demand_vars:
        parsed = parse_complex_series(dataset[var])
        demand_data[f"{var}_real"] = parsed.real
        demand_data[f"{var}_imag"] = parsed.imag

    # Check weather features
    weather_cols = ['temp', 'feelslike', 'humidity', 'windspeed', 'solarradiation']
    available_weather = [col for col in weather_cols if col in dataset.columns]
    
    input_features = np.hstack([
        demand_data['demand_mainbus_real'],  
        demand_data['demand_mainbus_imag'],
        demand_data['demand_broad_real'],
        demand_data['demand_broad_imag'],
        demand_data['demand_schlinger_real'],
        demand_data['demand_schlinger_imag'],
        demand_data['demand_resnick_real'],
        demand_data['demand_resnick_imag'],
        demand_data['demand_beckman_real'],
        demand_data['demand_beckman_imag'],
        demand_data['demand_braun_real'],
        demand_data['demand_braun_imag'],
        dataset[available_weather].values 
    ]) # 3*12+5 = 41 features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_input = scaler.fit_transform(input_features)
    all_col, weather_col = scaled_input.shape[1], 5
    
    n_in = 1
    n_out = 1

    columns_to_drop = []
    for i in range(n_out):
        start_col = (n_in + i) * all_col + (all_col-5)
        end_col = (n_in + i) * all_col + (all_col-1)
        columns_to_drop.extend(range(start_col, end_col + 1))

    reframed = series_to_supervised(scaled_input, n_in=n_in, n_out=n_out)
    reframed.drop(reframed.columns[columns_to_drop], axis=1, inplace=True)
    
    train = reframed.values[:n_train_hours, :]
    test = reframed.values[n_train_hours:, :]
    train_X, train_y = train[:, :all_col], train[:, all_col:]
    test_X, test_y = test[:, :all_col], test[:, all_col:]
    
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
