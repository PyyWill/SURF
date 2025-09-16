import sys
import os
print(os.getcwd())
sys.path.append('.')
from portfolio import synthetic, yfinance

if __name__ == '__main__':
    print('testing synthetic data...')
    loaders, y_info = synthetic.get_loaders(4, 0)
    print('y_info:', y_info)
    print(f"train size: {len(loaders['train'])}, calib size: {len(loaders['calib'])}, test size: {len(loaders['test'])}")
    print('Try loading 1 sample')
    for x, y in loaders['train']:
        print('x', x.shape, 'y', y.shape)
        print(x)
        print(y)
        break    

    print('testing yfinance data...')
    loaders, y_info = yfinance.get_loaders(4, 2012, 0, True)
    print('y_info:', y_info)
    print(f"train size: {len(loaders['train'])}, calib size: {len(loaders['calib'])}, test size: {len(loaders['test'])}")
    print('Try loading 1 sample')
    for x, y in loaders['train']:
        print('x', x.shape, 'y', y.shape)
        print(x)
        print(y)
        break    

