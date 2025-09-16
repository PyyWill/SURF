import os

import numpy as np
from torch import Tensor

from models.picnn_e2e import optimize

from models.picnn import (
    PICNN,
    fit_picnn_FALCON,
    conformal_q
)
from storage.data import (
    get_loaders,
    get_train_calib_split,
    get_traincalib_test_split,
    load_data
)
from storage.problems import StorageProblemPICNN


INPUT_DIM = 101
Y_DIM = 24
MAX_EPOCHS = 100
BATCH_SIZE = 256
OUT_DIR = 'out/storage_picnn/'
os.makedirs(OUT_DIR, exist_ok=True)

L2REG = 1e-4


def get_tensors(log_prices: bool) -> tuple[dict[str, Tensor], str | tuple[np.ndarray, np.ndarray]]:
    # Load data
    X, Y = load_data()
    y_info: str | tuple[np.ndarray, np.ndarray]
    if log_prices:
        Y = np.log(Y)
        y_info = 'log'
    else:
        Y_mean = Y.mean(axis=0)
        Y_std = Y.std(axis=0)
        Y = (Y - Y_mean) / Y_std
        y_info = (Y_mean, Y_std)
    tensors = get_traincalib_test_split(X, Y, shuffle=True)
    return tensors, y_info


if __name__ == '__main__':
    tensors, y_info = get_tensors(log_prices=False)
    assert isinstance(y_info, tuple)

    model = PICNN(input_dim=INPUT_DIM, y_dim=Y_DIM, hidden_dim=64, n_layers=2)

    prob = StorageProblemPICNN(T=Y_DIM, L=model.L, d=model.hidden_dim, y_mean=y_info[0], y_std=y_info[1])

    tag = ''

    seed = 1
    tensors_cv = get_train_calib_split(tensors, seed=seed)
    loaders = get_loaders(tensors_cv, batch_size=BATCH_SIZE)
    rng = np.random.default_rng(seed)
    result = {'seed': seed}

    # run the FALCON fitting algorithm
    model, _ = fit_picnn_FALCON(
        model=model,
        lr=1e-3,
        l2reg=L2REG,
        epochs=20,
        loader=loaders['train'],
        device='cpu',
        num_chains=20
    )

    # run the ETO optimization
    q = conformal_q(model, loaders['calib'], alpha=0.1).item()
    result['q'] = q
    print(q)

    try:
        res = optimize(model, prob, loaders['test'], q)
        print(res)
    except Exception as e:
        print(e)
        import pdb
        pdb.set_trace()
        print()
