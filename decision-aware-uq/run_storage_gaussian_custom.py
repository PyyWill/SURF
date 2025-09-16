import argparse
from collections.abc import Collection, Iterable
import functools
import itertools
import os
from typing import Any

import numpy as np
import pandas as pd
import torch.utils.data
from tqdm.auto import tqdm

from models.gaussian import GaussianRegressorSplit, train_gaussian_regressor_custom
from models.gaussian_e2e import train_e2e
from models.gaussian_eto import optimize_wrapper
from storage.data import get_loaders, get_tensors, get_train_calib_split
from storage.problems import StorageProblemEllipsoid
from utils.parse_args import parse_args
from utils.multiprocess import run_parallel


INPUT_DIM = 101
Y_DIM = 24
MAX_ETO_EPOCHS = 500
MAX_E2E_EPOCHS = 100
BATCH_SIZE = 256
SEEDS = range(10)
LOG_PRICES = False


def run_saved(
    obj: str, shuffle: bool, alpha: float, lr: float, l2reg: float, out_dir: str,
    device: str, multiprocess: int = 1
) -> None:
    """Saves a CSV file with
        seed, {train/test}_task_loss, {train/test}_coverage, q

    Args:
        obj: 'eto', 'e2e', or 'e2e_finetune'
        shuffle: whether to shuffle dataset before splitting into train/calib/test
        alpha: risk level in (0, 1), the uncertainty set is a (1-alpha)-confidence set
        lr: learning rate
        l2reg: L2 regularization strength
        out_dir: where to save the results
        device: either 'cpu' or 'cuda'
        multiprocess: # of seeds to run in parallel
    """
    if obj == 'eto':
        basename = 'gaussian_regressor'
    else:
        basename = f'{obj}_a{alpha:.2f}_lr{lr:.3g}_reg{l2reg:.3g}'

    func = functools.partial(
        optimize_single, alpha=alpha, shuffle=shuffle, device=device,
        splits=('train', 'calib', 'test'))
    kwargs_list = [
        dict(seed=seed, ckpt_path=os.path.join(out_dir, f'{basename}_s{seed}.pt'))
        for seed in SEEDS
    ]
    results = run_parallel(func, kwargs_list, workers=multiprocess)
    results = [x for x in results if x is not None]
    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, f'{basename}_runsaved.csv')
    print(f'Saving results to {csv_path}')
    df.to_csv(csv_path, index=False)


def optimize_single(
    seed: int, alpha: float, shuffle: bool, device: str, ckpt_path: str,
    splits: Iterable[str] = ('train', 'test')
) -> tuple[dict[str, float], str]:
    """Optimize over ellipsoid uncertainty set.

    Args:
        device: either 'cpu' or 'cuda'

    Returns:
        result: dict of performance metrics
        msg: message string
    """
    tensors, y_info = get_tensors(shuffle=shuffle, log_prices=LOG_PRICES)
    assert isinstance(y_info, tuple)
    y_mean, y_std = y_info
    tensors_cv, _ = get_train_calib_split(tensors, seed=seed)
    loaders = get_loaders(tensors_cv, batch_size=BATCH_SIZE)

    prob = StorageProblemEllipsoid(T=Y_DIM, y_mean=y_mean, y_std=y_std)

    # load estimated model
    model = GaussianRegressorSplit(input_dim=INPUT_DIM, y_dim=Y_DIM).to(device)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    # then optimize
    result = optimize_wrapper(
        model, prob=prob, loaders=loaders, y_dim=Y_DIM, alpha=alpha, device=device,
        splits=splits)
    result['seed'] = seed

    msg = (f'alpha {alpha:.2g} train task loss {result["train_task_loss"]:.3f}, '
           f'test task loss {result["test_task_loss"]:.3f}, '
           f'test coverage {result["test_coverage"]:.3f}')
    return result, msg


def eto(
    shuffle: bool, alphas: Collection[float], device: str, out_dir: str,
    multiprocess: int = 1
) -> None:
    """
    Saves a CSV file with columns:
        seed, {train/val}_nll_losses, best_epoch,
        {train/val/test}_coverage_no_conformal,
        {train/val/test}_nll_loss, {train/test}_coverage,
        {train/test}_task_loss, q

    Args:
        shuffle: whether to shuffle dataset before splitting into train/calib/test
        alphas: risk levels in (0, 1), the uncertainty set is a (1-alpha)-confidence set
        device: either 'cpu' or 'cuda'
        out_dir: where to save the results
        multiprocess: number of processes to use, 1 = serial processing
    """
    for alpha in alphas:
        print(f'Optimizing for alpha {alpha:.2f}')
        func = functools.partial(
            optimize_single, shuffle=shuffle, alpha=alpha, device=device)
        kwargs_list = [
            dict(seed=seed,
                 ckpt_path=os.path.join(out_dir, f'gaussian_regressor_s{seed}.pt'))
            for seed in SEEDS
        ]
        results = run_parallel(func, kwargs_list, workers=multiprocess)

        df = pd.DataFrame(results)
        csv_path = os.path.join(out_dir, f'eto_a{alpha:.2f}.csv')
        print(f'Saving results to {csv_path}')
        df.to_csv(csv_path, index=False)


def e2e(
    shuffle: bool, alpha: float, lr: float, l2reg: float, out_dir: str,
    saved_model_fmt: str = ''
) -> tuple[pd.DataFrame, str]:
    """
    Everything in this function is done on CPU.

    Args:
        shuffle: whether to shuffle dataset before splitting into train/calib/test
        alpha: risk level in (0, 1), the uncertainty set is a (1-alpha)-confidence set
        lr: learning rate
        l2reg: L2 regularization strength
        out_dir: where to save the results
        saved_model_fmt: format string for the saved model, takes `seed` as parameter,
            e.g., 'gaussian_regressor_s{seed}.pt'
    """
    tensors, y_info = get_tensors(shuffle=shuffle, log_prices=LOG_PRICES)
    assert isinstance(y_info, tuple)
    y_mean, y_std = y_info
    prob = StorageProblemEllipsoid(T=Y_DIM, y_mean=y_mean, y_std=y_std)

    tag = '' if saved_model_fmt == '' else '_finetune'
    basename = f'e2e{tag}_a{alpha:.2f}_lr{lr:.3g}_reg{l2reg:.3g}'

    results = []
    for seed in tqdm(SEEDS):
        tensors_cv, _ = get_train_calib_split(tensors, seed=seed)
        loaders = get_loaders(tensors_cv, batch_size=BATCH_SIZE)
        rng = np.random.default_rng(seed)
        result: dict[str, Any] = {'seed': seed}

        # load pre-trained GaussianRegressor
        if saved_model_fmt != '':
            saved_model_path = os.path.join(out_dir, saved_model_fmt.format(seed=seed))
        else:
            saved_model_path = ''

        model = GaussianRegressorSplit(input_dim=INPUT_DIM, y_dim=Y_DIM)
        train_result = train_e2e(
            model, loaders, alpha=alpha, max_epochs=MAX_E2E_EPOCHS, lr=lr, l2reg=l2reg,
            prob=prob, rng=rng, nll_loss_frac=0.1, saved_model_path=saved_model_path)
        result |= train_result

        # evaluate on test
        opt_result = optimize_wrapper(
            model, prob=prob, loaders=loaders, y_dim=Y_DIM, alpha=alpha)
        result |= opt_result
        results.append(result)

        # save model
        ckpt_path = os.path.join(out_dir, f'{basename}_s{seed}.pt')
        torch.save(model.cpu().state_dict(), ckpt_path)

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, f'{basename}.csv')
    df.to_csv(csv_path, index=False)
    return df, f'Saving results to {csv_path}'


def get_best_hp_for_seed(
    seed: int, shuffle: bool, device: str, out_dir: str, tag: str = '',
    saved_ckpt_fmt: str = '', **train_kwargs: Any
) -> tuple[list[tuple[float, float, int, float]], str]:
    tensors, _ = get_tensors(shuffle=shuffle, log_prices=LOG_PRICES)
    tensors_cv, _ = get_train_calib_split(tensors, seed=seed)
    loaders = get_loaders(tensors_cv, batch_size=BATCH_SIZE)

    best_val_loss = np.inf
    best_model = None

    lrs = 10. ** np.arange(-4, -1.4, 0.5)
    l2regs = [1e-4]

    pbar = tqdm(total=len(lrs) * len(l2regs))
    losses = []
    for lr, l2reg in itertools.product(lrs, l2regs):
        model = GaussianRegressorSplit(input_dim=INPUT_DIM, y_dim=Y_DIM)
        if saved_ckpt_fmt != '':
            saved_ckpt_path = os.path.join(out_dir, saved_ckpt_fmt.format(seed=seed))
            model.load_state_dict(torch.load(saved_ckpt_path, weights_only=True))

        try:
            result = train_gaussian_regressor_custom(
                model, loaders, max_epochs=MAX_ETO_EPOCHS, lr=lr, l2reg=l2reg,
                return_best_model=True, device=device, show_pbar=True, **train_kwargs)
            losses.append((lr, l2reg, seed, result['val_loss']))

            if result['val_loss'] < best_val_loss:
                best_val_loss = result['val_loss']
                best_model = model.cpu()

        except Exception as e:
            tqdm.write(f'(lr {lr:.3g}, l2reg {l2reg:.3g}, seed {seed}) failed: {e}')
            losses.append((lr, l2reg, seed, np.nan))
        pbar.update(1)

    assert best_model is not None
    ckpt_path = os.path.join(out_dir, f'gaussian_regressor_s{seed}{tag}.pt')
    torch.save(best_model.state_dict(), ckpt_path)

    return losses, f'best val loss: {best_val_loss:.3f}'


def get_best_hyperparams(
    shuffle: bool, device: str, out_dir: str, tag: str = '',
    saved_ckpt_fmt: str = '', multiprocess: int = 1, **train_kwargs: Any
) -> None:
    """Finds best learning rate and L2 regularization strength for Gaussian regressor.

    Saves a CSV file to "{out_dir}/hyperparams{tag}.csv" with columns:
        lr, l2reg, seed, loss

    Saves the best model for each seed to
        "{out_dir}/gaussian_regressor_s{seed}{tag}.pt".

    Args:
        shuffle: whether to shuffle dataset before splitting into train/calib/test
        device: 'cpu' or 'cuda'
        out_dir: where to save the results
        tag: additional tag to append to CSV and saved model filenames
        saved_ckpt_fmt: format string for the saved model, takes `seed` as parameter,
            e.g., 'gaussian_regressor_s{seed}.pt'
        multiprocess: # of seeds to run in parallel
        train_kwargs: passed to `train_gaussian_regressor_custom`
    """
    func = functools.partial(
        get_best_hp_for_seed, shuffle=shuffle, device=device, out_dir=out_dir,
        tag=tag, saved_ckpt_fmt=saved_ckpt_fmt, **train_kwargs)
    kwargs_list = [dict(seed=seed) for seed in SEEDS]
    results = run_parallel(func, kwargs_list, workers=multiprocess)

    losses = []
    for result in results:
        if result is not None:
            losses.extend(result)

    df = pd.DataFrame(losses, columns=['lr', 'l2reg', 'seed', 'loss'])
    csv_path = os.path.join(out_dir, f'hyperparams{tag}.csv')
    df.to_csv(csv_path, index=False)
    print(f'Saved results to {csv_path}')


def main(args: argparse.Namespace) -> None:
    tag = args.tag
    device = args.device
    shuffle = args.shuffle

    if shuffle:
        out_dir = f'out/storage_gaussian_custom{tag}_shuffle/'
        # best l2reg found by get_best_hyperparams()
        L2REG = 1e-3
    else:
        out_dir = f'out/storage_gaussian_custom{tag}/'
        # best l2reg found by get_best_hyperparams()
        L2REG = 1e-4
    os.makedirs(out_dir, exist_ok=True)

    if args.command == 'run_saved':
        for alpha, lr, l2reg in itertools.product(args.alpha, args.lr, args.l2reg):
            run_saved(obj=args.obj, shuffle=shuffle, alpha=alpha, lr=lr, l2reg=l2reg,
                      out_dir=out_dir, device=device, multiprocess=args.multiprocess)

    elif args.command == 'best_hp':
        # train whole model from scratch using NLL only
        # - NOTE: this doesn't work as well as the multi-step training procedure
        #   below, where we first train to predict mean/std, then finetune the
        #   whole model.
        # get_best_hyperparams(
        #     shuffle=shuffle, device=device, out_dir=out_dir, tag='_nll',
        #     loss_name='nll', multiprocess=args.multiprocess)

        # train to predict mean/std only
        get_best_hyperparams(
            shuffle=shuffle, device=device, out_dir=out_dir, tag='_nlldiag',
            loss_name='nlldiag', multiprocess=args.multiprocess)

        # Optional step: fix embedding and loc_net, train to predict covariance matrix
        # - this sometimes helps improve NLL loss, but not always
        # get_best_hyperparams(
        #     shuffle=shuffle, device=device, out_dir=out_dir, tag='_nlldiag_nll',
        #     saved_ckpt_fmt='gaussian_regressor_s{seed}_nlldiag.pt',
        #     loss_name='nll', freeze=('embed', 'loc_net'), multiprocess=args.multiprocess)

        # finally, finetune the whole model
        get_best_hyperparams(
            shuffle=shuffle, device=device, out_dir=out_dir, tag='',
            saved_ckpt_fmt='gaussian_regressor_s{seed}_nlldiag.pt',
            loss_name='nll', multiprocess=args.multiprocess)

    elif args.command == 'eto':
        eto(shuffle=shuffle, alphas=args.alpha, device=device, out_dir=out_dir,
            multiprocess=args.multiprocess)

    elif args.command == 'e2e':
        func = functools.partial(
            e2e, shuffle=shuffle, l2reg=L2REG, out_dir=out_dir,
            saved_model_fmt='gaussian_regressor_s{seed}.pt')
        kwargs_list = [
            dict(alpha=alpha, lr=lr)
            for alpha, lr in itertools.product(args.alpha, args.lr)
        ]
        run_parallel(func, kwargs_list, workers=args.multiprocess)

    else:
        raise ValueError(f'Unknown command: {args.command}')


if __name__ == '__main__':
    # For E2E, we use the L2REG found from get_best_hyperparams(), and tune
    #     lr in [1e-2, 1e-3, 1e-4].
    commands = ('best_hp', 'eto', 'e2e', 'run_saved')
    args = parse_args(commands, lr=True, l2reg=True)

    # args = argparse.Namespace(
    #     command='best_hp',
    #     alpha=[0.1],
    #     device='cuda',
    #     tag='',
    #     shuffle=False,
    # )
    main(args)
