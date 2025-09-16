"""
This module implements optimization over PICNN sublevel set uncertainty.
"""
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import torch
from torch import nn, Tensor

from models.picnn import PICNN


def einsum_batch_diag(A: Tensor, B: Tensor) -> Tensor:
    """
    Computes A @ batch_diag(B) using einsum.

    Equivalent to
        torch.stack([
            A @ torch.diag(B[i]) for i in range(B.shape[0])
        ])

    Args:
        A: shape (m, n)
        B: shape (batch_size, n)

    Returns:
        result: shape (batch_size, m, n)
    """
    # einsum expression explanation:
    # 'ik,kj->ij' represents the matrix multiplication of A and diag(B_i)
    return torch.einsum('ij,bj->bij', A, B)


class PICNNProblem:
    # instance variables
    L: int
    d: int
    dual_constraints: list[cp.Constraint]
    dual_obj: cp.Expression
    dual_vars: dict[str, cp.Variable]
    params: dict[str, cp.Parameter]

    # subclass must provide these instance variables
    prob: cp.Problem

    def __init__(
        self, y_dim: int, L: int, d: int, y_mean: np.ndarray, y_std: np.ndarray,
        Fz: cp.Expression, epsilon: float = 0.
    ):
        assert epsilon >= 0.

        self.L = L
        self.d = d
        self.epsilon = epsilon
        self.y_mean = y_mean
        self.y_std = y_std

        nu = cp.Variable(2*L*d + 1, nonneg=True)
        nus = [
            nu[d*l: d*(l+1)] for l in range(2*L + 1)
        ]
        mu = nu[-1]
        rho = cp.Variable(2*y_dim, nonneg=True)
        eta = cp.Variable(nonneg=True)
        self.dual_vars = {'nu': nu, 'rho': rho, 'eta': eta}

        # there are L+1 Vs, L Ws, L+1 bs, and 1 q
        V_stacked = cp.Parameter((L*d + 1, y_dim))
        W_stacked = cp.Parameter(((L-1)*d + 1, d))
        Ws = [W_stacked[d*l: d*(l+1)] for l in range(L)]
        bs = cp.Parameter((L, d))
        b_fin = cp.Parameter()
        q = cp.Parameter()
        gamma = cp.Parameter(nonneg=True)
        self.params = {
            'V_stacked': V_stacked,
            'W_stacked': W_stacked,
            'bs': bs,
            'b_fin': b_fin,
            'q': q,
            'gamma': gamma,
        }

        # objective
        obj = - b_fin * mu - q * eta
        for l in range(L):
            obj -= bs[l] @ nus[L + l]
        obj += y_mean @ Fz
        # add a regularization term on the dual variable nu to avoid numerical issues
        # obj += 1e-5 * cp.sum_squares(nu)
        self.dual_obj = obj

        # constraints
        constr: list[cp.Constraint] = [
            V_stacked.T @ nu[L*d:] + gamma*rho[:y_dim] - gamma*rho[y_dim:] == cp.multiply(y_std, Fz)
        ]
        for l in range(L):
            constr.append(Ws[l].T @ nus[L+l+1] - nus[L+l] - nus[l] == 0)
        constr.append(-cp.sum(rho) + mu == 0)
        constr.append(eta + mu == epsilon)

        self.dual_constraints = constr

    def solve_primal_max(self, x: np.ndarray, model: PICNN, q: float) -> cp.Problem:
        assert model.L == self.L
        assert model.hidden_dim == self.d

        L = self.L
        d = self.d
        ReLU = nn.ReLU()

        sigma = cp.Variable((L, d), nonneg=True)
        y = cp.Variable(model.y_dim)
        kappa = cp.Variable()
        qhat = cp.Variable()

        constraints = []

        with torch.no_grad():
            u = torch.from_numpy(x)
            for l in range(L+1):
                if l > 0:
                    W_hat_vec = ReLU(model.W_hat_layers[l](u))
                    W = model.W_bar_layers[l].weight @ torch.diag(W_hat_vec)
                V_hat_vec = model.V_hat_layers[l](u)
                V = model.V_bar_layers[l].weight @ torch.diag(V_hat_vec)
                b = model.b_layers[l](u)
                if l < L:
                    u = ReLU(model.u_layers[l](u))

                if l == 0:
                    constraints.append(sigma[l] >= V.numpy() @ y + b)
                elif l <= L - 1:
                    constraints.append(sigma[l] >= W.numpy() @ sigma[l-1] + V.numpy() @ y + b)
                else:
                    constraints.append(W.numpy() @ sigma[-1] + V.numpy() @ y + kappa + b <= qhat)

        constraints.append(kappa >= model.gamma * y)
        constraints.append(kappa >= -model.gamma * y)
        constraints.append(qhat >= q)

        # sample a random Gaussian cost vector
        c = np.random.randn(model.y_dim)

        prob = cp.Problem(cp.Maximize(c @ y - self.epsilon * qhat), constraints=constraints)
        try:
            prob.solve(solver=cp.CLARABEL)  # solver=cp.MOSEK or solver=cp.CLARABEL
        except Exception as e:
            print(e)
            prob.solve(solver=cp.MOSEK)

        if prob.status != 'optimal':
            print('Primal Maximization Problem status:', prob.status)
        return prob

    def solve(self, x: np.ndarray, model: PICNN, q: float) -> cp.Problem:
        assert model.L == self.L
        assert model.hidden_dim == self.d

        L = self.L
        d = self.d
        ReLU = nn.ReLU()

        V_stacked = np.zeros((L*d + 1, model.y_dim))
        Vs = [
            V_stacked[d*l: d*(l+1)] for l in range(L + 1)
        ]
        W_stacked = np.zeros(((L-1)*d + 1, d))
        Ws = [
            W_stacked[d*l: d*(l+1)] for l in range(L)
        ]
        bs = np.zeros((L, d))

        with torch.no_grad():
            u = torch.from_numpy(x)
            for l in range(L):
                if l > 0:
                    W_hat_vec = ReLU(model.W_hat_layers[l](u))
                    W = model.W_bar_layers[l].weight @ torch.diag(W_hat_vec)
                    Ws[l-1][:] = W.numpy()
                V_hat_vec = model.V_hat_layers[l](u)
                V = model.V_bar_layers[l].weight @ torch.diag(V_hat_vec)
                b = model.b_layers[l](u)
                u = ReLU(model.u_layers[l](u))
                Vs[l][:] = V.numpy()
                bs[l] = b.numpy()
            l = L
            W_hat_vec = ReLU(model.W_hat_layers[l](u))
            W = model.W_bar_layers[l].weight @ torch.diag(W_hat_vec)
            Ws[l-1][:] = W.numpy()  # shape [1, d]
            V_hat_vec = model.V_hat_layers[l](u)
            V = model.V_bar_layers[l].weight @ torch.diag(V_hat_vec)
            Vs[-1][:] = V.numpy()  # shape [1, y_dim]
            b_fin = model.b_layers[l](u)[0].numpy()

        self.params['V_stacked'].value = V_stacked
        self.params['W_stacked'].value = W_stacked
        self.params['bs'].value = bs
        self.params['b_fin'].value = b_fin
        self.params['q'].value = q
        self.params['gamma'].value = model.gamma

        prob = self.prob
        try:
            prob.solve(solver=cp.CLARABEL)  # solver=cp.MOSEK or solver=cp.CLARABEL
        except Exception as e:
            print(e)
            prob.solve(solver=cp.MOSEK)

        if prob.status != 'optimal':
            print('Problem status:', prob.status)
        return prob

    def solve_cvxpylayers(
        self, x: Tensor, model: PICNN, q: Tensor, layer: CvxpyLayer
    ) -> list[Tensor]:
        ReLU = nn.ReLU()
        L = model.L
        d = model.hidden_dim
        batch_size = x.shape[0]
        V_stacked = torch.zeros((batch_size, L*d + 1, model.y_dim))
        Vs = [
            V_stacked[:, d*l: d*(l+1)] for l in range(L + 1)
        ]
        W_stacked = torch.zeros((batch_size, (L-1)*d + 1, d))
        Ws = [
            W_stacked[:, d*l: d*(l+1)] for l in range(L)
        ]
        bs = torch.zeros((batch_size, L, d))

        u = x
        for l in range(L):
            if l > 0:
                W_hat_vec = ReLU(model.W_hat_layers[l](u))
                Ws[l-1][:] = einsum_batch_diag(model.W_bar_layers[l].weight, W_hat_vec)
            V_hat_vec = model.V_hat_layers[l](u)
            Vs[l][:] = einsum_batch_diag(model.V_bar_layers[l].weight, V_hat_vec)
            bs[:, l] = model.b_layers[l](u)
            u = ReLU(model.u_layers[l](u))
        l = L
        W_hat_vec = ReLU(model.W_hat_layers[l](u))
        Ws[l-1][:] = einsum_batch_diag(model.W_bar_layers[l].weight, W_hat_vec)
        V_hat_vec = model.V_hat_layers[l](u)
        Vs[l][:] = einsum_batch_diag(model.V_bar_layers[l].weight, V_hat_vec)
        b_fin = model.b_layers[l](u)[:, 0]

        return layer(V_stacked, W_stacked, bs, b_fin, q, torch.tensor(model.gamma))
