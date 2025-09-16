import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import torch
import yaml

class opf_problem:
    def __init__(self, 
                T: int, 
                N: int, 
                Y_A: np.ndarray, 
                adj_matrix: np.ndarray, 
                node_class: dict, 
                RTP: np.ndarray, 
                real_mainbus: torch.Tensor,
                real_schlinger: torch.Tensor,
                real_resnick: torch.Tensor,
                real_beckman: torch.Tensor,
                real_braun: torch.Tensor,
                imag_mainbus: torch.Tensor,
                imag_schlinger: torch.Tensor,
                imag_resnick: torch.Tensor,
                imag_beckman: torch.Tensor,
                imag_braun: torch.Tensor
                 ):
        """
        Initialize an optimal power flow problem instance.

        Parameters:
        - T (int): The number of time periods.
        - N (int): The number of buses.
        - Z_common (np.ndarray): The common impedance matrix. (change)
        - adj_matrix (np.ndarray): The matrix representing the grid topology
        - node_class (dict): A dictionary categorizing each node into different functionalities. (battery, impedance, PV)
        - RTP (np.ndarray): Real-Time Pricing rates for each time period, an array of length T.
        - demand (np.ndarray): The electrical demand for each node across the time periods.

        Returns:
        None. This is a constructor method for initializing the class instance with the given parameters.
        """
        self.T = T
        self.N = N
        self.Y_A = Y_A
        self.adj_matrix = adj_matrix
        self.RTP = RTP
        ###
        self.real_mainbus = cp.Parameter([3,1])
        self.imag_mainbus = cp.Parameter([3,1])
        self.real_schlinger = cp.Parameter([3,1])
        self.imag_schlinger = cp.Parameter([3,1])
        self.real_resnick = cp.Parameter([3,1])
        self.imag_resnick = cp.Parameter([3,1])
        self.real_beckman = cp.Parameter([3,1])
        self.imag_beckman = cp.Parameter([3,1])
        self.real_braun = cp.Parameter([3,1])
        self.imag_braun = cp.Parameter([3,1])
        ###
        self.battery_node_ls = node_class[0]
        self.impedance_node_ls = node_class[1]
        self.PV_node_ls = node_class[2]

        self.gamma = np.array([
            [1, -0.5 + 0.5 * np.sqrt(3) * 1j, -0.5 - 0.5 * np.sqrt(3) * 1j],
            [-0.5 - 0.5 * np.sqrt(3) * 1j, 1, -0.5 + 0.5 * np.sqrt(3) * 1j],
            [-0.5 + 0.5 * np.sqrt(3) * 1j, -0.5 - 0.5 * np.sqrt(3) * 1j, 1],
        ])
        self.B = np.outer(self.gamma[:,0], self.gamma[:,0].conj())
        self.Y = {}
        self.constraints = []
        self._create_impedance_matrix()
        self.grid_variable = self.create_grid_variable()
        self.device_variable = self.create_device_variable()

    def _create_impedance_matrix(self):
        connections = np.where(self.adj_matrix == 1)
        for i, j in zip(*connections):
            self.Y[f"{i}{j}"] = self.Y_A[f"{i}_{j}"]

    def dfs(self, start, visited=None):
        mat = np.triu(self.adj_matrix)
        if visited is None:
            visited = set()
        visited.add(start)
        for neighbor, isConnected in enumerate(mat[start]):
            if isConnected and neighbor not in visited:
                self.dfs(neighbor, visited)
        return visited

    def find_path(self, target, current=0, visited=None, path=None):
        mat = np.triu(self.adj_matrix)
        if visited is None:
            visited = set()
        if path is None:
            path = []
        visited.add(current)
        if current == target:
            return path
        for neighbor in range(len(mat[current])):
            if self.adj_matrix[current][neighbor] == 1 and neighbor not in visited:
                path.append((current, neighbor))
                result = self.find_path(target, neighbor, visited, path)
                if result is not None:
                    return result
                path.pop()
        return None

    def create_grid_variable(self):
        S = {}; V = {}; I = {}
        s = {}; v = {}; i = {}
        lam = {}; va = {}
        connections = np.where(self.adj_matrix == 1)
        for t in range(self.T):
            for n in range(self.N):
                s[f"{n}_{t}"] = cp.Variable((3, 1), complex=True)
                V[f"{n}_{t}"] = cp.Variable((3, 1), complex=True)
                v[f"{n}_{t}"] = cp.Variable((3, 3), complex=True)
                va[f"{n}_{t}"] = cp.Variable(complex=False)
                self.constraints += [v[f"{n}_{t}"] - va[f"{n}_{t}"]*self.B == 0]
                self.constraints += [v[f"{n}_{t}"] >> 0]
            for j, k in zip(*connections):
                S[f"{j}{k}_{t}"] = cp.Variable((3, 3), complex=True)
                I[f"{j}{k}_{t}"] = cp.Variable((3, 1), complex=True)
                i[f"{j}{k}_{t}"] = cp.Variable((3, 3), complex=True)
                lam[f"{j}{k}_{t}"] = cp.Variable((3, 1), complex=True)
                self.constraints += [i[f"{j}{k}_{t}"] >> 0]
        return (S, V, I, s, v, i, lam, va)

    def create_device_variable(self):
        p_battery = {}; p_PV = {}; p_impedance = {}
        for t in range(self.T):
            for n in self.battery_node_ls:
                p_battery[f"{n}_{t}"] = cp.Variable((3, 1), complex=True)
            for n in self.PV_node_ls:
                p_PV[f"{n}_{t}"] = cp.Variable((3, 1), complex=True)
            for n in self.impedance_node_ls:
                p_impedance[f"{n}_{t}"] = cp.Variable((3, 1), complex=True)
        return (p_battery, p_PV, p_impedance)

    def solve(self):
        self.lindistflow_constraints()
        self.operational_constraints()
        self.nodal_injection_equation()
        # self.power_flow_equation()

        '''
        Need change: objective_func --> 1. mean value for bus power; 2. deployment cost
        '''
        objective_func = sum(self.RTP[t] * cp.real(self.grid_variable[3][f"0_{t}"][0]) for t in range(self.T))
        objective = cp.Minimize(objective_func)
        problem = cp.Problem(objective, self.constraints)
        result = problem.solve()
        print("Result: ", result)

        return None


    def lindistflow_constraints(self):
        S, V, I, s, v, i, lam, va = self.grid_variable

        # Total power constraint: Power in should equal power out
        for t in range(self.T):
            self.constraints += [sum(s[f"{n}_{t}"] for n in range(self.N)) == 0]

        # Tree structure constraints: Power flows should be consistent through the tree
        for t in range(self.T):
            connections = np.where(self.adj_matrix == 1)
            for j, k in zip(*connections):
                sub_node = self.dfs(k)
                subtree_power = sum(s[f"{n}_{t}"] for n in sub_node)
                self.constraints += [lam[f"{j}{k}_{t}"] + subtree_power == 0]
                self.constraints += [S[f"{j}{k}_{t}"] - cp.matmul(self.gamma, cp.diag(lam[f"{j}{k}_{t}"])) == 0]

        # Voltage - s constraints
        for t in range(self.T):
            for j, k in zip(*connections):
                if j>k: # for Y_common
                    voltage_diff = self.Y[f"{j}{k}"] @ (v[f"{j}_{t}"]-v[f"{k}_{t}"]) @ cp.conj(self.Y[f"{j}{k}"]).T
                    self.constraints += [voltage_diff - cp.conj(S[f"{j}{k}_{t}"]).T @ cp.conj(self.Y[f"{j}{k}"]).T - self.Y[f"{j}{k}"] @ S[f"{j}{k}_{t}"] == 0]  


    def operational_constraints(self):
        S, V, I, s, v, i, lam, va = self.grid_variable
        connections = np.where(self.adj_matrix == 1)
        '''
        Need change: v, s constraints --> 1. Set rational value for boundaries
        '''

        # v, s constraints
        for t in range(self.T):
            for n in range(self.N):
                self.constraints += [cp.abs(cp.diag(v[f"{n}_{t}"])[i]) <= 300 for i in range(3)]
                self.constraints += [cp.abs(s[f"{n}_{t}"][i]) <= 80 for i in range(3)]
        # i constraints
        for j, k in zip(*connections):
            self.constraints += [cp.abs(i[f"{j}{k}_{t}"][p]) <= 100 for p in range(3)]

    def nodal_injection_equation(self):
        S, V, I, s, v, i, lam, va = self.grid_variable
        p_battery, p_PV, p_impedance = self.device_variable

        for t in range(self.T):
            # Impedance power injection constraints
            self.constraints += [p_impedance[f"0_{t}"] - (self.real_mainbus + self.imag_mainbus*1j)  - s[f"0_{t}"] == 0]
            self.constraints += [p_impedance[f"1_{t}"] - (self.real_schlinger + self.imag_schlinger*1j)  - s[f"1_{t}"] == 0]
            for n in self.impedance_node_ls:
                self.constraints += [cp.abs(p_impedance[f"{n}_{t}"][p]) <= 100 for p in range(3)]
                # self.constraints += [cp.real(p_impedance[f"{n}_0"]) <= 0]

            # PV power injection constraints

            # Battery power constraints
            self.constraints += [p_battery[f"2_{t}"] - (self.real_resnick + self.imag_resnick*1j)  - s[f"2_{t}"] == 0]
            self.constraints += [p_battery[f"3_{t}"] - (self.real_beckman + self.imag_beckman*1j)  - s[f"3_{t}"] == 0]
            self.constraints += [p_battery[f"4_{t}"] - (self.real_braun + self.imag_braun*1j)  - s[f"4_{t}"] == 0]
            for n in self.battery_node_ls:
                # Battery state-of-charge constraints
                self.constraints += [cp.abs(p_battery[f"{n}_{t}"][p]) <= 100 for p in range(3)]
                # self.constraints += [p_battery[f"{n}_0"] == 0]


    def power_flow_equation(self):
        S, V, I, s, v, i, lam, va = self.grid_variable
        connections = np.where(self.adj_matrix == 1)
        
        for t in range(self.T):
            for j, k in zip(*connections):
                self.constraints += [S[f"{j}{k}_{t}"] == cp.matmul(V[f"{j}_{t}"], cp.conj(I[f"{j}{k}_{t}"]).T)]
            for n in range(self.N):
                self.constraints += [v[f"{n}_{t}"] == cp.matmul(V[f"{n}_{t}"], cp.conj(V[f"{n}_{t}"]).T)]


class opf_problem_optnn(opf_problem):
    def __init__(self, T, N, Y_A, adj_matrix, node_class, RTP,
                 real_mainbus, real_schlinger, real_resnick, real_beckman, real_braun,
                 imag_mainbus, imag_schlinger, imag_resnick, imag_beckman, imag_braun):
        super().__init__(T, N, Y_A, adj_matrix, node_class, RTP,
                         real_mainbus, real_schlinger, real_resnick, real_beckman, real_braun,
                         imag_mainbus, imag_schlinger, imag_resnick, imag_beckman, imag_braun)

        self.lindistflow_constraints()
        self.operational_constraints()
        self.nodal_injection_equation()
        # self.power_flow_equation()

        objective_func = sum(self.RTP[t] * cp.real(self.grid_variable[3][f"0_{t}"][0]) for t in range(self.T))
        objective = cp.Minimize(objective_func)
        prob = cp.Problem(objective, self.constraints)
        self.output = []
        p_battery, p_PV, p_impedance, soc_battery = self.device_variable
        for t in range(T):
            for n in range(N):
                if n in self.battery_node_ls:
                    self.output.append(p_battery[f"{n}_{t}"])
                    # Add SOC variables to output for monitoring
                    self.output.append(soc_battery[f"{n}_{t}"])
                if n in self.PV_node_ls:
                    self.output.append(p_PV[f"{n}_{t}"])
                if n in self.impedance_node_ls:
                    self.output.append(p_impedance[f"{n}_{t}"])

        assert prob.is_dpp()
        self.prob = prob
        self.layer = CvxpyLayer(prob, 
                                parameters=[self.real_mainbus, self.real_schlinger, self.real_resnick, self.real_beckman, self.real_braun,
                                            self.imag_mainbus, self.imag_schlinger, self.imag_resnick, self.imag_beckman, self.imag_braun], 
                                variables=self.output)

    def torch_loss(self,
                   real_mainbus, real_schlinger, real_resnick, real_beckman, real_braun,
                   imag_mainbus, imag_schlinger, imag_resnick, imag_beckman, imag_braun):
        device_variable = self.forward(
            real_mainbus, real_schlinger, real_resnick, real_beckman, real_braun,
            imag_mainbus, imag_schlinger, imag_resnick, imag_beckman, imag_braun)
        return self.compute_loss(real_mainbus, real_schlinger, real_resnick, real_beckman, real_braun,
                                device_variable)
    
    def compute_loss(self,
                     real_mainbus, real_schlinger, real_resnick, real_beckman, real_braun,
                     device_variable):
        p_battery = {}; p_PV = {}; p_impedance = {}
        for t in range(self.T):
            p_impedance[f"0_{t}"] = device_variable[t*5+0]
            p_impedance[f"1_{t}"] = device_variable[t*5+1]
            p_battery[f"2_{t}"] = device_variable[t*5+2]
            p_battery[f"3_{t}"] = device_variable[t*5+3]
            p_battery[f"4_{t}"] = device_variable[t*5+4]

        def cost_t(t):
            loss_t = 0
            cost_impedance_node = 0
            cost_battery_node = 0
            cost_PV_node = 0

            # phase a
            cost_impedance_node = real_mainbus - p_impedance[f"0_{t}"].real
            cost_impedance_node += real_schlinger - p_impedance[f"1_{t}"].real
            cost_battery_node = real_resnick - p_battery[f"2_{t}"].real
            cost_battery_node += real_beckman - p_battery[f"3_{t}"].real
            cost_battery_node += real_braun - p_battery[f"4_{t}"].real

            loss_t = cost_battery_node + cost_PV_node + cost_impedance_node
            return loss_t[0]

        cost = sum(self.RTP[t]*cost_t(t) for t in range(self.T))

        return cost
    
    def forward(self,
                real_mainbus, real_schlinger, real_resnick, real_beckman, real_braun,
                imag_mainbus, imag_schlinger, imag_resnick, imag_beckman, imag_braun):
        solution = self.layer(real_mainbus, real_schlinger, real_resnick, real_beckman, real_braun,
                              imag_mainbus, imag_schlinger, imag_resnick, imag_beckman, imag_braun)
        return solution

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def str_to_matrices(arr):
    return np.array([[complex(val.replace(' ', '')) for val in row] for row in arr], dtype=np.complex64)

def str_to_vector(arr):
    return np.array([complex(val) for val in arr], dtype=np.complex64)

if __name__ == "__main__":
    config = load_config("config.yaml")

    # 读取配置中的Y_A并转换为复数矩阵
    Y_A = {}
    for key, value in config['Y_A'].items():
        Y_A[key] = str_to_matrices(value)  # 使用str_to_complex函数解析复数字符串

    RTP = np.array([config['RTP']])
    T = config['T']
    adj_matrix = np.array(config['adj_matrix'])
    N = len(adj_matrix)

    node_class = config['node_class']

    # 读取并转换需求数据
    demands_mainbus = torch.tensor(str_to_vector(config['demands_mainbus']), dtype=torch.complex64)/1000
    demands_schlinger = torch.tensor(str_to_vector(config['demands_schlinger']), dtype=torch.complex64)/1000
    demands_resnick = torch.tensor(str_to_vector(config['demands_resnick']), dtype=torch.complex64)/1000
    demands_beckman = torch.tensor(str_to_vector(config['demands_beckman']), dtype=torch.complex64)/1000
    demands_braun = torch.tensor(str_to_vector(config['demands_braun']), dtype=torch.complex64)/1000

    def split_real_imag(tensor):
        real_part = (tensor.real).unsqueeze(-1)
        imag_part = (tensor.imag).unsqueeze(-1)
        return real_part, imag_part

    real_mainbus, imag_mainbus = split_real_imag(demands_mainbus)
    real_schlinger, imag_schlinger = split_real_imag(demands_schlinger)
    real_resnick, imag_resnick = split_real_imag(demands_resnick)
    real_beckman, imag_beckman = split_real_imag(demands_beckman)
    real_braun, imag_braun = split_real_imag(demands_braun)

    opf = opf_problem_optnn(T, N, Y_A, adj_matrix, node_class, RTP, 
                            real_mainbus, real_schlinger, real_resnick, real_beckman, real_braun,
                            imag_mainbus, imag_schlinger, imag_resnick, imag_beckman, imag_braun)

    torch_loss = opf.torch_loss(real_mainbus, real_schlinger, real_resnick, real_beckman, real_braun,
                                imag_mainbus, imag_schlinger, imag_resnick, imag_beckman, imag_braun)
    print(torch_loss)
