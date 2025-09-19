"""
Optimal Power Flow (OPF) Problem Solver with Decision-Aware Uncertainty Quantification.

This module provides classes for solving optimal power flow problems in electrical grids
using CVXPY optimization and PyTorch integration for neural network training.
"""

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import torch
import yaml
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum

# Constants
BIG_M = 10000
PHASE_COUNT = 3
NODE_INDICES = {
    'MAINBUS': 0,
    'BROAD': 1,
    'SCHLINGER': 2, 
    'RESNICK': 3,
    'BECKMAN': 4,
    'BRAUN': 5
}


class NodeType(Enum):
    """Node types in the power grid."""
    BATTERY = 0
    IMPEDANCE = 1
    PV = 2


@dataclass
class GridParameters:
    """Data class for grid parameters."""
    T: int
    N: int
    Y_A: Dict[str, np.ndarray]
    adj_matrix: np.ndarray
    node_class: Dict[int, List[int]]
    RTP: np.ndarray


@dataclass
class PowerDemands:
    """Data class for power demand parameters."""
    real_mainbus: torch.Tensor
    real_broad: torch.Tensor
    real_schlinger: torch.Tensor
    real_resnick: torch.Tensor
    real_beckman: torch.Tensor
    real_braun: torch.Tensor


class OptimalPowerFlowProblem:
    """
    Optimal Power Flow problem solver for electrical grids.
    
    This class formulates and solves the OPF problem using CVXPY optimization,
    including linear distribution flow constraints, operational constraints,
    and nodal injection equations.
    """
    
    def __init__(self, 
                 T: int, 
                 N: int, 
                 Y_A: np.ndarray, 
                 adj_matrix: np.ndarray, 
                 node_class: dict, 
                 RTP: np.ndarray, 
                 real_mainbus: torch.Tensor,
                 real_broad: torch.Tensor,
                 real_schlinger: torch.Tensor,
                 real_resnick: torch.Tensor,
                 real_beckman: torch.Tensor,
                 real_braun: torch.Tensor
                 ):
        """
        Initialize an optimal power flow problem instance.

        Args:
            T: The number of time periods.
            N: The number of buses.
            Y_A: The admittance matrix dictionary.
            adj_matrix: The matrix representing the grid topology.
            node_class: A dictionary categorizing each node into different functionalities.
            RTP: Real-Time Pricing rates for each time period.
            real_mainbus: Main bus real power demand.
            real_broad: Broad node real power demand.
            real_schlinger: Schlinger node real power demand.
            real_resnick: Resnick node real power demand.
            real_beckman: Beckman node real power demand.
            real_braun: Braun node real power demand.
        """
        self._setup_parameters(T, N, Y_A, adj_matrix, node_class, RTP, real_mainbus, real_broad, real_schlinger, real_resnick, real_beckman, real_braun)
        self._setup_electrical_parameters()
        self._setup_optimization_variables()
        
    def _setup_parameters(self, T: int, N: int, Y_A: np.ndarray, 
                         adj_matrix: np.ndarray, node_class: dict, RTP: np.ndarray, real_mainbus: torch.Tensor, real_broad: torch.Tensor, real_schlinger: torch.Tensor, real_resnick: torch.Tensor, real_beckman: torch.Tensor, real_braun: torch.Tensor) -> None:
        """Setup basic problem parameters."""
        self.T = T
        self.N = N
        self.Y_A = Y_A
        self.adj_matrix = adj_matrix
        self.RTP = RTP

        self.real_mainbus = cp.Parameter([T,3])
        self.real_broad = cp.Parameter([T,3])
        self.real_schlinger = cp.Parameter([T,3])
        self.real_resnick = cp.Parameter([T,3])
        self.real_beckman = cp.Parameter([T,3])
        self.real_braun = cp.Parameter([T,3])
        
        # Node classifications
        self.battery_nodes = node_class[NodeType.BATTERY.value]
        self.impedance_nodes = node_class[NodeType.IMPEDANCE.value]
        
    def _setup_electrical_parameters(self) -> None:
        """Setup electrical system parameters."""
        # Three-phase transformation matrix
        self.gamma = np.array([
            [1, -0.5 + 0.5 * np.sqrt(3) * 1j, -0.5 - 0.5 * np.sqrt(3) * 1j],
            [-0.5 - 0.5 * np.sqrt(3) * 1j, 1, -0.5 + 0.5 * np.sqrt(3) * 1j],
            [-0.5 + 0.5 * np.sqrt(3) * 1j, -0.5 - 0.5 * np.sqrt(3) * 1j, 1],
        ])
        
        # Voltage transformation matrix
        self.B = np.outer(self.gamma[:, 0], self.gamma[:, 0].conj())
        
        # Impedance matrix
        self.Y = self._create_impedance_matrix()
        
        # Optimization constraints
        self.constraints = []
        
    def _create_impedance_matrix(self) -> Dict[str, np.ndarray]:
        """Create impedance matrix from adjacency matrix."""
        Y = {}
        connections = np.where(self.adj_matrix == 1)
        for i, j in zip(*connections):
            Y[f"{i}{j}"] = self.Y_A[f"{i}_{j}"]
        return Y
        
    def _setup_optimization_variables(self) -> None:
        """Setup CVXPY optimization variables."""
        self.grid_variables = self._create_grid_variables()
        self.device_variables = self._create_device_variables()
        
    def _create_grid_variables(self) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Create grid-related optimization variables."""
        S = {}  # Power flow variables
        s = {}  # Node power injection
        v = {}  # Voltage matrix
        lam = {}  # Lagrange multipliers
        va = {}  # Voltage amplitude
        
        connections = np.where(self.adj_matrix == 1)
        
        for t in range(self.T):
            # Node variables
            for n in range(self.N):
                s[f"{n}_{t}"] = cp.Variable((PHASE_COUNT, 1), complex=True)
                v[f"{n}_{t}"] = cp.Variable((PHASE_COUNT, PHASE_COUNT), complex=True)
                va[f"{n}_{t}"] = cp.Variable(complex=False)
                
                # Voltage constraints
                self.constraints += [v[f"{n}_{t}"] - va[f"{n}_{t}"] * self.B == 0]
                # self.constraints += [v[f"{n}_{t}"] >> 0]
                
            # Connection variables
            for j, k in zip(*connections):
                S[f"{j}{k}_{t}"] = cp.Variable((PHASE_COUNT, PHASE_COUNT), complex=True)
                lam[f"{j}{k}_{t}"] = cp.Variable((PHASE_COUNT, 1), complex=True)
                
        return (S, s, v, lam, va)
        
    def _create_device_variables(self) -> Tuple[Dict, Dict]:
        """Create device-related optimization variables."""
        p_battery = {}
        p_impedance = {}
        
        for t in range(self.T):
            for n in self.battery_nodes:
                p_battery[f"{n}_{t}"] = cp.Variable((PHASE_COUNT, 1))
            for n in self.impedance_nodes:
                p_impedance[f"{n}_{t}"] = cp.Variable((PHASE_COUNT, 1))
                
        return (p_battery, p_impedance)
        
    def _dfs(self, start: int, visited: Optional[set] = None) -> set:
        """Depth-first search for tree structure analysis."""
        mat = np.triu(self.adj_matrix)
        if visited is None:
            visited = set()
        visited.add(start)
        
        for neighbor, is_connected in enumerate(mat[start]):
            if is_connected and neighbor not in visited:
                self._dfs(neighbor, visited)
        return visited
        
    def _add_linear_distribution_flow_constraints(self) -> None:
        """Add linear distribution flow constraints."""
        S, s, v, lam, va = self.grid_variables
        
        # Total power balance constraint
        for t in range(self.T):
            self.constraints += [sum(s[f"{n}_{t}"] for n in range(self.N)) == 0]
            
        # Tree structure constraints
        for t in range(self.T):
            connections = np.where(self.adj_matrix == 1)
            for j, k in zip(*connections):
                sub_nodes = self._dfs(k)
                subtree_power = sum(s[f"{n}_{t}"] for n in sub_nodes)
                self.constraints += [lam[f"{j}{k}_{t}"] + subtree_power == 0]
                self.constraints += [S[f"{j}{k}_{t}"] - cp.matmul(self.gamma, cp.diag(lam[f"{j}{k}_{t}"])) == 0]
                
        # Voltage-power constraints
        for t in range(self.T):
            for j, k in zip(*connections):
                if j > k:  # For Y_common
                    voltage_diff = (self.Y[f"{j}{k}"] @ (v[f"{j}_{t}"] - v[f"{k}_{t}"]) @ 
                                  cp.conj(self.Y[f"{j}{k}"]).T)
                    power_term = (cp.conj(S[f"{j}{k}_{t}"]).T @ cp.conj(self.Y[f"{j}{k}"]).T + 
                                self.Y[f"{j}{k}"] @ S[f"{j}{k}_{t}"])
                    self.constraints += [voltage_diff - power_term == 0]
                    
    def _add_operational_constraints(self) -> None:
        """Add operational constraints (voltage and power limits)."""
        S, s, v, lam, va = self.grid_variables
        
        # Voltage and power magnitude constraints
        for t in range(self.T):
            for n in range(self.N):
                self.constraints += [cp.abs(cp.diag(v[f"{n}_{t}"])[i]) <= BIG_M for i in range(PHASE_COUNT)]
                self.constraints += [cp.abs(s[f"{n}_{t}"][i]) <= BIG_M for i in range(PHASE_COUNT)]
                
    def _add_nodal_injection_constraints(self) -> None:
        """Add nodal power injection constraints."""
        S, s, v, lam, va = self.grid_variables
        p_battery, p_impedance = self.device_variables
        
        for t in range(self.T):
            # Impedance node constraints
            self.constraints += [p_impedance[f"{NODE_INDICES['MAINBUS']}_{t}"] - 
                               cp.reshape(self.real_mainbus[t], (3,1)) - cp.real(s[f"{NODE_INDICES['MAINBUS']}_{t}"]) == 0]
            self.constraints += [p_impedance[f"{NODE_INDICES['BROAD']}_{t}"] - 
                               cp.reshape(self.real_broad[t], (3,1)) - cp.real(s[f"{NODE_INDICES['BROAD']}_{t}"]) == 0]
            self.constraints += [p_impedance[f"{NODE_INDICES['SCHLINGER']}_{t}"] - 
                               cp.reshape(self.real_schlinger[t], (3,1)) - cp.real(s[f"{NODE_INDICES['SCHLINGER']}_{t}"]) == 0]
            
            for n in self.impedance_nodes:
                self.constraints += [cp.abs(p_impedance[f"{n}_{t}"][p]) <= BIG_M for p in range(PHASE_COUNT)]
                
            # Battery node constraints
            self.constraints += [p_battery[f"{NODE_INDICES['RESNICK']}_{t}"] - 
                               cp.reshape(self.real_resnick[t], (3,1)) - cp.real(s[f"{NODE_INDICES['RESNICK']}_{t}"]) == 0]
            self.constraints += [p_battery[f"{NODE_INDICES['BECKMAN']}_{t}"] - 
                               cp.reshape(self.real_beckman[t], (3,1)) - cp.real(s[f"{NODE_INDICES['BECKMAN']}_{t}"]) == 0]
            self.constraints += [p_battery[f"{NODE_INDICES['BRAUN']}_{t}"] - 
                               cp.reshape(self.real_braun[t], (3,1)) - cp.real(s[f"{NODE_INDICES['BRAUN']}_{t}"]) == 0]
            
            for n in self.battery_nodes:
                self.constraints += [cp.sum(p_battery[f"{n}_{t}"]) <= BIG_M]
                self.constraints += [sum(cp.sum(p_battery[f"{n}_{t}"]) for t in range(self.T)) <= BIG_M]
                
    def _create_objective_function(self) -> cp.Expression:
        """Create the objective function for optimization."""
        S, s, v, lam, va = self.grid_variables
        return sum(self.RTP * cp.real(s[f"{NODE_INDICES['MAINBUS']}_{t}"][0]) for t in range(self.T))
        
    def solve(self) -> Optional[float]:
        """
        Solve the optimal power flow problem.
        
        Returns:
            Optimization result value or None if failed
        """
        self._add_linear_distribution_flow_constraints()
        self._add_operational_constraints()
        self._add_nodal_injection_constraints()
        
        objective = cp.Minimize(self._create_objective_function())
        problem = cp.Problem(objective, self.constraints)
        result = problem.solve()
        
        print(f"Optimization result: {result}")
        return result


class OptimalPowerFlowNeuralNetwork(OptimalPowerFlowProblem):
    """
    Neural network-enabled OPF solver using CvxpyLayer.
    
    This class extends the basic OPF solver to integrate with PyTorch neural networks
    for end-to-end training and optimization.
    """
    
    def __init__(self, T, N, Y_A, adj_matrix, node_class, RTP,
                 real_mainbus, real_broad, real_schlinger, real_resnick, real_beckman, real_braun):
        """Initialize the neural network OPF solver."""
        super().__init__(T, N, Y_A, adj_matrix, node_class, RTP,
                         real_mainbus, real_broad, real_schlinger, real_resnick, real_beckman, real_braun)
        self._setup_neural_network()
        
    def _setup_neural_network(self) -> None:
        """Setup the neural network components."""
        self._add_linear_distribution_flow_constraints()
        self._add_operational_constraints()
        self._add_nodal_injection_constraints()
        
        # Create optimization problem
        objective = cp.Minimize(self._create_objective_function())
        self.problem = cp.Problem(objective, self.constraints)
        
        # Setup output variables for neural network
        self.output_variables = self._create_output_variables()
        
        # Verify problem is DPP (Disciplined Parametric Programming)
        assert self.problem.is_dpp(), "Problem must be DPP for CvxpyLayer"
        
        # Create CvxpyLayer
        self.layer = CvxpyLayer(
            self.problem,
            parameters=[self.real_mainbus, 
                       self.real_broad,
                       self.real_schlinger,
                       self.real_resnick,
                       self.real_beckman,
                       self.real_braun],
            variables=self.output_variables
        )
        
    def _create_output_variables(self) -> List[cp.Variable]:
        """Create output variables for neural network."""
        output = []
        p_battery, p_impedance = self.device_variables
        
        for t in range(self.T):
            for n in range(self.N):
                if n in self.battery_nodes:
                    output.append(p_battery[f"{n}_{t}"])
                if n in self.impedance_nodes:
                    output.append(p_impedance[f"{n}_{t}"])
                    
        return output
        
    def forward(self, 
                real_mainbus: torch.Tensor,
                real_broad: torch.Tensor,
                real_schlinger: torch.Tensor,
                real_resnick: torch.Tensor,
                real_beckman: torch.Tensor,
                real_braun: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.
        
        Args:
            real_mainbus: Main bus real power demand
            real_broad: Broad node real power demand
            real_schlinger: Schlinger node real power demand
            real_resnick: Resnick node real power demand
            real_beckman: Beckman node real power demand
            real_braun: Braun node real power demand
            
        Returns:
            Neural network output
        """
        return self.layer(real_mainbus, real_broad, real_schlinger, real_resnick, 
                         real_beckman, real_braun)
        
    def compute_loss(self,
                     real_mainbus: torch.Tensor,
                     real_broad: torch.Tensor,
                     real_schlinger: torch.Tensor,
                     real_resnick: torch.Tensor,
                     real_beckman: torch.Tensor,
                     real_braun: torch.Tensor,
                     device_variable: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss function for training.
        
        Args:
            real_mainbus: Main bus real power demand
            real_broad: Broad node real power demand
            real_schlinger: Schlinger node real power demand
            real_resnick: Resnick node real power demand
            real_beckman: Beckman node real power demand
            real_braun: Braun node real power demand
            device_variable: Device variables from forward pass
            
        Returns:
            Computed loss value
        """
        p_battery, p_impedance = {}, {}
        
        # Parse device variables
        for t in range(self.T):
            p_impedance[f"{NODE_INDICES['MAINBUS']}_{t}"] = device_variable[t * 5 + 0]
            p_impedance[f"{NODE_INDICES['BROAD']}_{t}"] = device_variable[t * 5 + 1]
            p_impedance[f"{NODE_INDICES['SCHLINGER']}_{t}"] = device_variable[t * 5 + 2]
            p_battery[f"{NODE_INDICES['RESNICK']}_{t}"] = device_variable[t * 5 + 2]
            p_battery[f"{NODE_INDICES['BECKMAN']}_{t}"] = device_variable[t * 5 + 3]
            p_battery[f"{NODE_INDICES['BRAUN']}_{t}"] = device_variable[t * 5 + 4]
            
        def compute_time_cost(t: int) -> torch.Tensor:
            """Compute cost for time period t."""
            # Impedance node costs
            impedance_cost = (real_mainbus[t].unsqueeze(-1) - p_impedance[f"{NODE_INDICES['MAINBUS']}_{t}"].real +
                            real_broad[t].unsqueeze(-1) - p_impedance[f"{NODE_INDICES['BROAD']}_{t}"].real +
                            real_schlinger[t].unsqueeze(-1) - p_impedance[f"{NODE_INDICES['SCHLINGER']}_{t}"].real)
            
            # Battery node costs
            battery_cost = (real_resnick[t].unsqueeze(-1) - p_battery[f"{NODE_INDICES['RESNICK']}_{t}"].real +
                          real_beckman[t].unsqueeze(-1) - p_battery[f"{NODE_INDICES['BECKMAN']}_{t}"].real +
                          real_braun[t].unsqueeze(-1) - p_battery[f"{NODE_INDICES['BRAUN']}_{t}"].real)
            
            return (impedance_cost + battery_cost)[0]
            
        return sum(self.RTP * compute_time_cost(t) for t in range(self.T))
        
    def torch_loss(self,
                   real_mainbus: torch.Tensor,
                   real_broad: torch.Tensor,
                   real_schlinger: torch.Tensor,
                   real_resnick: torch.Tensor,
                   real_beckman: torch.Tensor,
                   real_braun: torch.Tensor) -> torch.Tensor:
        """
        Compute PyTorch loss for training.
        
        Args:
            real_mainbus: Main bus real power demand
            real_broad: Broad node real power demand
            real_schlinger: Schlinger node real power demand
            real_resnick: Resnick node real power demand
            real_beckman: Beckman node real power demand
            real_braun: Braun node real power demand
            
        Returns:
            PyTorch loss tensor
        """
        device_variable = self.forward(real_mainbus, real_broad, real_schlinger, real_resnick, 
                                     real_beckman, real_braun)
        return self.compute_loss(real_mainbus, real_broad, real_schlinger, real_resnick, 
                               real_beckman, real_braun, device_variable)


# Utility functions
def load_config(config_file: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def str_to_matrices(arr: List[List[str]]) -> np.ndarray:
    """Convert string array to complex matrices."""
    return np.array([[complex(val.replace(' ', '')) for val in row] for row in arr], 
                   dtype=np.complex64)

def create_power_demands_from_config(config: dict) -> PowerDemands:
    """Create PowerDemands object from configuration."""

    # Load demand data
    demands_mainbus = torch.tensor(list(map(float, config['demands_mainbus'][0].split()))) / 1000
    demands_broad = torch.tensor(list(map(float, config['demands_broad'][0].split()))) / 1000
    demands_schlinger = torch.tensor(list(map(float, config['demands_schlinger'][0].split()))) / 1000
    demands_resnick = torch.tensor(list(map(float, config['demands_resnick'][0].split()))) / 1000
    demands_beckman = torch.tensor(list(map(float, config['demands_beckman'][0].split()))) / 1000
    demands_braun = torch.tensor(list(map(float, config['demands_braun'][0].split()))) / 1000
    
    return PowerDemands(
        real_mainbus=demands_mainbus.unsqueeze(0),
        real_broad=demands_broad.unsqueeze(0),
        real_schlinger=demands_schlinger.unsqueeze(0),
        real_resnick=demands_resnick.unsqueeze(0),
        real_beckman=demands_beckman.unsqueeze(0),
        real_braun=demands_braun.unsqueeze(0)
    )


def main():
    """Main execution function."""
    # Load configuration
    config = load_config("config.yaml")
    
    # Create grid parameters
    Y_A = {key: str_to_matrices(value) for key, value in config['Y_A'].items()}
    grid_params = GridParameters(
        T=config['T'],
        N=len(config['adj_matrix']),
        Y_A=Y_A,
        adj_matrix=np.array(config['adj_matrix']),
        node_class=config['node_class'],
        RTP=float(0.118580418)
    )
    
    # Create power demands
    power_demands = create_power_demands_from_config(config)
    
    # Create and solve OPF problem
    opf = OptimalPowerFlowNeuralNetwork(
        grid_params.T, grid_params.N, grid_params.Y_A, grid_params.adj_matrix, 
        grid_params.node_class, grid_params.RTP,
        power_demands.real_mainbus, power_demands.real_broad, power_demands.real_schlinger, 
        power_demands.real_resnick, power_demands.real_beckman, power_demands.real_braun
    )
    
    # Compute loss
    torch_loss = opf.torch_loss(
        power_demands.real_mainbus,
        power_demands.real_broad,
        power_demands.real_schlinger,
        power_demands.real_resnick,
        power_demands.real_beckman,
        power_demands.real_braun
    )
    
    print(f"PyTorch loss: {torch_loss}")


if __name__ == "__main__":
    main()