import yaml
import numpy as np
import torch

def load_config(config_file):
    """
    Load configuration from a YAML file.
    
    Args:
        config_file (str): Path to the configuration file.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def str_to_matrices(arr):
    """
    Convert a 2D array of strings to a 2D numpy array of complex numbers.
    
    Args:
        arr (list of list of str): 2D array of strings representing complex numbers.
    """
    return np.array([[complex(val.replace(' ', '')) for val in row] for row in arr], dtype=np.complex64)

def str_to_vector(arr):
    """
    Convert a list of strings to a numpy array of complex numbers.
    
    Args:
        arr (list of str): List of strings representing complex numbers.
    """
    return np.array([complex(val) for val in arr], dtype=np.complex64)

def split_real_imag(tensor):
    """
    Split a tensor into its real and imaginary parts.
    
    Args:
        tensor (np.ndarray): Numpy array of complex numbers.
    """
    real_part = (tensor.real).unsqueeze(-1)
    imag_part = (tensor.imag).unsqueeze(-1)
    return real_part, imag_part