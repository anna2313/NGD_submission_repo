import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, file: str, Xdim: int):
        """Initialize the dataset by reading a CSV file."""
        self.set = pd.read_csv(file)
        self.Xdim = Xdim

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.set)

    def __getitem__(self, idx: int):
        """Retrieve an item by index."""
        data_point = self.set.iloc[idx]
        X = torch.from_numpy(
            data_point[: self.Xdim].to_numpy(copy=True)
        )  # Convert to NumPy array for PyTorch compatibility
        y = torch.from_numpy(data_point[self.Xdim :].to_numpy(copy=True))
        return X, y


class CustomDatasetClassification(Dataset):
    def __init__(self, file: str):
        """Initialize the dataset by reading a CSV file."""
        self.set = pd.read_csv(file)

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.set)

    def __getitem__(self, idx: int):
        """Retrieve an item by index."""
        data_point = self.set.iloc[idx]
        X = torch.from_numpy(
            data_point[:-1].to_numpy(copy=True)
        )  # Convert to NumPy array for PyTorch compatibility
        # Keep a singleton class dimension so targets match model output shape [batch, 1].
        y = torch.tensor([data_point.iloc[-1]], dtype=torch.float64)
        return X, y


def generate_normal_data(
    sigma: np.ndarray,
    theta: np.ndarray,
    output_file: str,
    number_of_datapoints: int,
    mu: np.ndarray = None,
    allow_multiple_theta: bool = False,
):
    """
    Generate synthetic data with normal noise and save it to a CSV file.

    Args:
        sigma (np.ndarray): Standard deviation for each feature.
        theta (np.ndarray): Coefficients for the linear combination.
        output_file (str): Name of the CSV file to save the data.
        number_of_datapoints (int): Number of data points to generate.
        mu (np.ndarray): Mean for each feature. Defaults to None.
        allow_multiple_theta (bool): Flag to allow multiple theta values.
    """

    if sigma is None:
        raise ValueError("Sigma must be provided.")
    if theta is None:
        raise ValueError("Theta must be provided.")
    if output_file is None:
        raise ValueError("Output file must be provided.")
    if number_of_datapoints is None:
        raise ValueError("Number of data points must be provided.")
    if mu is not None and len(mu) != len(sigma):
        raise ValueError("Mean and sigma must have the same length.")
    elif mu is None:
        mu = np.ones(len(sigma))
    if allow_multiple_theta:
        if len(theta) != len(mu):
            raise ValueError("Theta and mu must have the same length.")

    if allow_multiple_theta:
        dim = len(theta[0, :])
    else:
        dim = len(theta)
    x = np.random.normal(
        loc=0.0, scale=1.0, size=(number_of_datapoints, dim)
    )  # can we make y zero mean and unit variance? (Now we depend on dimension of theta)
    y = np.ones(shape=(number_of_datapoints, len(sigma)))

    for i in range(len(sigma)):
        if allow_multiple_theta:
            y[:, i] = np.random.normal(loc=x @ theta[i, :].T * mu[i], scale=sigma[i])
        else:
            y[:, i] = np.random.normal(loc=x @ theta.T * mu[i], scale=sigma[i])

    # Create DataFrame directly from NumPy arrays
    data = np.concatenate((x, y), axis=1)
    df = pd.DataFrame(
        data,
        columns=[f"x{i}" for i in range(dim)] + [f"y{s}" for s in range(len(sigma))],
    )

    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")


def generate_normal_sin_data(
    sigma: np.ndarray,
    output_file: str,
    theta: np.ndarray,
    number_of_datapoints: int,
):
    """
    Generate synthetic data with normal noise and save it to a CSV file.

    Args:
        sigma (np.ndarray): Standard deviation for each feature.
        theta (np.ndarray): scale factor for the sine function.
        output_file (str): Name of the CSV file to save the data.
        number_of_datapoints (int): Number of data points to generate.
    """

    if sigma is None:
        raise ValueError("Sigma must be provided.")
    if theta is None:
        raise ValueError("Theta must be provided.")
    if len(theta) != 1:
        raise ValueError("Theta must be a scalar or a single-element array.")
    if output_file is None:
        raise ValueError("Output file must be provided.")
    if number_of_datapoints is None:
        raise ValueError("Number of data points must be provided.")
    if sigma < 0.0:
        raise ValueError("Sigma must be non-negative.")

    dim = len(sigma)
    x = np.random.uniform(low=-1.0, high=1.0, size=(number_of_datapoints, dim))
    y = np.ones(shape=(number_of_datapoints, dim))

    for i in range(dim):
        y[:, i] = np.random.normal(loc=np.sin(x[:, i] / theta), scale=sigma[i])

    # Create DataFrame directly from NumPy arrays
    data = np.concatenate((x, y), axis=1)
    df = pd.DataFrame(
        data,
        columns=[f"x{i}" for i in range(dim)] + [f"y{s}" for s in range(dim)],
    )

    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")


def generate_circular_classification_data(
    r_inner: float,
    r_outer: float,
    sigma: float,
    number_of_datapoints: int,
    output_file: str,
    sigma_outer: float = None,
):
    """
    Generate synthetic circular classification data and save it to a CSV file.

    Args:
        r_inner (float): Radius of the inner circle.
        r_outer (float): Radius of the outer circle.
        sigma (float): Standard deviation of the noise.
        sigma_outer (float): Standard deviation of the noise for the outer circle. If not given or set to None, it will be the same as sigma.
        number_of_datapoints (int): Total number of data points to generate.
        output_file (str): Name of the CSV file to save the data.
    """

    if r_inner < 0 or r_outer < 0:
        raise ValueError("Radii must be non-negative.")
    if sigma < 0.0:
        raise ValueError("Sigma must be non-negative.")
    if sigma_outer is None:
        sigma_outer = sigma
    if sigma_outer < 0.0:
        raise ValueError("Sigma outer must be non-negative.")
    if number_of_datapoints <= 0:
        raise ValueError("Number of data points must be positive.")
    if output_file is None:
        raise ValueError("Output file must be provided.")

    # Generate angles for inner and outer circles
    n_inner = number_of_datapoints // 2
    n_outer = number_of_datapoints - n_inner
    angles_inner = np.random.uniform(0, 2 * np.pi, n_inner)
    angles_outer = np.random.uniform(0, 2 * np.pi, n_outer)

    # Generate points for inner circle
    x_inner = np.column_stack(
        (
            r_inner * np.cos(angles_inner)
            + np.random.normal(0, sigma, size=angles_inner.shape),
            r_inner * np.sin(angles_inner)
            + np.random.normal(0, sigma, size=angles_inner.shape),
        )
    )
    y_inner = np.ones(shape=(n_inner,))  # Label for inner circle

    # Generate points for outer circle
    x_outer = np.column_stack(
        (
            r_outer * np.cos(angles_outer)
            + np.random.normal(0, sigma_outer, size=angles_outer.shape),
            r_outer * np.sin(angles_outer)
            + np.random.normal(0, sigma_outer, size=angles_outer.shape),
        )
    )
    y_outer = np.zeros(shape=(n_outer,))  # Label for outer circle

    # Combine inner and outer circle data
    x = np.concatenate((x_inner, x_outer), axis=0)
    y = np.concatenate((y_inner, y_outer))

    # Create DataFrame directly from NumPy arrays
    data = np.column_stack((x, y))
    df = pd.DataFrame(data, columns=["x", "y", "label"])

    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
