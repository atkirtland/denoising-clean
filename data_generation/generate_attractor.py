"""

RIPS AFRL 2020
Version August 19, 2020

This file generates state variable signals from selected known attractors to be used as test data.
It should be run before any experiment script.
Outputs CSV file of signals according to parameters assigned in the main function, into the data directory.
Includes Lorenz, Chen, Rossler, and Lorenz96 attractors.

"""

import numpy as np
import sys
import os
import time
from scipy.integrate import odeint
from scipy.interpolate import griddata, Rbf
from numba import jit


def generate_lorenz(rho, sigma, beta, timestep, start_time, end_time, state0):

    """
    This function generates the Lorenz system data given the specified parameters.

    Attractor-specific parameters:
        rho: A shape-changing parameter to the system
        sigma: A shape-changing parameter to the system
        beta: A shape-changing parameter to the system
        state0: Initial conditions of state variables

    Returns:
        states: The generated Lorenz state variable data
    """

    # jit compiles the function with lower level code for computational efficiency.
    # Gives the same values with/without the @jit.
    # @jit
    def f(state, t):
        x, y, z = state
        # These equations specify the ODEs of the Lorenz system
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

    # Solves system of equations for specified parameters and times
    t = np.arange(start_time, end_time, timestep)
    # states = odeint(f, state0, t, full_output=1)
    states = odeint(f, state0, t)

    return states


def generate_chen(a, b, c, timestep, start_time, end_time, state0):

    """
    This function generates the Chen system data given the specified parameters.

    Attractor-specific parameters:
        a: A shape-changing parameter to the system
        b: A shape-changing parameter to the system
        c: A shape-changing parameter to the system
        state0: Initial conditions of state variables
    Returns:
        states: The generated Chen state variable data
    """

    # jit compiles the function to lower level code
    # I tested and it gives the same values with/without the @jit, so there should be no reason to remove it
    # @jit
    def f(state, t):
        x, y, z = state
        # These equations specify the ODEs of the Chen system
        return a * (y - x), (c - a) * x - x * z + c * y, x * y - b * z

    t = np.arange(start_time, end_time, timestep)
    states = odeint(f, state0, t)

    return states


def generate_rossler(a, b, c, timestep, start_time, end_time, state0):

    """
    This function generates the Rossler system data given the specified parameters.

    Attractor-specific parameters:
        a: A shape-changing parameter to the system
        b: A shape-changing parameter to the system
        c: A shape-changing parameter to the system
        state0: Initial conditions of state variables
    Returns:
        states: The generated Rossler state variable data
    """

    # jit compiles the function to lower level code
    # Same values are given with/without the @jit, so there is no reason to remove it
    @jit
    def f(state, t):
        x, y, z = state
        # These equations specify the ODEs of the Rossler system
        return -y - z, x + a * y, b * x - c * z + x * z

    t = np.arange(start_time, end_time, timestep)
    states = odeint(f, state0, t)

    return states


def generate_lorenz96(N, F, timestep, start_time, end_time, state0):

    """
    This function generates the Lorenz96 system data given the specified parameters.
    Attractor-specific parameters:
        N: A parameter to the system: Number of variables
        F: A parameter to the system: Forcing factor
        state0: Initial conditions of state variables
    Returns:
        states: The generated Lorenz96 state variable data
    """

    @jit
    def lorenz96(x, t):
        # Compute state derivatives of Lorenz96 model
        d = np.empty(N)
        for i in range(N):
            d[i] = (x[(i + 1) % N] - x[(i - 2) % N]) * x[(i - 1) % N] - x[i]
        d = d + F

        # Return the state derivatives
        return d

    t = np.arange(start_time, end_time, timestep)
    states = odeint(lorenz96, state0, t)

    return states


def main():
    # Record time it takes to run code
    start = time.time()

    # Select which attractor to use: "lorenz", "chen", "rossler", or "lorenz96"
    system = "chen"

    """
    These parameters are not specific to the attractor used.
    
    timestep: Controls how fine the data is -- typically 0.01
    start_time: Time at which to start generating data -- typically 0.0
    end_time: Time at which to stop generating data -- varies based on desired training time, but typically > 10000
    """

    timestep = 0.001
    start_time = 0.0
    end_time = 100000.0

    # Create path to store data if necessary
    if not os.path.exists("data"):
        os.makedirs("data")

    # Retrieve data from generate_ATTRACTOR function
    if system == "lorenz":

        """
        Adjust system parameters here, as explained in generate_lorenz function definition
        Typical values:
            rho = 30.0
            sigma = 10.0
            beta = 8.0/3.0
            state0 = [1.0, 1.0, 1.0]
        """

        rho = 30.0
        sigma = 10.0
        beta = 8.0 / 3.0
        state0 = [-0.1, 0.5, -0.6]
        state0 = [1.0, 1.0, 1.0]

        data = generate_lorenz(rho, sigma, beta, timestep, start_time, end_time, state0)

    elif system == "chen":

        """
        Adjust system parameters here, as explained in generate_chen function definition
        Typical values:
        a = 40.0
        b = 3.0
        c = 28.0
        state0 = [-0.1, 0.5, -0.6]
        """

        a = 40.0
        b = 3.0
        c = 28.0
        state0 = [-0.1, 0.5, -0.6]

        data = generate_chen(a, b, c, timestep, start_time, end_time, state0)

    elif system == "rossler":

        """
        Adjust system parameters here, as explained in generate_rossler function definition
        Typical values:
        a = 0.38
        b = 0.35
        c = 4.5
        state0 = [2.0, 2.0, 2.0]
        """

        a = 0.38
        b = 0.35
        c = 4.5
        state0 = [2.0, 2.0, 2.0]

        data = generate_rossler(a, b, c, timestep, start_time, end_time, state0)

    elif system == "lorenz96":

        """
            Adjust system parameters here, as explained in generate_rossler function definition
            Typical values:
            a = 0.38
            b = 0.35
            c = 4.5
            state0 = [2.0, 2.0, 2.0]
        """

        N = 12
        F = 8

        state0 = F * np.random.normal(size=N)

        data = generate_lorenz96(N, F, timestep, start_time, end_time, state0)

    # Name and save CSV files in data directory
    for i in range(len(data[0])):
        np.save(os.path.join("data", system + "_" + str(i) + ".npy"), data[:, i])

    # Record time it takes to run code
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    main()
