# MPPI-MPC-Controllers

# MPPI & MPC Controllers

Welcome to my repository! This repo contains implementations of Model Predictive Path Integral (MPPI) and Model Predictive Control (MPC) controllers.

## How to Use This Repo

To get started, clone this repository to your local system:
```sh
git clone https://github.com/devodita/MPPI-MPC-Controllers.git
```
Navigate to this directory:
   ```sh
   cd MPPI-MPC-Controllers
   ```

### Running MPC

1. **Create and activate a Conda environment:**
   ```sh
   conda create --name MPC python=3.8  # Adjust Python version if needed
   conda activate MPC
   ```

2. **Navigate to the `MPC` directory:**
   ```sh
   cd MPC
   ```

3. **Inside the `MPC` directory, you will find two implementations:**
   - `casadi/` - Uses **CasADi** for MPC implementation.
   - `qpsolvers/` - A simpler implementation using a **QP solver**.

4. **Install the required dependencies:**
   - For CasADi:
     ```sh
     pip install casadi
     ```
   - For QP solvers:
     ```sh
     pip install qpsolvers
     pip install qpsolvers[cvxopt]
     ```

5. **Choose the implementation you want to run:**
   ```sh
   cd casadi  # or cd qpsolvers
   ```

6. **Run the script to see the simulations and visualizations:**
   ```sh
   python filename.py  # change filename as per requirement
   ```

### Running MPPI
1. Navigate to the `mppi` directory:
   ```sh
   cd mppi
   ```
2. Read the `README.md` inside the `mppi` directory for a detailed explanation on how to use it.

## Acknowledgements
This repository is a compilation work for enthusiasts like myself who are exploring controllers. The contents have been sourced from various references, including:
- Official implementations of **CasADi_Python**.
- **Mark Misin**'s implementations and contributions to control systems.
- **ROS 2 Navigation Systems**.

Special thanks to the open-source community and my seniors who guided me in this process.

## Contributing
Feel free to contribute to this repository! If you have any suggestions or improvements, open an issue or submit a pull request to make this repo more informative.

Happy coding!

