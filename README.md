# MPPI-MPC-Controllers

## CBF, MPC & MPPI Controllers

Welcome to my repository! This repo contains implementations of:
- Control Barrier Function (CBF)
- Model Predictive Control (MPC)
- Model Predictive Path Integral (MPPI)
- My project work in collaboration with DRiverless Intelligent VEhicle (DRIVE) Lab, Robotics Institute, Carnegie Mellon University under the guidance of Dr. John Dolan and Yiwei Lyu. This work is still under progress.

---

## How to Use This Repo

To get started, clone this repository to your local system:

```sh
git clone https://github.com/devodita/MPPI-MPC-Controllers.git
```

Navigate to this directory:

```sh
cd MPPI-MPC-Controllers
```

---

## Running CBF (Control Barrier Function)

1. **Navigate to the `CBF` directory:**
   ```sh
   cd CBF
   ```

2. **Run the main script:**
   ```sh
   python CBF_function.py
   ```

3. **Play around with the `config` file** to explore different scenarios and configurations.

---

## Running MPC

1. **Navigate to the `MPC` directory:**
   ```sh
   cd MPC
   ```

2. **Create and activate a Conda environment:**
   ```sh
   conda create --name MPC python=3.8 
   conda activate MPC
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

---

## Running MPPI

1. **Navigate to the `mppi` directory:**
   ```sh
   cd mppi
   ```
2.  **Install the required dependencies:**
     ```sh
     pip install pygame
     ```
     ```sh
     pip install gym
     ```

3. **Run the main script:**
   ```sh
   python MPPI.py
   ```

4. Play around with the constants to explore different scenarios and configurations.

---

## Running the project work simulation

1. **Navigate to the `Project` directory:**
   ```sh
   cd Project
   ```

2. **Run the simulation:**
   ```sh
   python run_visualization_demo.py
   ```

---

## Acknowledgements

This repository is a compilation work for enthusiasts like myself who are exploring controllers. The contents have been sourced from various references, including:
- Official implementations of **CasADi_Python**
- **Mark Misin**'s implementations and contributions to control systems

Special thanks to the open-source community and my seniors who guided me in this process.

---

### Further Reading

You can check my ongoing work and updates in the following Google Slides presentation:

👉 [Click here to view the progress](https://docs.google.com/presentation/d/1csi9oMNgO3u7VMuONpWCDqABk5728hYO6HDcmZNTCbM/edit?usp=sharing)

---

## Contributing

Feel free to contribute to this repository! If you have any suggestions or improvements, open an issue or submit a pull request to make this repo more informative.

**Happy coding!**
