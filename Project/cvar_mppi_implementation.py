import numpy as np
from scipy.stats import norm
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import time

class VehicleState:
    """Represents the state of a vehicle in the environment."""
    def __init__(self, x=0.0, y=0.0, vx=0.0, vy=0.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        
    def as_array(self) -> np.ndarray:
        """Convert state to numpy array."""
        return np.array([self.x, self.y, self.vx, self.vy])
    
    @staticmethod
    def from_array(arr: np.ndarray) -> 'VehicleState':
        """Create state from numpy array."""
        return VehicleState(arr[0], arr[1], arr[2], arr[3])
    
    def copy(self) -> 'VehicleState':
        """Create a copy of the state."""
        return VehicleState(self.x, self.y, self.vx, self.vy)


class Vehicle:
    """Base class for all vehicles in the simulation."""
    def __init__(self, initial_state: VehicleState):
        self.state = initial_state
        self.dt = 0.1  # Time step for dynamics
        
    def update(self, control: np.ndarray, noise: Optional[np.ndarray] = None) -> None:
        """Update the vehicle state based on control input and optional noise."""
        raise NotImplementedError("Subclasses must implement this method")


class EgoVehicle(Vehicle):
    """The ego vehicle controlled by the MPPI algorithm."""
    def __init__(self, initial_state: VehicleState, dt: float = 0.1):
        super().__init__(initial_state)
        self.dt = dt
        # Simple physical limits
        self.max_accel = 3.0  # m/s^2
        self.max_steer = 0.5  # radians
        
    def update(self, control: np.ndarray, noise: Optional[np.ndarray] = None) -> None:
        """
        Update vehicle state using a simple kinematic model
        control[0]: acceleration
        control[1]: steering angle (simplified as direct lateral acceleration)
        """
        accel = np.clip(control[0], -self.max_accel, self.max_accel)
        lateral_accel = np.clip(control[1], -self.max_steer, self.max_steer)
        
        # Add noise if provided
        if noise is not None:
            accel += noise[0]
            lateral_accel += noise[1]
        
        # Simple kinematic model
        self.state.x += self.state.vx * self.dt
        self.state.y += self.state.vy * self.dt
        self.state.vx += accel * self.dt
        self.state.vy += lateral_accel * self.dt
    
    def linearize_dynamics(self, ref_state: np.ndarray, ref_control: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize the vehicle dynamics around a reference state and control
        Returns A, B matrices for the linear system dx = A*x + B*u
        """
        # For this simple model, linearization is straightforward
        # State = [x, y, vx, vy]
        # Control = [accel, lateral_accel]
        A = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        B = np.array([
            [0, 0],
            [0, 0],
            [self.dt, 0],
            [0, self.dt]
        ])
        
        return A, B


class HumanVehicle(Vehicle):
    """Human-driven vehicle with parameterized behavior model."""
    def __init__(self, initial_state: VehicleState, dt: float = 0.1):
        super().__init__(initial_state)
        self.dt = dt
        self.max_accel = 4.0  # m/s^2
        self.max_steer = 0.6  # radians
        
    def compute_control(self, ego_state: VehicleState, params: Dict[str, float], 
                         noise_std: Optional[float] = None) -> np.ndarray:
        """
        Compute human control action based on IDM-like model with parameters
        
        params:
            - aggressiveness: How aggressively the human responds (higher = more aggressive)
            - desired_gap: Preferred gap to maintain from other vehicles
            - desired_speed: Target speed for the human
            - reaction_time: Simulated reaction delay
        """
        # Extract parameters with defaults
        aggressiveness = params.get('aggressiveness', 1.0)
        desired_gap = params.get('desired_gap', 15.0)
        desired_speed = params.get('desired_speed', 20.0)
        
        # Compute longitudinal control (IDM-inspired)
        # Calculate relative position and speed
        rel_x = ego_state.x - self.state.x
        rel_v = ego_state.vx - self.state.vx
        
        # Compute gap and desired acceleration
        current_gap = max(0.1, abs(rel_x))
        gap_error = current_gap - desired_gap
        
        # Speed control
        speed_error = desired_speed - self.state.vx
        
        # Combine into acceleration command (simplified IDM)
        accel = 0.5 * speed_error
        
        # Gap control (if ego is ahead)
        if rel_x > 0 and current_gap < 2 * desired_gap:
            accel += 0.5 * aggressiveness * gap_error - 0.5 * rel_v
            
        # Lateral control (simplified lane keeping)
        target_y = 0.0  # Assume lane center is at y=0
        if ego_state.y - self.state.y < desired_gap/2 and abs(rel_x) < desired_gap:
            # Avoid ego vehicle by steering away
            lateral_offset = 1.0 * np.sign(self.state.y)
            target_y += lateral_offset * aggressiveness
        
        lateral_accel = 0.5 * (target_y - self.state.y)
        
        # Apply limits
        accel = np.clip(accel, -self.max_accel, self.max_accel)
        lateral_accel = np.clip(lateral_accel, -self.max_steer, self.max_steer)
        
        # Add noise if specified
        if noise_std is not None:
            accel += np.random.normal(0, noise_std)
            lateral_accel += np.random.normal(0, noise_std)
            
        return np.array([accel, lateral_accel])
    
    def update(self, ego_state: VehicleState, params: Dict[str, float], 
               noise_std: Optional[float] = None) -> None:
        """Update the human vehicle state based on the ego vehicle state and human parameters."""
        control = self.compute_control(ego_state, params, noise_std)
        
        # Simple kinematic model
        self.state.x += self.state.vx * self.dt
        self.state.y += self.state.vy * self.dt
        self.state.vx += control[0] * self.dt
        self.state.vy += control[1] * self.dt


class HumanBeliefModel:
    """Maintains and updates a belief over human driver parameters."""
    def __init__(self, n_particles: int = 100):
        self.n_particles = n_particles
        # Initialize particles for human parameters
        self.particles = []
        self.weights = np.ones(n_particles) / n_particles
        
        # Initialize with a prior distribution over typical parameters
        for _ in range(n_particles):
            self.particles.append({
                'aggressiveness': np.random.uniform(0.5, 2.0),
                'desired_gap': np.random.uniform(10.0, 30.0),
                'desired_speed': np.random.uniform(15.0, 25.0)
            })
    
    def get_mean_params(self) -> Dict[str, float]:
        """Get the weighted mean of the parameter distribution."""
        mean_params = {}
        for key in self.particles[0].keys():
            values = np.array([p[key] for p in self.particles])
            mean_params[key] = np.sum(values * self.weights)
        return mean_params
    
    def get_sample_params(self) -> Dict[str, float]:
        """Sample parameters from the current belief."""
        idx = np.random.choice(self.n_particles, p=self.weights)
        return self.particles[idx].copy()
    
    def update_belief(self, observed_action: np.ndarray, predicted_actions: List[np.ndarray], 
                      observation_noise_std: float = 0.5) -> None:
        """
        Update the belief over human parameters using Bayes rule
        
        Args:
            observed_action: The actual control action taken by the human
            predicted_actions: The predicted actions for each parameter particle
            observation_noise_std: Standard deviation of observation noise
        """
        # Compute likelihood of each parameter given observation
        for i in range(self.n_particles):
            # Likelihood under Gaussian noise model
            error = np.linalg.norm(observed_action - predicted_actions[i])
            likelihood = norm.pdf(error, loc=0, scale=observation_noise_std)
            # Update weight using Bayes rule
            self.weights[i] *= likelihood
            
        # Normalize weights
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            # Handle weight collapse by resetting to uniform
            self.weights = np.ones(self.n_particles) / self.n_particles
            
        # Resample if effective number of particles is too low
        n_eff = 1.0 / np.sum(np.square(self.weights))
        if n_eff < self.n_particles / 2:
            self._resample()
    
    def _resample(self) -> None:
        """Resample particles based on weights to maintain diversity."""
        indices = np.random.choice(
            self.n_particles, size=self.n_particles, replace=True, p=self.weights
        )
        new_particles = [self.particles[i].copy() for i in indices]
        
        # Add small noise to parameters to maintain diversity
        for particle in new_particles:
            particle['aggressiveness'] += np.random.normal(0, 0.05)
            particle['desired_gap'] += np.random.normal(0, 0.5)
            particle['desired_speed'] += np.random.normal(0, 0.5)
            
            # Ensure parameters stay in reasonable bounds
            particle['aggressiveness'] = np.clip(particle['aggressiveness'], 0.1, 3.0)
            particle['desired_gap'] = np.clip(particle['desired_gap'], 5.0, 40.0)
            particle['desired_speed'] = np.clip(particle['desired_speed'], 10.0, 30.0)
            
        self.particles = new_particles
        self.weights = np.ones(self.n_particles) / self.n_particles


class CovarianceSteeringController:
    """Computes feedback gains for steering the covariance of trajectories."""
    def __init__(self, state_dim: int, control_dim: int, horizon: int, dt: float = 0.1):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.horizon = horizon
        self.dt = dt
        self.Q = np.eye(state_dim)  # State cost matrix
        self.R = np.eye(control_dim)  # Control cost matrix
        
    def compute_gains(self, A_list: List[np.ndarray], B_list: List[np.ndarray], 
                      target_cov: np.ndarray) -> List[np.ndarray]:
        """
        Compute feedback gains to achieve a target terminal covariance
        
        Args:
            A_list: List of A matrices for linearized dynamics at each time step
            B_list: List of B matrices for linearized dynamics at each time step
            target_cov: Target terminal state covariance
            
        Returns:
            List of feedback gain matrices K for each time step
        """
        # Initialize with terminal cost
        P = np.linalg.inv(target_cov)  # Terminal cost to achieve covariance
        K_list = []
        
        # Backward recursion to compute gains
        for t in range(self.horizon-1, -1, -1):
            A = A_list[t]
            B = B_list[t]
            
            # LQR feedback gain
            K = -np.linalg.inv(self.R + B.T @ P @ B) @ B.T @ P @ A
            K_list.insert(0, K)
            
            # Update cost-to-go
            P = self.Q + A.T @ P @ A + A.T @ P @ B @ K
            
        return K_list


class CVaRMPPI:
    """CVaR-integrated Model Predictive Path Integral controller."""
    def __init__(self, ego_vehicle: EgoVehicle, human_vehicle: HumanVehicle, 
                 belief_model: HumanBeliefModel, config: Dict = None):
        # Vehicle and belief models
        self.ego = ego_vehicle
        self.human = human_vehicle
        self.belief = belief_model
        
        # Default configuration
        self.config = {
            'dt': 0.1,                 # Time step
            'horizon': 10,             # Number of time steps in prediction
            'n_samples': 200,          # Number of trajectory samples
            'temperature': 1.0,        # Temperature parameter for weighting
            'control_dim': 2,          # Dimension of control vector [accel, steering]
            'state_dim': 4,            # Dimension of state vector [x, y, vx, vy]
            'control_noise_std': 0.3,  # Standard deviation of control noise
            'process_noise_std': 0.1,  # Standard deviation of process noise
            'cvar_alpha': 0.9,         # CVaR confidence level (e.g., 90%)
            'cvar_samples': 10,        # Number of samples for CVaR estimation
            'cvar_weight': 10.0,       # Weight of CVaR term in cost
            'cvar_threshold': 100.0,   # Threshold for CVaR constraint
            'use_covariance_steering': True,  # Whether to use covariance steering
            'target_cov_scale': 0.5,   # Scaling factor for target covariance
        }
        
        # Update config with provided values
        if config is not None:
            self.config.update(config)
            
        # Initialize control sequence
        self.horizon = self.config['horizon']
        self.control_dim = self.config['control_dim']
        self.state_dim = self.config['state_dim']
        self.u_nom = np.zeros((self.horizon, self.control_dim))
        
        # Initialize covariance steering controller
        self.cov_steering = CovarianceSteeringController(
            self.state_dim, self.control_dim, self.horizon, self.config['dt']
        )
        
    def plan(self, ego_state: VehicleState, human_state: VehicleState) -> np.ndarray:
        """
        Run one iteration of CVaR-MPPI planning
        
        Args:
            ego_state: Current state of the ego vehicle
            human_state: Current state of the human vehicle
            
        Returns:
            Optimal control action for the current time step
        """
        # Initialize variables
        n_samples = self.config['n_samples']
        u_samples = np.zeros((n_samples, self.horizon, self.control_dim))
        costs = np.zeros(n_samples)
        
        # 1. Linearize dynamics around nominal trajectory for covariance steering
        A_list = []
        B_list = []
        x_ref = [ego_state.as_array()]
        
        # Simulate nominal trajectory for linearization
        ego_sim = EgoVehicle(ego_state.copy(), self.config['dt'])
        for t in range(self.horizon):
            A, B = ego_sim.linearize_dynamics(x_ref[-1], self.u_nom[t])
            A_list.append(A)
            B_list.append(B)
            
            # Propagate nominal trajectory
            ego_sim.update(self.u_nom[t])
            x_ref.append(ego_sim.state.as_array())
        
        # 2. Compute target covariance and feedback gains
        if self.config['use_covariance_steering']:
            # Target covariance (could be adapted based on scenario)
            target_pos_var = self.config['target_cov_scale'] * 1.0  # Position variance
            target_vel_var = self.config['target_cov_scale'] * 0.5  # Velocity variance
            target_cov = np.diag([target_pos_var, target_pos_var, target_vel_var, target_vel_var])
            
            # Compute feedback gains
            K_gains = self.cov_steering.compute_gains(A_list, B_list, target_cov)
        else:
            K_gains = [np.zeros((self.control_dim, self.state_dim)) for _ in range(self.horizon)]
        
        # 3. Generate trajectory samples with covariance steering feedback
        human_params = self.belief.get_mean_params()
        for k in range(n_samples):
            # Initialize simulation from current state
            ego_sim = EgoVehicle(ego_state.copy(), self.config['dt'])
            human_sim = HumanVehicle(human_state.copy(), self.config['dt'])
            
            # Generate control noise for this sample
            noise = np.random.normal(
                0, self.config['control_noise_std'], 
                (self.horizon, self.control_dim)
            )
            
            # Roll out trajectory
            states = [ego_sim.state.as_array()]
            controls = []
            
            for t in range(self.horizon):
                # Get state deviation from reference
                state_dev = states[-1] - x_ref[t]
                
                # Apply covariance steering feedback
                feedback = K_gains[t] @ state_dev
                
                # Generate control with nominal + noise + feedback
                u_k = self.u_nom[t] + noise[t] + feedback
                controls.append(u_k)
                
                # Simulate one step
                ego_sim.update(u_k, np.random.normal(0, self.config['process_noise_std'], 2))
                human_sim.update(ego_sim.state, human_params, self.config['process_noise_std'])
                
                # Store new state
                states.append(ego_sim.state.as_array())
            
            # Store control sequence
            u_samples[k] = np.array(controls)
            
            # 4. Evaluate cost of this trajectory
            traj_cost = self._compute_trajectory_cost(states, controls, human_sim)
            
            # 5. Compute CVaR cost component
            cvar_cost = self._evaluate_cvar(ego_state, human_state, u_samples[k], human_params)
            
            # Combine costs
            costs[k] = traj_cost + self.config['cvar_weight'] * cvar_cost
        
        # 6. Compute weights and update nominal control
        weights = self._compute_weights(costs)
        
        # Update nominal control sequence
        u_new = np.zeros_like(self.u_nom)
        for k in range(n_samples):
            u_new += weights[k] * u_samples[k]
        
        self.u_nom = u_new
        
        # Shift control sequence for warm start (not strictly necessary if we replan each step)
        if self.horizon > 1:
            self.u_nom = np.vstack([self.u_nom[1:], np.zeros((1, self.control_dim))])
            
        # Return first control action
        return self.u_nom[0]
    
    def _compute_trajectory_cost(self, states: List[np.ndarray], controls: List[np.ndarray], 
                                human_sim: HumanVehicle) -> float:
        """
        Compute the cost of a trajectory
        
        Args:
            states: List of ego states along the trajectory
            controls: List of controls applied
            human_sim: Human vehicle simulator at the end of trajectory
            
        Returns:
            Total cost of the trajectory
        """
        cost = 0.0
        
        # Target state (end of lane merge)
        target_x = 100.0
        target_y = 0.0
        target_vx = 20.0
        
        # Weights for different cost components
        w_dist = 1.0        # Distance to target
        w_vel = 0.5         # Velocity tracking
        w_lane = 2.0        # Lane keeping
        w_control = 0.1     # Control effort
        w_collision = 50.0  # Collision avoidance
        
        # Evaluate costs along trajectory
        for i in range(1, len(states)):
            state = states[i]
            control = controls[i-1]
            
            # Distance to target
            dist_cost = w_dist * (np.square(state[0] - target_x) + np.square(state[1] - target_y))
            
            # Velocity tracking
            vel_cost = w_vel * np.square(state[2] - target_vx)
            
            # Lane keeping (assuming lane center is at y=0)
            lane_cost = w_lane * np.square(state[1])
            
            # Control effort
            control_cost = w_control * np.sum(np.square(control))
            
            # Collision avoidance (with human vehicle)
            human_state = human_sim.state
            dx = state[0] - human_state.x
            dy = state[1] - human_state.y
            dist = np.sqrt(dx*dx + dy*dy)
            collision_cost = w_collision * np.exp(-dist/2.0) if dist < 5.0 else 0.0
            
            # Sum all costs
            cost += dist_cost + vel_cost + lane_cost + control_cost + collision_cost
        
        return cost
    
    def _evaluate_cvar(self, ego_state: VehicleState, human_state: VehicleState,
                       control_sequence: np.ndarray, human_params: Dict[str, float]) -> float:
        """
        Evaluate CVaR risk measure for a control sequence
        
        Args:
            ego_state: Initial ego vehicle state
            human_state: Initial human vehicle state
            control_sequence: Control sequence to evaluate
            human_params: Parameters for human driver model
            
        Returns:
            CVaR cost component
        """
        cvar_samples = self.config['cvar_samples']
        alpha = self.config['cvar_alpha']
        
        # Generate multiple cost realizations under different disturbances
        costs = []
        
        for _ in range(cvar_samples):
            # Initialize simulation
            ego_sim = EgoVehicle(ego_state.copy(), self.config['dt'])
            human_sim = HumanVehicle(human_state.copy(), self.config['dt'])
            
            # Sample human parameters for this rollout (to model uncertainty)
            sample_params = human_params.copy()
            sample_params['aggressiveness'] += np.random.normal(0, 0.3)
            sample_params['desired_gap'] += np.random.normal(0, 2.0)
            
            # Simulate trajectory with these controls but different disturbances
            states = [ego_sim.state.as_array()]
            controls = []
            
            for t in range(self.horizon):
                u_t = control_sequence[t]
                controls.append(u_t)
                
                # Add more aggressive noise for CVaR evaluation
                process_noise = np.random.normal(0, 2 * self.config['process_noise_std'], 2)
                ego_sim.update(u_t, process_noise)
                human_sim.update(ego_sim.state, sample_params, 2 * self.config['process_noise_std'])
                
                states.append(ego_sim.state.as_array())
            
            # Compute cost with focus on safety-critical aspects
            collision_cost = 0
            for i in range(len(states)):
                state = states[i]
                human_state = human_sim.state
                dx = state[0] - human_state.x
                dy = state[1] - human_state.y
                dist = np.sqrt(dx*dx + dy*dy)
                
                # Higher weight on collisions for CVaR
                if dist < 3.0:
                    collision_cost += 100.0 * np.exp(-dist)
            
            costs.append(collision_cost)
        
        # Sort costs and compute CVaR
        if not costs:
            return 0.0
            
        costs = np.sort(costs)
        cutoff_idx = int(np.ceil(alpha * cvar_samples))
        
        # Handle edge cases
        if cutoff_idx >= len(costs):
            var = costs[-1]
            cvar = costs[-1]
        else:
            var = costs[cutoff_idx]
            cvar = np.mean(costs[cutoff_idx:])
        
        # Compute CVaR constraint cost
        cvar_cost = max(0, cvar - self.config['cvar_threshold']) ** 2
        
        return cvar_cost
        
    def _compute_weights(self, costs: np.ndarray) -> np.ndarray:
        """Compute weights for the MPPI update using softmax."""
        temperature = self.config['temperature']
        
        # Normalize costs to avoid numerical issues
        min_cost = np.min(costs)
        costs = costs - min_cost
        
        # Compute weights
        weights = np.exp(-costs / temperature)
        
        # Avoid division by zero
        sum_weights = np.sum(weights)
        if sum_weights > 1e-12:
            weights /= sum_weights
        else:
            # If all costs are very high, use uniform weights
            weights = np.ones_like(weights) / len(weights)
            
        return weights
        
    def update_belief(self, ego_state: VehicleState, human_state: VehicleState, 
                      observed_action: np.ndarray) -> None:
        """
        Update the belief about human parameters based on observed actions
        
        Args:
            ego_state: Current ego vehicle state
            human_state: Current human vehicle state before action
            observed_action: Actual control action taken by human
        """
        # Generate predictions for each particle
        n_particles = self.belief.n_particles
        predicted_actions = []
        
        for i in range(n_particles):
            params = self.belief.particles[i]
            predicted = self.human.compute_control(ego_state, params)
            predicted_actions.append(predicted)
        
        # Update belief using Bayesian inference
        self.belief.update_belief(observed_action, predicted_actions)


class LaneMergeSimulation:
    """Simulation environment for lane merging scenario."""
    def __init__(self, config: Dict = None):
        # Default configuration
        self.config = {
            'dt': 0.1,
            'simulation_time': 15.0,
            'ego_init_x': 0.0,
            'ego_init_y': 3.5,  # Assume ego starts in right lane
            'ego_init_vx': 15.0,
            'ego_init_vy': 0.0,
            'human_init_x': 20.0,
            'human_init_y': 0.0,  # Human in left lane
            'human_init_vx': 18.0,
            'human_init_vy': 0.0,
            'true_human_params': {
                'aggressiveness': 1.2,
                'desired_gap': 20.0,
                'desired_speed': 20.0
            },
            'enable_visualization': True,
        }
        
        # Update config with provided values
        if config is not None:
            self.config.update(config)
            
        # Initialize vehicles
        ego_init = VehicleState(
            self.config['ego_init_x'],
            self.config['ego_init_y'],
            self.config['ego_init_vx'],
            self.config['ego_init_vy']
        )
        human_init = VehicleState(
            self.config['human_init_x'],
            self.config['human_init_y'],
            self.config['human_init_vx'],
            self.config['human_init_vy']
        )
        
        self.ego = EgoVehicle(ego_init, self.config['dt'])
        self.human = HumanVehicle(human_init, self.config['dt'])
        
        # Initialize belief model
        self.belief = HumanBeliefModel(n_particles=100)
        
        # Initialize MPPI controller
        mppi_config = {
            'dt': self.config['dt'],
            'horizon': 20,
            'n_samples': 200,
            'temperature': 1.0,
            'control_noise_std': 0.5,
            'process_noise_std': 0.1,
            'cvar_alpha': 0.9,
            'cvar_samples': 10,
            'cvar_weight': 5.0,
            'cvar_threshold': 10.0,
            'use_covariance_steering': True,
            'target_cov_scale': 0.5,
        }
        self.controller = CVaRMPPI(self.ego, self.human, self.belief, mppi_config)
        
        # For storing trajectory data
        self.ego_trajectory = []
        self.human_trajectory = []
        self.belief_history = []
        self.time_history = []
    
    def run(self) -> None:
        """Run the full simulation."""
        t = 0.0
        steps = int(self.config['simulation_time'] / self.config['dt'])
        
        print("Starting simulation...")
        start_time = time.time()
        
        for i in range(steps):
            # Store current state
            self.ego_trajectory.append([
                self.ego.state.x, self.ego.state.y, 
                self.ego.state.vx, self.ego.state.vy
            ])
            self.human_trajectory.append([
                self.human.state.x, self.human.state.y,
                self.human.state.vx, self.human.state.vy
            ])
            self.time_history.append(t)
            
            # Store current belief
            self.belief_history.append(self.belief.get_mean_params())
            
            # Step 1: Get current states
            ego_state = self.ego.state
            human_state = self.human.state
            
            # Step 2: Plan optimal control using MPPI
            control = self.controller.plan(ego_state, human_state)
            
            # Step 3: Apply control to ego vehicle
            self.ego.update(control)
            
            # Step 4: Simulate human response
            human_action = self.human.compute_control(
                ego_state, 
                self.config['true_human_params'],
                noise_std=0.1
            )
            self.human.update(ego_state, self.config['true_human_params'], noise_std=0.1)
            
            # Step 5: Update belief about human parameters
            self.controller.update_belief(ego_state, human_state, human_action)
            
            # Increment time
            t += self.config['dt']
            
            # Print progress
            if i % 10 == 0:
                print(f"Simulation progress: {i/steps*100:.1f}% (t={t:.1f}s)")
                
        # Compute simulation time
        elapsed = time.time() - start_time
        print(f"Simulation completed in {elapsed:.2f} seconds.")
        
        # Visualize results
        if self.config['enable_visualization']:
            self.visualize_results()
    
    def visualize_results(self) -> None:
        """Visualize simulation results."""
        # Convert trajectory lists to numpy arrays
        ego_traj = np.array(self.ego_trajectory)
        human_traj = np.array(self.human_trajectory)
        time_hist = np.array(self.time_history)
        
        # Plot trajectories
        plt.figure(figsize=(12, 10))
        
        # Position plot
        plt.subplot(3, 1, 1)
        plt.plot(ego_traj[:, 0], ego_traj[:, 1], 'b-', label='Ego')
        plt.plot(human_traj[:, 0], human_traj[:, 1], 'r-', label='Human')
        plt.plot(ego_traj[0, 0], ego_traj[0, 1], 'bo', label='Ego Start')
        plt.plot(human_traj[0, 0], human_traj[0, 1], 'ro', label='Human Start')
        plt.plot(ego_traj[-1, 0], ego_traj[-1, 1], 'bx', label='Ego End')
        plt.plot(human_traj[-1, 0], human_traj[-1, 1], 'rx', label='Human End')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)  # Left lane
        plt.axhline(y=3.5, color='k', linestyle='--', alpha=0.3)  # Right lane
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Vehicle Trajectories')
        plt.legend()
        plt.grid(True)
        
        # Velocity plot
        plt.subplot(3, 1, 2)
        plt.plot(time_hist, ego_traj[:, 2], 'b-', label='Ego Vx')
        plt.plot(time_hist, human_traj[:, 2], 'r-', label='Human Vx')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Longitudinal Velocity')
        plt.legend()
        plt.grid(True)
        
        # Belief evolution
        plt.subplot(3, 1, 3)
        belief_data = np.array([(b['aggressiveness'], b['desired_gap'], b['desired_speed']) 
                                for b in self.belief_history])
        plt.plot(time_hist, belief_data[:, 0], 'g-', label='Agg. Estimate')
        plt.axhline(y=self.config['true_human_params']['aggressiveness'], 
                    color='g', linestyle='--', alpha=0.5, label='True Agg.')
        plt.plot(time_hist, belief_data[:, 1]/10, 'b-', label='Gap Est./10')
        plt.axhline(y=self.config['true_human_params']['desired_gap']/10, 
                    color='b', linestyle='--', alpha=0.5, label='True Gap/10')
        plt.plot(time_hist, belief_data[:, 2]/10, 'r-', label='Speed Est./10')
        plt.axhline(y=self.config['true_human_params']['desired_speed']/10, 
                    color='r', linestyle='--', alpha=0.5, label='True Speed/10')
        plt.xlabel('Time (s)')
        plt.ylabel('Parameter Value')
        plt.title('Human Parameter Belief Evolution')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('lane_merge_results.png')
        plt.show()


def main():
    """Run a complete lane merging simulation."""
    # Configuration
    config = {
        'dt': 0.1,
        'simulation_time': 15.0,
        'ego_init_x': 0.0,
        'ego_init_y': 3.5,  # Right lane
        'ego_init_vx': 15.0,
        'ego_init_vy': 0.0,
        'human_init_x': 20.0,
        'human_init_y': 0.0,  # Left lane
        'human_init_vx': 18.0,
        'human_init_vy': 0.0,
        'true_human_params': {
            'aggressiveness': 1.2,
            'desired_gap': 20.0,
            'desired_speed': 20.0
        },
        'enable_visualization': True,
    }
    
    # Initialize and run simulation
    sim = LaneMergeSimulation(config)
    sim.run()


if __name__ == "__main__":
    main()