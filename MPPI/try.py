import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import time


class MPPI():
    """ MMPI according to algorithm 2 in Williams et al., 2017
        'Information Theoretic MPC for Model-Based Reinforcement Learning' """

    def __init__(self, env, K, T, U, lambda_=1.0, noise_mu=0, noise_sigma=1, u_init=1, noise_gaussian=True, downward_start=True):
        self.K = K  # N_SAMPLES
        self.T = T  # TIMESTEPS
        self.lambda_ = lambda_
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.U = U
        self.u_init = u_init
        self.cost_total = np.zeros(shape=(self.K))

        self.env = env
        self.env.reset()
        if downward_start:
            self.env.env.state = [np.pi, 1]
        self.x_init = self.env.env.state

        if noise_gaussian:
            self.noise = np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=(self.K, self.T))
        else:
            self.noise = np.full(shape=(self.K, self.T), fill_value=0.9)
            
        # For visualization
        self.state_history = []
        self.action_history = []
        self.cost_history = []
        self.weight_history = []
        self.predicted_trajectories = []
        
    def _compute_total_cost(self, k):
        self.env.env.state = self.x_init
        states = [self.x_init.copy()]  # Track states for visualization
        
        for t in range(self.T):
            perturbed_action_t = self.U[t] + self.noise[k, t]
            # Updated to handle 5 return values from env.step()
            observation, reward, terminated, truncated, _ = self.env.step([perturbed_action_t])
            self.cost_total[k] += -reward
            states.append(self.env.env.state.copy())
            
        # Store trajectory for best samples
        if k < 5:  # Store a few trajectories for visualization
            self.predicted_trajectories.append(states)

    def _ensure_non_zero(self, cost, beta, factor):
        return np.exp(-factor * (cost - beta))

    def visualize_state_action_history(self):
        """Visualize the pendulum state and actions over time"""
        if not self.state_history:
            print("No state history available for visualization")
            return
            
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Convert states to angle and angular velocity
        angles = [state[0] % (2 * np.pi) for state in self.state_history]
        velocities = [state[1] for state in self.state_history]
        time_steps = list(range(len(angles)))
        
        # Plot angle over time
        ax1.plot(time_steps, angles, 'b-')
        ax1.set_ylabel('Angle (rad)')
        ax1.set_title('Pendulum Angle Over Time')
        ax1.axhline(y=0, color='r', linestyle='--')  # Target angle (upright position)
        
        # Plot angular velocity over time
        ax2.plot(time_steps, velocities, 'g-')
        ax2.set_ylabel('Angular Velocity')
        ax2.set_title('Pendulum Angular Velocity Over Time')
        ax2.axhline(y=0, color='r', linestyle='--')  # Target velocity
        
        # Plot actions over time
        ax3.plot(time_steps[:len(self.action_history)], self.action_history, 'r-')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Control Action')
        ax3.set_title('Control Actions Over Time')
        
        plt.tight_layout()
        plt.savefig('pendulum_state_action_history.png')
        plt.show()
        
    def visualize_cost_distribution(self, iteration):
        """Visualize the cost distribution for the current iteration"""
        plt.figure(figsize=(12, 6))
        
        # Sort costs for better visualization
        sorted_costs = np.sort(self.cost_total)
        
        # Plot distribution
        plt.subplot(1, 2, 1)
        plt.hist(self.cost_total, bins=30, alpha=0.7)
        plt.title(f'Cost Distribution (Iteration {iteration})')
        plt.xlabel('Total Cost')
        plt.ylabel('Frequency')
        
        # Plot sorted costs
        plt.subplot(1, 2, 2)
        plt.plot(sorted_costs)
        plt.title('Sorted Costs')
        plt.xlabel('Sample Index (Sorted)')
        plt.ylabel('Cost')
        
        plt.tight_layout()
        plt.savefig(f'cost_distribution_iter_{iteration}.png')
        plt.close()
    
    def visualize_weights(self, omega, iteration):
        """Visualize the weight distribution for trajectory combination"""
        plt.figure(figsize=(10, 6))
        
        # Sort weights for better visualization
        sorted_indices = np.argsort(self.cost_total)
        sorted_weights = omega[sorted_indices]
        
        plt.bar(range(len(sorted_weights)), sorted_weights, alpha=0.7)
        plt.title(f'Weight Distribution (Iteration {iteration})')
        plt.xlabel('Sample Index (Sorted by Cost)')
        plt.ylabel('Weight')
        plt.yscale('log')  # Log scale to see small weights
        
        plt.tight_layout()
        plt.savefig(f'weight_distribution_iter_{iteration}.png')
        plt.close()
        
    def visualize_control_update(self, iteration):
        """Visualize the control sequence and its update"""
        plt.figure(figsize=(10, 6))
        
        # Plot the control sequence
        plt.plot(range(self.T), self.U, 'b-', label='Current Control Sequence')
        
        # Plot the previous control sequence if available
        if hasattr(self, 'previous_U'):
            plt.plot(range(self.T), self.previous_U, 'r--', label='Previous Control Sequence')
        
        plt.title(f'Control Sequence (Iteration {iteration})')
        plt.xlabel('Time Step')
        plt.ylabel('Control Action')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'control_sequence_iter_{iteration}.png')
        plt.close()
        
        # Store the current control sequence for next iteration
        self.previous_U = self.U.copy()
        
    def visualize_predicted_trajectories(self, iteration, num_trajectories=5):
        """Visualize a few predicted trajectories"""
        if not self.predicted_trajectories:
            return
            
        plt.figure(figsize=(12, 6))
        
        # Plot angle trajectories
        plt.subplot(1, 2, 1)
        for i, trajectory in enumerate(self.predicted_trajectories[:num_trajectories]):
            angles = [state[0] % (2 * np.pi) for state in trajectory]
            plt.plot(angles, label=f'Trajectory {i+1}')
            
        plt.axhline(y=0, color='r', linestyle='--', label='Target')
        plt.title('Predicted Angle Trajectories')
        plt.xlabel('Time Step')
        plt.ylabel('Angle (rad)')
        plt.legend()
        
        # Plot velocity trajectories
        plt.subplot(1, 2, 2)
        for i, trajectory in enumerate(self.predicted_trajectories[:num_trajectories]):
            velocities = [state[1] for state in trajectory]
            plt.plot(velocities, label=f'Trajectory {i+1}')
            
        plt.axhline(y=0, color='r', linestyle='--', label='Target')
        plt.title('Predicted Velocity Trajectories')
        plt.xlabel('Time Step')
        plt.ylabel('Angular Velocity')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'predicted_trajectories_iter_{iteration}.png')
        plt.close()

    def control(self, iter=1000, visualize=True, visualize_every=50):
        for iteration in range(iter):
            # Clear the predicted trajectories for this iteration
            self.predicted_trajectories = []
            
            for k in range(self.K):
                self._compute_total_cost(k)

            beta = np.min(self.cost_total)  # minimum cost of all trajectories
            cost_total_non_zero = self._ensure_non_zero(cost=self.cost_total, beta=beta, factor=1/self.lambda_)

            eta = np.sum(cost_total_non_zero)
            omega = 1/eta * cost_total_non_zero
            
            # Store weights for visualization
            self.weight_history.append(omega.copy())

            # Store the original control sequence for visualization
            if hasattr(self, 'previous_U'):
                self.previous_U = self.U.copy()
            else:
                self.previous_U = self.U.copy()

            # Properly update U using numpy operations
            for t in range(self.T):
                self.U[t] += np.sum(omega * self.noise[:, t])

            self.env.env.state = self.x_init
            # Updated to handle 5 return values from env.step()
            s, r, terminated, truncated, _ = self.env.step([self.U[0]])
            
            # Store data for visualization
            self.state_history.append(self.x_init.copy())
            self.action_history.append(self.U[0])
            self.cost_history.append(-r)
            
            print(f"Iteration {iteration}: action taken: {self.U[0]:.2f} cost received: {-r:.2f}")
            
            # Visualize every few iterations to avoid too many plots
            if visualize and iteration % visualize_every == 0:
                self.visualize_cost_distribution(iteration)
                self.visualize_weights(omega, iteration)
                self.visualize_control_update(iteration)
                #self.visualize_predicted_trajectories(iteration)
            
            self.env.render()

            self.U = np.roll(self.U, -1)  # shift all elements to the left
            self.U[-1] = self.u_init  #
            self.cost_total[:] = 0
            self.x_init = self.env.env.state
            
        # Final visualization at the end
        if visualize:
            self.visualize_state_action_history()
            print("Visualization completed. Check the current directory for saved plots.")


def animate_pendulum(state_history):
    """Create an animation of the pendulum swinging"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    
    line, = ax.plot([], [], 'o-', lw=2)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    def animate(i):
        theta = state_history[i][0]
        x = np.sin(theta)
        y = -np.cos(theta)
        line.set_data([0, x], [0, y])
        time_text.set_text(f'Time step: {i}')
        return line, time_text
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(state_history),
                          interval=50, blit=True)
    
    # Save animation
    #anim.save('pendulum_animation.gif', writer='pillow', fps=20)
    plt.close()
    
    return anim


if __name__ == "__main__":
    ENV_NAME = "Pendulum-v1"
    TIMESTEPS = 20  # T
    N_SAMPLES = 1000  # K
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0

    noise_mu = 0
    noise_sigma = 10
    lambda_ = 1

    U = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH, size=TIMESTEPS)  # pendulum joint effort in (-2, +2)

    env = gym.make(ENV_NAME, render_mode="human")  # Set render_mode for visualization
    mppi_gym = MPPI(env=env, K=N_SAMPLES, T=TIMESTEPS, U=U, lambda_=lambda_, noise_mu=noise_mu, noise_sigma=noise_sigma, u_init=0, noise_gaussian=True)
    
    # Run with visualization
    mppi_gym.control(iter=200, visualize=True, visualize_every=10)
    
    # Create animation of the pendulum
    animate_pendulum(mppi_gym.state_history)