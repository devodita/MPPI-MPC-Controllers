import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation
from typing import List, Tuple
import time

from cvar_mppi_implementation import LaneMergeSimulation, VehicleState


class AnimatedVisualizer:
    """Visualization class for animated lane merging scenario."""
    def __init__(self, simulation: LaneMergeSimulation):
        self.sim = simulation
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Set up the plot
        self.ax.set_xlim(-10, 500)
        self.ax.set_ylim(-5, 8)
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_title('Lane Merging with CVaR-MPPI and Belief Updating')
        
        # Lane markings
        self.ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)  # Left lane center
        self.ax.axhline(y=3.5, color='k', linestyle='-', alpha=0.5)  # Right lane center
        self.ax.axhline(y=-1.75, color='k', linestyle='--', alpha=0.3)  # Bottom edge
        self.ax.axhline(y=1.75, color='k', linestyle='--', alpha=0.3)  # Center divider
        self.ax.axhline(y=5.25, color='k', linestyle='--', alpha=0.3)  # Top edge
        
        # Vehicle visualizations
        self.ego_car = Rectangle((0, 0), 4.5, 1.8, color='blue', alpha=0.7)
        self.human_car = Rectangle((0, 0), 4.5, 1.8, color='red', alpha=0.7)
        self.ax.add_patch(self.ego_car)
        self.ax.add_patch(self.human_car)
        
        # Trajectory tracking
        self.ego_traj_line, = self.ax.plot([], [], 'b-', alpha=0.5)
        self.human_traj_line, = self.ax.plot([], [], 'r-', alpha=0.5)
        
        # Belief visualization
        self.belief_circle = Circle((0, 0), 0.5, color='green', alpha=0.2)
        self.ax.add_patch(self.belief_circle)
        
        # Text info
        self.time_text = self.ax.text(5, 6.5, "", fontsize=10)
        self.belief_text = self.ax.text(5, -4, "", fontsize=10)
        
        # Legend
        self.ax.legend([self.ego_car, self.human_car], 
                      ['Ego Vehicle', 'Human Vehicle'],
                      loc='upper right')
    
    def update_frame(self, i: int) -> Tuple:
        """Update animation frame."""
        # Extract data
        ego_data = np.array(self.sim.ego_trajectory[:i+1])
        human_data = np.array(self.sim.human_trajectory[:i+1])
        
        if i >= len(self.sim.time_history):
            return self.ego_car, self.human_car, self.ego_traj_line, self.human_traj_line, self.time_text, self.belief_text, self.belief_circle
        
        t = self.sim.time_history[i]
        belief = self.sim.belief_history[i]
        
        # Update vehicle positions
        if i < len(self.sim.ego_trajectory):
            ego_x, ego_y = ego_data[i, 0], ego_data[i, 1]
            self.ego_car.set_xy((ego_x - 2.25, ego_y - 0.9))  # Center the car
            
        if i < len(self.sim.human_trajectory):
            human_x, human_y = human_data[i, 0], human_data[i, 1]
            self.human_car.set_xy((human_x - 2.25, human_y - 0.9))  # Center the car
        
        # Update trajectories
        if len(ego_data) > 0:
            self.ego_traj_line.set_data(ego_data[:, 0], ego_data[:, 1])
        if len(human_data) > 0:
            self.human_traj_line.set_data(human_data[:, 0], human_data[:, 1])
        
        # Update belief visualization
        if i < len(self.sim.human_trajectory):
            # Position the belief circle near the human vehicle
            self.belief_circle.center = (human_x, human_y + 2.5)
            # Scale based on belief confidence (could use belief variance)
            agg = belief['aggressiveness']
            self.belief_circle.radius = 0.5 + 0.5 * agg
        
        # Update text
        self.time_text.set_text(f"Time: {t:.1f}s")
        self.belief_text.set_text(f"Belief: Agg={belief['aggressiveness']:.2f}, " +
                                 f"Gap={belief['desired_gap']:.1f}, " +
                                 f"Speed={belief['desired_speed']:.1f}")
        
        return self.ego_car, self.human_car, self.ego_traj_line, self.human_traj_line, self.time_text, self.belief_text, self.belief_circle
    
    def animate(self, save_animation: bool = False) -> None:
        """Run the animation."""
        frames = len(self.sim.time_history)
        ani = FuncAnimation(self.fig, self.update_frame, frames=frames,
                          interval=self.sim.config['dt']*1000, blit=True)
        
        if save_animation:
            try:
                print("Trying to save animation with ffmpeg...")
                ani.save('lane_merge_animation.mp4', fps=10, dpi=200, writer='ffmpeg')
                print("Animation saved successfully using ffmpeg.")
            except ValueError:
                try:
                    print("ffmpeg not available. Trying to save as GIF instead...")
                    ani.save('lane_merge_animation.gif', fps=10, dpi=200, writer='pillow')
                    print("Animation saved as GIF successfully.")
                except Exception as e:
                    print(f"Could not save animation: {str(e)}")
                    print("Displaying animation without saving...")
        
        plt.show()


def run_visualization_demo():
    """Run a demonstration of the lane merging with animation."""
    # Configuration with more aggressive human driver
    config = {
        'dt': 0.1,
        'simulation_time': 15.0,
        'ego_init_x': 0.0,
        'ego_init_y': 3.5,  # Right lane
        'ego_init_vx': 35.0,
        'ego_init_vy': 0.0,
        'human_init_x': 2.0,
        'human_init_y': 0.0,  # Left lane
        'human_init_vx': 35.0,
        'human_init_vy': 0.0,
        'true_human_params': {
            'aggressiveness': 1.5,  # More aggressive human
            'desired_gap': 5.0,    # Smaller gap preference
            'desired_speed': 35.0   # Higher speed preference
        },
        'enable_visualization': True,
    }
    
    print("Running lane merging simulation with CVaR-MPPI...")
    start_time = time.time()
    
    # Initialize and run simulation
    sim = LaneMergeSimulation(config)
    sim.run()
    
    print(f"Simulation completed in {time.time() - start_time:.2f} seconds.")
    print("Starting animation...")
    
    # Create and run the animator
    visualizer = AnimatedVisualizer(sim)
    visualizer.animate(save_animation=True)


if __name__ == "__main__":
    run_visualization_demo()