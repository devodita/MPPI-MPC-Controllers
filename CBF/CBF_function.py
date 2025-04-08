import numpy as np
from cvxopt import matrix, solvers
import os
import json
import scipy.linalg
from simple_pid import PID
import scipy
import matplotlib.pyplot as plt

solvers.options['show_progress'] = False

config_file = os.environ.get('CONFIG_FILE', 'config.json')
with open(config_file) as file:
    config = json.load(file)
    
class Agent:
    def __init__(self, agent_id, q_i, xi_i, neigh):
        '''
        Use: initialises the agent controller node with relevant variables
        '''
        self.kr = config["kr"]
        self.ka = config["ka"]
        self.h = config["h"]
        self.tau = config["tau"]
        self.epsilon = config["epsilon"]
        self.alpha_c = config["alpha_c"]
        self.alpha_k = config["alpha_k"]
        self.dt = config["dt"]
        self.r_i = config["r_i"]
        self.l = config["l"]
        self.dmin = config["dmin"]
        self.update_frequency = config["alpha_c"]
        self.L_inv = np.linalg.inv([[1, 0],
                                    [0, 1/self.l]])
        self.GAMMA_i = np.array([[25, 0],
                                 [0, 1]]) # technically this is gamma_i ^ 2
        
        # hardcoded obstacle details (turtlebot3_world)
        self.R_k = 0.1 # obst radius
        self.obst_id = [i for i in range(9)]
        #self.obst_id = []
        obst_coord = np.array([[-1.1, -1.1], [-1.1, 0.0], [-1.1, 1.1], 
                               [0.0, -1.1] , [0.0, 0.0] , [0.0, 1.1], 
                               [1.1, -1.1] , [1.1, 0.0] , [1.1, 1.1]])
        self.m_k = {self.obst_id[i]:obst_coord[i] for i in range(len(self.obst_id))}

        # initialise the communication params
        self.agent_id = agent_id
        self.q_i = q_i

        # initialise self and updating variables
        self.xi_i = np.array(xi_i)
        self.p_i = np.zeros(2) # p_x_i p_y_i
        self.u_nom_i = np.zeros(2) # v w (omega)
        self.u_safe_i = np.zeros(2) # v w (omega)
        self.obst_lambda_i = {i:0.1 for i in self.obst_id} # obst_id : lambda_i_obst_id
        self.neigh = neigh
        # PID Control
        self.pid_angle = PID(0.4, 0, 0, setpoint=0)
        self.pid_angle.output_limits = (-0.5, 0.5)
        self.pid_angle.set_auto_mode(enabled=True)

        self.pid_distance = PID(0.2, 0, 0, setpoint=0)
        self.pid_distance.output_limits = (-0.2, 0.2)
        self.pid_distance.set_auto_mode(enabled=True)
        
        self.set_p_i()

    def set_p_i(self):
        s_i = np.array(self.xi_i[0:2])
        e1 = np.array([1,0])
        self.Rot = np.array([[np.cos(self.xi_i[2]), -np.sin(self.xi_i[2])],
                        [np.sin(self.xi_i[2]), np.cos(self.xi_i[2])]])
        self.p_i = np.array(s_i + self.l * self.Rot @ e1)
    
    # input u_i provided through Plant to every agent for update
    def step_i(self, u_i):
        # define the state_dot to be used by the dyna_solver
        def xi_i_dot(t, xi_i):
            x_i_dot = u_i[0] * np.cos(xi_i[2])
            y_i_dot = u_i[0] * np.sin(xi_i[2])
            theta_i_dot = u_i[1]
            return np.array([x_i_dot, y_i_dot, theta_i_dot])
        
        t_eval = np.linspace(0, self.dt, 10) # 10 steps to store, only last used
        sol = scipy.integrate.solve_ivp(xi_i_dot, [0,self.dt], self.xi_i, t_eval=t_eval)

        # update the xi_i and p_i states of the agent
        self.xi_i = np.array(sol.y[:, -1])
        self.set_p_i()
    
    def nominal_controller(self):
        err = np.sqrt((self.p_i[0] - self.q_i[0])**2 + (self.p_i[1] - self.q_i[1])**2 )
        beta = np.arctan2(self.q_i[1] - self.p_i[1], self.q_i[0] - self.p_i[0])
        # """FOR PID"""
        # # updating u_nom_i (nominal controller)
        # if np.abs(beta - self.xi_i[2]) > 0.1:
        #     self.u_nom_i[1] = self.pid_angle(self.xi_i[2]-beta)
        # else:
        #     self.u_nom_i[1]  = 0
        # if err > 0.010:
        #     self.u_nom_i[0]  = self.pid_distance(0.1-err)
        # else:
        #     self.u_nom_i[0]  = 0
        """IN PAPER"""
        print(f"Agent {self.agent_id}, error: {err}, beta: {beta}")
        self.u_nom_i[0] = self.kr * err * np.cos(beta - self.xi_i[2])
        self.u_nom_i[1] = self.ka * (beta - self.xi_i[2]) + self.kr/2*np.sin(2*(beta - self.xi_i[2])) * (beta+(self.h-1)*self.xi_i[2])/(beta - self.xi_i[2])
        
class Plant:
    # N = length(init_vals)
    def __init__(self, N = 1, init_vals = [[0,0,0]], q_i=[[0,0]], neigh=[[]]):
        self.N = N
        self.agents = []
        for i in range(self.N):
            self.agents.append(Agent(agent_id=i, q_i=q_i[i], xi_i=init_vals[i], neigh=neigh[i]))
        print(self.agents)
        
    def safe_controller(self):
        P = np.empty((0, self.agents[0].GAMMA_i.shape[1])) 
        q = np.empty((0,))  
        G = np.empty((0, self.agents[0].GAMMA_i.shape[1]))
        h = np.empty((0,)) 
        for i in range(self.N):
            self.agents[i].nominal_controller()
            P = scipy.linalg.block_diag(P, self.agents[i].GAMMA_i)
            q = np.append(q, -self.agents[i].GAMMA_i @ self.agents[i].u_nom_i)
            g_i = np.zeros((1,2)) 
            h_i = 0
            for j in self.agents[i].neigh:
                dist = self.agents[i].p_i - self.agents[j].p_i
                g_i = g_i - 2 * dist @ (self.agents[i].Rot) @ self.agents[i].L_inv
                h_i = h_i + self.agents[i].alpha_c * (np.dot(dist,dist) - self.agents[i].dmin)
            if len(self.agents[i].obst_id):
                g_i_obst = np.zeros((len(self.agents[i].obst_id),2))
                h_i_obst = np.zeros((len(self.agents[i].obst_id)))
                for j in self.agents[i].obst_id:
                    dist = (self.agents[i].p_i - self.agents[i].m_k[j])
                    g_i_obst[j,:] = -2 * dist @ (self.agents[i].Rot @ self.agents[i].L_inv)
                    dist_cbf = np.dot(dist,dist) - self.agents[i].dmin ** 2
                    h_i_obst[j] = self.agents[i].alpha_k * dist_cbf         
                g_i = np.append(g_i, g_i_obst, axis=0)
                h_i = np.append(h_i, h_i_obst)
            G = scipy.linalg.block_diag(G, g_i)
            h = np.append(h, h_i) 
        G = G[:,2:]
        P = P[:,2:]        
        P = matrix(P, tc='d')
        q = matrix(q, tc='d')
        G = matrix(G, tc='d')  
        h = matrix(h, tc='d')
        
        # Solve the QP problem
        sol = solvers.coneqp(P, q, G, h)
        if sol['status'] != 'optimal':
            print("QP did not find an optimal solution.")
            return None        
        self.u_safe_i = np.array(sol['x']).flatten()  # Ensure u_safe is a 1D array  

def visualise(data):
    flag = True
    dmin = config["dmin"]
    tot_t = config["t"]
    dt = config["dt"]
    no_agents, no_points, _ = data.shape
    for i in range(no_points):
        for j in range(no_agents):
            for k in range(j+1, no_agents):
                dist = np.linalg.norm(data[j,i,:] - data[k,i,:])
                if dist < dmin:
                    flag = False
                    print(f"Collision at time {i/no_points*tot_t}: Agents {j}, {k}")
                    break
    if flag:
        print("Yes, no collision is found")
    else:
        print("Nope, collision is found")
    # Define colors for the 3 sets of points and for the final position
    colors = ['red', 'blue', 'green']
    final_color = 'black'
    
    # Create a plot
    plt.figure(figsize=(8, 8))
    
    # Plot each set of points with circles of fixed radius
    for i in range(no_agents):
        x = data[i, :, 0]
        y = data[i, :, 1]
        # Plot all points except the final one
        for j in range(no_points-1):
            circle = plt.Circle((x[j], y[j]), 0.1, color=colors[i], alpha=0.7)
            plt.gca().add_patch(circle)
        # Plot the final position with a distinctive color
        final_circle = plt.Circle((x[-1], y[-1]), 0.1, color=final_color, alpha=0.9, linewidth=2)
        plt.gca().add_patch(final_circle)
    
    # Set axis limits to accommodate circle radius
    plt.xlim(data[:, :, 0].min() - 0.2, data[:, :, 0].max() + 0.2)
    plt.ylim(data[:, :, 1].min() - 0.2, data[:, :, 1].max() + 0.2)
    
    #Plotting the obstacles
    obst_coord = np.array([[-1.1, -1.1], [-1.1, 0.0], [-1.1, 1.1], 
                               [0.0, -1.1] , [0.0, 0.0] , [0.0, 1.1], 
                               [1.1, -1.1] , [1.1, 0.0] , [1.1, 1.1]])
    
    for (x, y) in obst_coord:
        circle = plt.Circle((x, y), 0.1, color='gray', alpha=0.7)
        plt.gca().add_patch(circle)
    handles = [plt.Line2D([0], [0], color=color, lw=2) for color in colors]
    labels = ['Agent 0', 'Agent 1', 'Agent 2']
    plt.legend(handles, labels, loc='upper right')
    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Visualization of (x, y) circles at different time instances')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    
    """
    Plotting the states
    """
    time_values = np.arange(0, tot_t, dt)  # Generate time values from 0 to tot_t with step dt

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    # Plot x-coordinates over time
    for i in range(no_agents):
        axes[0].plot(time_values, data[i, :, 0], label=f'Agent {i}', color=colors[i])
    axes[0].set_title('X vs t for the Agents')
    axes[0].legend()
    axes[0].grid(True)

    # Plot y-coordinates over time
    for i in range(no_agents):
        axes[1].plot(time_values, data[i, :, 1], label=f'Agent {i}', color=colors[i])
    axes[1].set_title('Y vs t for the Agents')
    axes[1].legend()
    axes[1].grid(True)
    handles = [plt.Line2D([0], [0], color=color, lw=2) for color in colors]
    labels = ['Agent 0', 'Agent 1', 'Agent 2']
    plt.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.show()
            
if __name__ == "__main__":
    n_agents = 3
    init_vals = config["init_vals"]
    q_i = config["q_i"]
    env = Plant(N=n_agents,
                init_vals=init_vals,
                q_i =q_i,
                neigh=[[1,2],[0,2],[0,1]])
    tot_t = config["t"]
    no_points = int(tot_t/env.agents[0].dt)
    x = np.zeros((n_agents,no_points,2))
    for i in range(no_points):
        env.safe_controller()
        # For the Safe Controller
        u_safe = env.u_safe_i
        for j in range(n_agents):
            u_safe_i = u_safe[j*2:(j+1)*2]
            env.agents[j].step_i(u_safe_i)
            x[j,i,:] = env.agents[j].xi_i[:2]
            print(f"Agent ID : {j}, U_NOM:{env.agents[j].u_nom_i}, U_SAFE: {u_safe_i}")
        #  For the Nominal Controller
        # for j in range(n_agents):
        #     u_safe_i = env.agents[j].u_nom_i
        #     env.agents[j].step_i(u_safe_i)
        #     x[j,i,:] = env.agents[j].xi_i[:2]
    visualise(x) 

    # make kr=0.1 for no collision and 0.5 for collision or of course play around with the values!