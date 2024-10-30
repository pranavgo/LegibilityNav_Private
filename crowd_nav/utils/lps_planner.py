import numpy as np
from scipy.optimize import minimize
import math

class LPSLegiblePredictable():
    def __init__(self, state, lps_weight):
        self.state = state
        self.interacting_agents = []
        self.agent_weights = []
        self.lps_weight = lps_weight

    def get_interacting_agents(self):
        robot_state = self.state.robot_state
        for idx, human_state in enumerate(self.state.human_states):
            direction_to_agent = np.arctan2(human_state.py - robot_state.py, human_state.px - robot_state.px)
            distance_to_agent = np.sqrt((human_state.py - robot_state.py) ** 2 + (human_state.px - robot_state.px) ** 2)
            direction_to_agent = np.degrees(direction_to_agent)
            theta_robot = np.degrees(robot_state.theta)
            relative_angle = direction_to_agent - theta_robot
            relative_angle = (relative_angle + 180) % 360 - 180

            if -90 <= relative_angle <= 90 and distance_to_agent < 3.0:
                self.interacting_agents.append(human_state)
                self.agent_weights.append(1 / distance_to_agent)

    def get_lps_cost(self, state, u):
        cost = 100.0
        goal_score = np.sqrt((self.state.robot_state.gx - state[0]) ** 2 + (self.state.robot_state.gy - state[1]) ** 2) / 8.24
        robot_pos = np.array(self.state.robot_state.position)
        if self.interacting_agents:
            for i, agent in enumerate(self.interacting_agents):
                agent_pos = np.array(agent.position)
                r_c = (robot_pos + agent_pos) / 2
                r_ac = robot_pos - r_c
                r_bc = agent_pos - r_c
                l_ab = np.cross(r_ac, np.array(self.state.robot_state.velocity)) + np.cross(r_bc, np.array(agent.velocity))
                pred_pose = agent_pos + np.array(agent.velocity) * self.dt
                r_c_hat = (state[:2] + pred_pose) / 2
                r_ac_hat = state[:2] - r_c_hat
                r_bc_hat = pred_pose - r_c_hat
                l_ab_hat = np.cross(r_ac_hat, np.array([u[0] * np.cos(self.state.robot_state.theta + u[1]), 
                                                       u[0] * np.sin(self.state.robot_state.theta + u[1])])) + np.cross(r_bc_hat, np.array(agent.velocity))

                if np.dot(l_ab, l_ab_hat) > 0:
                    cost += -1 * self.lps_weight * np.abs(l_ab_hat) * self.agent_weights[i]
                else:
                    cost += 10000
            cost += goal_score
        else:
            cost = goal_score
        return cost


class LPSLocalPlanner(LPSLegiblePredictable):
    def __init__(self, horizon, dt, obstacles, line_obstacles, robot_radius, max_speed, sim_state, lps_weight, max_theta):
        super().__init__(sim_state, lps_weight)
        self.horizon = horizon
        self.dt = dt
        self.obstacles = obstacles
        self.robot_radius = robot_radius
        self.max_speed = max_speed
        self.max_theta = max_theta
        self.line_obs = line_obstacles

    def objective(self, u, current_state):
        trajectory = self.simulate_trajectory(current_state, u)
        cost = 0
        for t, state in enumerate(trajectory):
            cost += self.get_lps_cost(state, u)
        return cost

    def simulate_trajectory(self, initial_state, controls):
        trajectory = [initial_state]
        state = initial_state
        for u in controls.reshape(-1, 2):
            next_state = self.simple_dynamics(state, u)
            trajectory.append(next_state)
            state = next_state
        return np.array(trajectory)

    def simple_dynamics(self, state, u):
        x, y, theta = state
        v, r = u
        next_theta = theta + r
        next_x = x + np.cos(next_theta) * v * self.dt
        next_y = y + np.sin(next_theta) * v * self.dt
        return np.array([next_x, next_y, next_theta])

    def plan(self, current_state):
        n_controls = self.horizon * 2  # 2 control inputs (vx, vy) per timestep
        initial_guess = np.ones(n_controls)
        v_min, v_max = -self.max_speed, self.max_speed
        r_min, r_max = -self.max_theta, self.max_theta
        self.get_interacting_agents()
        bounds = [(v_min, v_max), (r_min, r_max)] * self.horizon

        result = minimize(self.objective, initial_guess, args=(current_state,), method='SLSQP', bounds=bounds)
        return result.x
