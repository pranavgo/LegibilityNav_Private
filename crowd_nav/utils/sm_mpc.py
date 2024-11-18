import numpy as np
from scipy.optimize import minimize
import math
from shapely.geometry import Point
from shapely.geometry import MultiPolygon, Polygon

class SMLegible():
    def __init__(self, state, sm_weight, start):
        self.state = state
        self.agent_weights = []
        self.sm_weight = sm_weight
        self.max_goal_score = np.linalg.norm(np.array(self.state.robot_state.goal_position) - start)

    def get_interacting_agents(self, state):
        self.interacting_agents = []
        robot_state = self.state.robot_state
        for idx, human_state in enumerate(self.state.human_states):
            direction_to_agent = np.arctan2(human_state.py - state[1], human_state.px - state[0])
            distance_to_agent = np.sqrt((human_state.py - state[1])**2 + (human_state.px - state[0])**2)
            direction_to_agent = np.degrees(direction_to_agent)
            theta_robot = np.degrees(robot_state.theta)
            relative_angle = direction_to_agent - theta_robot
            relative_angle = (relative_angle + 180) % 360 - 180  
            # if np.sum((np.array(human_state.position) - np.array(robot_state.position)) * np.squeeze(np.column_stack((np.cos(robot_state.theta), np.sin(robot_state.theta)))), axis=-1) > 0:   alternate check
            if -90 <= relative_angle <= 90 and distance_to_agent < 4.0:
                self.interacting_agents.append(human_state)
                self.agent_weights.append(1/distance_to_agent)

    def get_sm_cost(self,state,prev_state, u):
        cost = 0.0
        goal_score = np.linalg.norm(np.array(self.state.robot_state.goal_position) - state)/self.max_goal_score
        if self.interacting_agents:
            for i, agent in enumerate(self.interacting_agents):
                agent_pos = np.array(agent.position)
                r_c = (prev_state + agent_pos) / 2
                r_ac = prev_state - r_c
                r_bc = agent_pos - r_c
                l_ab = np.cross(r_ac, np.array(self.state.robot_state.velocity)) + np.cross(r_bc, np.array(agent.velocity))
                pred_pose = agent_pos + np.array(agent.velocity)*self.dt
                r_c_hat = (state + pred_pose) / 2
                r_ac_hat = state - r_c_hat
                r_bc_hat = pred_pose - r_c_hat
                l_ab_hat = np.cross(r_ac_hat, np.array([u[0],u[1]])) + np.cross(r_bc_hat, np.array(agent.velocity))
                if np.dot(l_ab, l_ab_hat) > 0:
                    cost += -1 *self.sm_weight*(np.abs(l_ab_hat))*self.agent_weights[i]
                else:
                    cost += 1000
            cost = cost + (1-self.sm_weight)*(goal_score)
        else:
            cost = goal_score
        return cost
        
    
class MPCLocalPlanner(SMLegible):
    def __init__(self, horizon, dt, obstacles, static_obstacles, robot_radius, max_speed, sim_state, sm_weight, start):
        super().__init__(sim_state, sm_weight, start)
        self.horizon = horizon
        self.dt = dt
        self.obstacles = obstacles
        self.robot_radius = robot_radius
        self.max_speed = max_speed
        self.static_obs = static_obstacles

    def objective(self, u, current_state):
        trajectory = self.simulate_trajectory(current_state, u)
        prev_state = current_state
        cost = 0
        for t, state in enumerate(trajectory):
            cost += self.get_sm_cost(state,prev_state, [u[2*t],u[2*t + 1]])
            if self.static_obs is not None:
                cost += self.obstacle_avoidance_cost(state)
            prev_state = state
            # cost += self.obstacle_cost(state)
        return cost

    def simulate_trajectory(self, initial_state, controls):
        trajectory = []
        state = initial_state
        for u in controls.reshape(-1, 2):
            next_state = self.simple_dynamics(state, u)
            trajectory.append(next_state)
            state = next_state
        trajectory = np.array(trajectory)
        return trajectory

    def simple_dynamics(self, state, u):
        x, y = state
        vx, vy = u
        next_x = x + vx * self.dt
        next_y = y + vy * self.dt
        return np.array([next_x, next_y])


    def obstacle_cost(self, state, t):
        cost = 0.0
        x = state[0]
        y = state[1]
        for idx, human_state in enumerate(self.state.human_states): # for each obstacle
            ox = human_state.px
            oy = human_state.py
            next_x = ox + human_state.vx * self.dt *(t+1)
            next_y = oy + human_state.vy * self.dt *(t+1)
            # if the obstacle intersects with a point on the arc. i.e. there is a collision on the arc
            dist = np.sqrt((x - next_x)**2 + (y - next_y)**2)
            if dist < 1.0:
                # calculate distance to obstacle from robot's current position
                cost =+ 3.5/dist
        return (cost)
    

    def obstacle_avoidance_cost(self, state):
        """
        Defines a set of inequality constraints for obstacle avoidance using shapely.
        
        Args:
        traj_points: List of points in the trajectory [(x1, y1), (x2, y2), ..., (xn, yn)].
        polygons: List of shapely polygons, where each polygon is a shapely Polygon object.
        
        Returns:
        constraints: List of constraints for scipy.minimize, ensuring that trajectory points do not enter any polygon.
        """
        cost = 0
        min_dist = 100000000000000000
        for obs in self.static_obs:
            polygon = Polygon(obs)
            dist = polygon.distance(Point(state))
            if min_dist > dist:
                min_dist =  dist
                if 0 < min_dist < 0.4:
                    cost += 3.5/dist
                elif min_dist == 0:
                    cost += 10000
        return cost

    
    def point_to_segment_dist(self, x1, y1, x2, y2, x3, y3):
        px = x2 - x1
        py = y2 - y1

        if px == 0 and py == 0:
            return np.linalg.norm((x3-x1, y3-y1))

        u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

        if u > 1:
            u = 1
        elif u < 0:
            u = 0

        # (x, y) is the closest point to (x3, y3) on the line segment
        x = x1 + u * px
        y = y1 + u * py

        return np.linalg.norm((x - x3, y-y3))

    def goal_cost(self, state):
        return np.linalg.norm(self.goal - state[:2])
    
    # def constraint(self, u, current_state):
    #     trajectory = self.simulate_trajectory(current_state, u)
    #     constraints = []
    #     for state in trajectory:
    #         for obstacle in self.obstacles:
    #             dist = np.linalg.norm(state[:2] - obstacle[:2])
    #             if dist < 0.5:
    #                 constraints.append((self.robot_radius + obstacle[2]) - dist)
    #     return np.array(constraints)
        
    def plan(self, current_state):
        self.get_interacting_agents(current_state)
        n_controls = self.horizon * 2  # 2 control inputs (vx, vy) per timestep
        initial_guess = 0.01*np.ones(n_controls)
        # constraints = {'type': 'ineq', 'fun': lambda u: self.obstacle_avoidance_constraint(u, current_state)}
        result = minimize(
            self.objective,
            initial_guess,
            args=(current_state,),
            method='SLSQP',
            bounds=[(-self.max_speed, self.max_speed)] * n_controls,  # Assuming velocity limits of -1 to 1
        )

        optimal_controls = result.x
        return optimal_controls

