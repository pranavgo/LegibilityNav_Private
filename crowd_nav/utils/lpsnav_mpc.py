import numpy as np
from scipy.optimize import minimize
from crowd_nav.utils import helper
import math
from collections import defaultdict
from shapely.geometry import Point
from shapely.geometry import MultiPolygon, Polygon


class LPSnav():
    def __init__(self, state, start):
        self.state = state
        self.interacting_agents = []
        self.agent_weights = []
        self.max_cost = 1e2
        self.max_speed = 1
        self.sensing_dist = 4
        self.receding_horiz = 2
        self.prim_horiz = 1
        self.sensing_horiz = 2*self.receding_horiz
        self.subgoal_priors = np.array([0.48, 0.01, 0.5])
        self.long_space = [0.1, 0.2]
        self.lat_space = [0.1, 0.2]
        self.cost_rt = self.receding_horiz
        self.cost_tg = {}
        self.pred_pos = {}
        self.cost_rg = {}
        self.cost_rtg = {}
        self.cost_tpg = {}
        self.current_leg_score = {}
        self.pred_int_lines = {}
        self.cost_tp = 1
        self.int_lines = defaultdict(list)
        self.max_goal_score = np.linalg.norm(np.array(self.state.robot_state.goal_position) - start)

    def get_int_costs(self, id, agent, state, next_state, u, t):
        t_hist = np.linspace(0.1, self.receding_horiz, int(self.receding_horiz / 0.1))
        self.pos_hist = np.array(state) - np.array(self.state.robot_state.velocity) * t_hist[:, None]
        receded_line = self.int_lines[id] - np.array(agent.velocity) * self.receding_horiz
        self.cost_rg[id] = helper.dynamic_pt_cost(
            self.pos_hist[-1],
            self.max_speed,
            receded_line,
            self.int_line_heading,
            np.array(agent.velocity),
        )
        self.cost_tg[id] = helper.dynamic_prim_cost(
            np.array(state),
            np.array(next_state),
            self.max_speed,
            np.array([u[0],u[1]]),
            self.pred_int_lines[id],
            self.int_line_heading,
            np.array(agent.velocity),
            self.int_lines[id],
        )
        self.cost_tpg[id] = self.cost_tp + t + self.cost_tg[id]
        if np.any(self.cost_tg[id] == 0):
            partial_cost_tpg = helper.directed_cost_to_line(
                np.array(state), np.array([u[0],u[1]]), self.int_lines[id], np.array(agent.velocity)
            )
            self.cost_tpg[id] = np.where(self.cost_tg[id] == 0, partial_cost_tpg, self.cost_tpg[id])
        self.cost_rtg[id] = self.cost_rt + self.cost_tpg[id]

    def update_int_line(self,state):
        self.int_line_heading = helper.wrap_to_pi(helper.angle(np.array(state) - np.array(self.state.robot_state.goal_position)))
        self.rel_int_lines = {}
        n = len(self.state.human_states)
        for k, a in enumerate(self.state.human_states):
            dtheta = helper.wrap_to_pi(np.arctan2(a.vy,a.vx) - self.int_line_heading - np.pi / 2)
            lat_space = self.lat_space[0] + (self.lat_space[1] - self.lat_space[0]) / n
            long_space = self.long_space[0] + (self.long_space[1] - self.long_space[0]) / n
            long_space = lat_space + min(1, np.sqrt(a.vx**2 + a.vy**2) / self.max_speed) * (long_space - lat_space)
            r = 0.6
            col_width = helper.polar_ellipse(long_space + r, lat_space + r, dtheta)
            # col_width = 0.7
            pts = np.array([[0, -col_width], [0, col_width]])
            self.rel_int_lines[k] = helper.rotate(pts, self.int_line_heading)
            self.int_lines[k] = self.rel_int_lines[k] + np.array(a.position)

    def compute_leg(self, id):
        arg = self.cost_rg[id] - self.cost_rtg[id]
        arg = np.where(self.cost_rtg[id] > self.max_cost, -np.inf, arg)
        arg = np.clip(arg, -self.max_cost, 0)
        self.current_leg_score[id] = np.exp(arg) * self.subgoal_priors
        self.current_leg_score[id] = self.current_leg_score[id] / np.sum(self.current_leg_score[id], axis=0)
        self.current_leg_score[id] = np.delete(self.current_leg_score[id], 1, 0)

    def get_interacting_agents(self,state):
        self.interacting_agents = {}
        robot_state = self.state.robot_state
        self.int_line_heading = helper.wrap_to_pi(helper.angle(np.array(state) - np.array(robot_state.goal_position)))
        for k, a in enumerate(self.state.human_states):
            time_to_interaction = helper.cost_to_line(
                np.array(state), np.sqrt(robot_state.vx**2 + robot_state.vy**2) , self.int_lines[k], a.velocity
            )
            in_radius = helper.dist(np.array(state), np.array(a.position)) < self.sensing_dist
            in_horiz = time_to_interaction < self.sensing_horiz
            # in_front = helper.in_front(np.array(a.position), self.int_line_heading, np.array(state[:2]))
            intersecting = helper.is_intersecting(np.array(state), np.array(robot_state.goal_position), *self.int_lines[k])
            if in_radius and in_horiz and intersecting:
                self.interacting_agents[k] = a

    def predict_pos(self, id, agent, t):
        self.pred_pos[id] = np.array(agent.position) + (np.array(agent.velocity)*(t+1)*self.prim_horiz)
        self.pred_int_lines[id] = self.pred_pos[id] + self.rel_int_lines[id]

    def get_cost(self,prev_state, state, u, t):
        total_cost = 0.0 
        self.update_int_line(prev_state)
        self.get_interacting_agents(prev_state)
            # Calculate costs based on LPSNav logic
        score = np.inf
        for k, a in enumerate(self.state.human_states):
            if k in self.interacting_agents:
                self.predict_pos(k, a, t)
                self.get_int_costs(k, a, prev_state, state, u, t)  # Get interaction costs
                self.compute_leg(k)  # Legibility score
                score = np.minimum(score, np.max(self.current_leg_score[k]))  # Add legibility cost
        if self.interacting_agents:
            total_cost -= score
        else:
            goal_cost = np.linalg.norm(np.array(state) - np.array(self.state.robot_state.goal_position))/self.max_goal_score
            total_cost += goal_cost
        return total_cost

        
    
class MPCLocalPlanner(LPSnav):
    def __init__(self, horizon, dt, obstacles, static_obstacles, robot_radius, max_speed, sim_state, start):
        super().__init__(sim_state, start)
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
            cost += self.get_cost(prev_state, state, np.array([u[2*t],u[2*t + 1]]), t)
            if self.static_obs is not None:
                cost += self.obstacle_avoidance_cost(state, [u[2*t],u[2*t + 1]])
            prev_state = state
            # cost += self.obstacle_cost(state)
        # self.pos_hist = np.roll(self.pos_hist, 1, axis=0)
        # self.pos_hist[0] = np.array(state[:2])
        print(cost)
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
        x, y = state[:2]
        vx, vy = u
        next_x = x + vx * self.dt
        next_y = y + vy * self.dt
        return np.array([next_x, next_y])
    
    def obstacle_cost(self, state):
        cost = 0.0
        x = state[0]
        y = state[1]
        for j in range(self.obstacles.shape[0]): # for each obstacle
            ox = self.obstacles[j, 0]
            oy = self.obstacles[j, 1]
            orad = self.obstacles[j, 2]
            # if the obstacle intersects with a point on the arc. i.e. there is a collision on the arc
            dist = np.sqrt((x - ox)**2 + (y - oy)**2)
            if dist < 2.0:
                # calculate distance to obstacle from robot's current position
                cost =+ 100/dist
        return (cost)
    
    def obstacle_avoidance_cost(self, state, u):
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
    
    def constraint(self, u, current_state):
        trajectory = self.simulate_trajectory(current_state, u)
        constraints = []
        for state in trajectory:
            for obstacle in self.obstacles:
                dist = np.linalg.norm(state[:2] - obstacle[:2])
                if dist < 0.5:
                    constraints.append((self.robot_radius + obstacle[2]) - dist)
        return np.array(constraints)
        
    def plan(self, current_state):
        n_controls = self.horizon * 2  # 2 control inputs (vx, vy) per timestep
        self.prim_horiz = 1
        initial_guess = 0.001*np.ones(n_controls) # Interaction lines
        # constraints = {'type': 'ineq', 'fun': lambda u: self.constraint(u, current_state)}
        result = minimize(
            self.objective,
            initial_guess,
            args=(current_state,),
            method='SLSQP',
           bounds=[(-self.max_speed, self.max_speed)] * n_controls  # Assuming velocity limits of -1 to 1
            # constraints=constraints
        )

        optimal_controls = result.x
        return optimal_controls

