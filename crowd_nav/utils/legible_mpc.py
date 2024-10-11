import numpy as np
from scipy.optimize import minimize
import math

class AncaLegible():
    def __init__(self, goal, other_goals, start, current_trajectory, nframes, skipFrames, legible_horizon):
        self.goal = np.array(goal)
        self.start = np.array(start)
        self.other_goals = other_goals
        self.skipFrames = skipFrames
        self.legible_horizon = legible_horizon
        vec_to_goal = self.goal - np.array(self.start)
        self.optimalCost = np.sqrt(np.dot(vec_to_goal, vec_to_goal))
        self.nframes = nframes
        self.currentTraj = current_trajectory
        self.control_pts  = len(current_trajectory)
        for i in range(nframes):
            self.currentTraj.append(np.array([0,0]))

    def LegibleCost(self, t, state, **kwargs):
        # legibility = 0.0
        if t not in self.skipFrames:
            self.currentTraj[self.control_pts + t] = state
            weight = 1 - ((t + self.control_pts)/( self.nframes + self.control_pts))
            
            # prob to goal
            goalProb, current_cost = self.ProbGgivenTraj(t, self.goal)
            # compute Bayesian regularization factor
            regularizer_Z = goalProb
            for i in range(len(self.other_goals)):
                prob, cost = self.ProbGgivenTraj(t, np.array(self.other_goals[i]))
                regularizer_Z = regularizer_Z + prob
            regularizer_Z = 1 / regularizer_Z

            # compute legibility score from
            # https://www.ri.cmu.edu/pub_files/2013/3/legiilitypredictabilityIEEE.pdf
            Prob = regularizer_Z*goalProb
            legibility = -(weight * Prob - 0.1*current_cost)
        return legibility

    def ProbGgivenTraj(self, t, G):
        size = len(self.currentTraj[t + self.control_pts])
        # optimal cost to goal at current position
        Dist_optimal_to_goal = G[:size] - self.currentTraj[t + self.control_pts]
        mag = np.dot(Dist_optimal_to_goal, Dist_optimal_to_goal)  # hack to fix ad problem with 0 length vectors
        Cost_Q_G = 0.0
        if mag != 0.0:
            Cost_Q_G = np.sqrt(mag)

        # current trajectory cost to current point
        Cost_S_Q = 0.0
        if len(self.currentTraj) > 1:
            if len(self.currentTraj) > self.legible_horizon:
                start = len(self.currentTraj) - self.legible_horizon
            else:
                start = 0
            for i in range(t + self.control_pts-1,start,-1):
                Dist_vec = self.currentTraj[i] - self.currentTraj[i-1]
                mag = np.dot(Dist_vec, Dist_vec)  # hack to fix ad problem with 0 length vectors
                if mag != 0.0:
                    Cost_S_Q = Cost_S_Q + np.sqrt(mag)

        # calc prob
        tc = Cost_Q_G + Cost_S_Q
        num = np.power(math.e, -(Cost_Q_G + Cost_S_Q))
        den = np.power(math.e, -self.optimalCost)
        P = num / den
        return P, tc
    
class MPCLocalPlanner(AncaLegible):
    def __init__(self, horizon, dt, obstacles, line_obstacles, goal, other_goals, start, nframes, skipFrames, robot_radius, max_speed, current_trajectory, legible_horizon):
        super().__init__(goal, other_goals, start, current_trajectory, nframes, skipFrames, legible_horizon)
        self.horizon = horizon
        self.dt = dt
        self.obstacles = obstacles
        self.robot_radius = robot_radius
        self.max_speed = max_speed

        self.line_obs = line_obstacles

    def objective(self, u, current_state):
        trajectory = self.simulate_trajectory(current_state, u)
        cost = 0
        for t, state in enumerate(trajectory):
            cost += self.LegibleCost(t, state)
            cost += self.obstacle_cost(state)
        return cost

    def simulate_trajectory(self, initial_state, controls):
        trajectory = [initial_state]
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
            if dist < 1.0:
                # calculate distance to obstacle from robot's current position
                cost =+ 3.5/dist

        # for j, line in enumerate(self.line_obs): # for each line obstacle obstacle
        #     arc_delta_obj = self.point_to_segment_dist(line[0][0], line[0][1], line[1][0], line[1][1], x, y)
        #     # if the obstacle intersects with a point on the arc. i.e. there is a collision on the arc
        #     if arc_delta_obj < 0.01:
        #         # calculate distance to obstacle from robot's current position
        #         cost =+ 0.01*(arc_delta_obj)

        return (cost)


    
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
    #             radius = obstacle[2]
    #             constraints.append(-(dist - radius))
    #     return np.array(constraints)


    def plan(self, current_state):
        n_controls = self.horizon * 2  # 2 control inputs (vx, vy) per timestep
        initial_guess = np.ones(n_controls)
        # constraints = {'type': 'ineq', 'fun': lambda u: self.constraint(u, current_state)}
        result = minimize(
            self.objective,
            initial_guess,
            args=(current_state,),
            method='SLSQP',
            bounds=[(-self.max_speed, self.max_speed)] * n_controls,  # Assuming velocity limits of -1 to 1
            # constraints=constraints
        )

        optimal_controls = result.x
        return optimal_controls

