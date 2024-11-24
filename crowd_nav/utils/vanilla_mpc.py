import numpy as np
from scipy.optimize import minimize
import math
from shapely.geometry import Point
from shapely.geometry import MultiPolygon, Polygon


    
class MPCLocalPlanner():
    def __init__(self, horizon, dt, static_obstacles, robot_radius, max_speed, goal, start, sim_state):
        self.state = sim_state
        self.horizon = horizon
        self.dt = dt
        self.robot_radius = robot_radius
        self.max_speed = max_speed
        self.static_obs = static_obstacles
        self.goal = np.array(goal)
        self.max_goal_score = np.linalg.norm(np.array(self.goal) - start)

    def objective(self, u, current_state):
        trajectory = self.simulate_trajectory(current_state, u)
        cost = 0
        for t, state in enumerate(trajectory):
            cost += 2*self.goal_cost(state)
            cost += self.obstacle_cost(state, t)
            if self.static_obs is not None:
                cost += self.obstacle_avoidance_cost(state)
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
                cost =+ 0.40/(1 + dist)
        return (cost)
    
    # def collision_avoidance_cost(self,state):
    #     polygons = []
    #     for obs in self.static_obs:
    #         polygons.append(Polygon(obs))
    #     polygon = MultiPolygon(polygons)
    #     Point_state = Point(state)
    #     if polygon.contains(Point_state):
    #         cost = 10
    #     else:
    #         cost =  -1
    #     return cost
    


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
        return np.linalg.norm(self.goal - state)/self.max_goal_score
    
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
        initial_guess = 0.001*np.ones(n_controls)
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

