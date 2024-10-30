import numpy as np
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import ObservableState, JointState

class LPSNavPolicy(Policy):
    def __init__(self, config):
        super().__init__()
        self.name = 'LPSNav'
        self.max_speed = 1
        self.min_speed = 0
        self.max_accel = 3
        self.max_ang_accel = 5
        self.radius = 0.25
        self.goal_tol = 0.1
        self.heading_span = 3.14
        self.speed_samples = 10
        self.heading_samples = 31
        self.prim_horiz = 4
        self.sensing_dist = 20
        self.col_horiz = 4
        self.receding_horiz = 5
        self.sensing_horiz = 10
        self.max_cost = 1e2
        self.subgoal_priors = np.array([0.48, 0.02, 0.5])
        self.beta = 1.0  # rationality parameter
        self.time_step = 0.25
        self.kinematics = 'holonomic'
        self.multiagent_training = True
        self.goal_weight = 0.01

    def set_phase(self, phase):
        self.phase = phase

    def predict(self, state, border=None,radius=None, baseline=None):
        self_state = state.robot_state
        human_states = state.human_states

        if self.has_reached_goal(self_state):
            return ActionRot(0, 0)  # Stop at the goal  

        # Update interaction lines for each human
        self.update_int_lines(self_state, human_states)

        # Compute legibility scores
        leg_scores = self.compute_scores(self_state, human_states)

        # Optimize action based on legibility
        best_action = self.optimize_action(leg_scores, self_state, human_states)

        print(best_action[0]," ",best_action[1])
        return ActionXY(best_action[0], best_action[1])

    def update_int_lines(self, self_state, human_states):
        self.int_lines = {}
        for i, human in enumerate(human_states):
            self.int_lines[i] = self.compute_int_line(self_state, human)

    def compute_int_line(self, self_state, human_state):
        goal_vec = np.array(self_state.goal_position, dtype=float) - np.array(self_state.position, dtype=float)
        perp_vec = np.array([-goal_vec[1], goal_vec[0]], dtype=float)
        perp_vec /= np.linalg.norm(perp_vec)
        human_pos = np.array(human_state.position, dtype=float)
        left_point = human_pos + self.radius * perp_vec
        right_point = human_pos - self.radius * perp_vec
        return left_point, right_point

    def compute_scores(self, self_state, human_states):
        leg_scores = {}
        for i, human in enumerate(human_states):
            leg_scores[i] = self.compute_legibility_scores(self_state, human)
        return leg_scores

    def compute_legibility_scores(self, self_state, human_state):
        # Calculate the goal vector and ensure it's a float array
        goal_vec = np.array(self_state.goal_position, dtype=float) - np.array(self_state.position, dtype=float)
        
        # Create a perpendicular vector and ensure it's a float array
        perp_vec = np.array([-goal_vec[1], goal_vec[0]], dtype=float)
        
        # Normalize the perpendicular vector
        perp_vec /= np.linalg.norm(perp_vec)
        
        # Calculate interaction line endpoints
        human_pos = np.array(human_state.position, dtype=float)
        int_line_left = human_pos + self.radius * perp_vec
        int_line_right = human_pos - self.radius * perp_vec
        
        # Calculate costs for each interaction possibility
        costs = np.zeros(3)  # [left, collision, right]
        costs[0] = np.linalg.norm(int_line_left - np.array(self_state.position)) / self.max_speed
        costs[1] = np.linalg.norm(human_pos - np.array(self_state.position)) / self.max_speed
        costs[2] = np.linalg.norm(int_line_right - np.array(self_state.position)) / self.max_speed

        # Calculate legibility scores using a Boltzmann distribution
        arg = self.max_cost - costs
        arg = np.clip(arg, 0, self.max_cost)
        scores = np.exp(self.beta * arg) * self.subgoal_priors

        dist_to_human = np.linalg.norm(np.array(self_state.position) - human_pos)
    
        # Reduce legibility influence when far from humans
        legibility_factor = np.exp(-dist_to_human / self.sensing_dist)
        scores = legibility_factor * scores + (1 - legibility_factor) * np.array([1/3, 1/3, 1/3])
        
        return scores / np.sum(scores)

        # return np.zeros_like(scores)

    def optimize_action(self, leg_scores, self_state, human_states):
        best_score = -float('inf')
        best_action = None

        goal_dir = np.array(self_state.goal_position, dtype=float) - np.array(self_state.position, dtype=float)
        goal_dist = np.linalg.norm(goal_dir)
        
        if goal_dist > 0:
            goal_dir = goal_dir.astype(float)  # Ensure float type
            goal_dir /= goal_dist

        for speed in np.linspace(self.min_speed, self.max_speed, self.speed_samples):
            for heading in np.linspace(-self.heading_span/2, self.heading_span/2, self.heading_samples):
                action = [speed * np.cos(heading), speed * np.sin(heading)]
                score = self.evaluate_action(action, leg_scores, self_state, human_states)
                
                if goal_dist > 0:
                    goal_alignment = np.dot(goal_dir, action)
                    score += goal_alignment * self.goal_weight
                
                if score > best_score:
                    best_score = score
                    best_action = action

        return best_action

    def evaluate_action(self, action, leg_scores, self_state, human_states):
        total_score = 0
        next_pos = np.array(self_state.position) + np.array(action) * self.time_step

        # print(self_state.goal_position, " ", next_pos)

        goal_score = np.linalg.norm(np.array(self_state.goal_position) - next_pos)
        total_score += goal_score * self.goal_weight  # Add a new parameter self.goal_weight

        for i, human in enumerate(human_states):
            next_pos = np.array(self_state.position) + np.array(action) * self.time_step
            left_point, right_point = self.int_lines[i]
            dist_to_line = point_to_segment_dist(
                np.array(next_pos), 
                left_point, 
                right_point
            )
            
            mid_point = (left_point + right_point) / 2
            if np.dot(next_pos - mid_point, left_point - right_point) > 0:
                score = leg_scores[i][0]  # left
            elif np.dot(next_pos - mid_point, right_point - left_point) > 0:
                score = leg_scores[i][2]  # right
            else:
                score = leg_scores[i][1]  # collision
            
            total_score += score

        return total_score

    def get_action(self, state):
        action = self.predict(state)
        return action
    
    def has_reached_goal(self, self_state):
        dist_to_goal = np.linalg.norm(np.array(self_state.position) - np.array(self_state.goal_position))
        return dist_to_goal < self.goal_tol
    
def point_to_segment_dist(p, a, b):
    """
    Compute the distance from point p to line segment [a, b].
    
    :param p: numpy array of shape (2,), the point
    :param a: numpy array of shape (2,), start point of the line segment
    :param b: numpy array of shape (2,), end point of the line segment
    :return: float, the distance from p to line segment [a, b]
    """
    # Convert inputs to numpy arrays if they aren't already
    p = np.array(p)
    a = np.array(a)
    b = np.array(b)
    
    # vector from a to b
    ab = b - a
    # vector from a to p
    ap = p - a
    
    # If the line segment has zero length, return distance to either endpoint
    if np.all(ab == 0):
        return np.linalg.norm(ap)
    
    # Consider the line extending the segment, parameterized as a + t (b - a)
    # We find projection of point p onto the line. 
    # It falls where t = [(p-a) . (b-a)] / |b-a|^2
    t = np.dot(ap, ab) / np.dot(ab, ab)
    
    if np.all(t < 0.0):
        # Beyond the 'a' end of the segment
        return np.linalg.norm(p - a)
    elif np.all(t > 1.0):
        # Beyond the 'b' end of the segment
        return np.linalg.norm(p - b)
    
    # Projection falls on the segment
    projection = a + t * ab
    return np.linalg.norm(p - projection)