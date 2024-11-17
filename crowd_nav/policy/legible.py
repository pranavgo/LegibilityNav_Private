import numpy as np
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.utils.legible_mpc import MPCLocalPlanner
import copy

class Legible(Policy):
    def __init__(self, config):
        super().__init__()
        self.trainable = False
        self.kinematics = 'holonomic'
        self.multiagent_training = True
        self.name = 'legible'
        self.other_goals = config.env.other_goals
        self.radius = config.robot.radius


    def configure(self, config):
        self.legible_horizon = config.legible.legible_horizon
        self.max_speed = config.legible.max_speed
        self.horizon = config.legible.mpc_horizon
        self.dt = config.legible.dt
        self.ob = np.zeros((1, 3))
        self.line_obs = []
        self.current_trajectory = []
        self.iter = 0
        self.static_obstacles = [[(-5, 5), (-1, 5), (-1, 1), (-5, 1)],[(5, 5), (1, 5), (1, 1), (5, 1)],[(5, -5), (1, -5), (1, -1), (5, -1)], [(-5, -5), (-1, -5), (-1, -1), (-5, -1)]]



    def predict(self, state, border=None,radius=None, baseline=None):

        # if there are a different number of obstacles
        self_state = state.robot_state
        self.iter = self.iter + 1
        self.current_trajectory.append(np.array([self_state.px,self_state.py]))
        if len(self.ob) != len(state.human_states):
            self.ob = np.zeros((len(state.human_states), 3))

        # update state of obstacles for the sim_config
        for idx, human_state in enumerate(state.human_states):
            self.ob[idx, :] = [human_state.position[0], human_state.position[1], human_state.radius]
        # Initialize MPCLocalPlanner
        mpc_planner = MPCLocalPlanner(self.horizon, self.dt, obstacles = self.ob, static_obstacles = self.static_obstacles, max_speed=self.max_speed, goal = self_state.goal_position, robot_radius=self.radius, other_goals=self.other_goals, start=self.current_trajectory[0], nframes=11, skipFrames=[], current_trajectory = copy.deepcopy(self.current_trajectory), legible_horizon = self.legible_horizon)

        # Plan a trajectory
        optimal_controls = mpc_planner.plan([self_state.px,self_state.py])
        velocity = optimal_controls.reshape(-1, 2)
        action = ActionXY(velocity[0][0],velocity[0][1])

        return action
