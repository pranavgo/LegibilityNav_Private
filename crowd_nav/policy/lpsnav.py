import numpy as np
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_nav.utils.lpsnav_mpc import MPCLocalPlanner
import copy

class LPSnavLegible(Policy):
    def __init__(self, config):
        super().__init__()
        self.trainable = False
        self.kinematics = 'holonomic'
        self.multiagent_training = True
        self.name = 'lpsnav'
        self.other_goals = config.env.other_goals
        self.radius = config.robot.radius
        if config.env.obstacle:
            self.static_obstacles = config.env.static_obstacles
        else:
            self.static_obstacles = None

    def configure(self, config):
        self.max_cost = config.lpsnav_legible.max_cost
        self.max_speed = config.lpsnav_legible.max_speed
        self.receding_horiz = config.lpsnav_legible.receding_horiz
        self.sensing_dist = config.lpsnav_legible.sensing_dist
        self.sensing_horiz = config.lpsnav_legible.sensing_horiz 
        self.prim_horiz = config.lpsnav_legible.prim_horiz 
        self.subgoal_priors = config.lpsnav_legible.subgoal_priors 
        self.long_space = config.lpsnav_legible.long_space 
        self.lat_space = config.lpsnav_legible.lat_space
        self.horizon = config.lpsnav_legible.mpc_horizon
        self.dt = config.lpsnav_legible.dt
        self.ob = np.zeros((1, 3))
        self.iter = 0
        self.current_trajectory = []
        


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
        mpc_planner = MPCLocalPlanner(horizon=self.horizon, dt=self.dt, static_obstacles = self.static_obstacles, max_cost=self.max_cost, receding_horiz=self.receding_horiz, sensing_dist = self.sensing_dist, sensing_horiz=self.sensing_horiz, prim_horiz= self.prim_horiz, subgoal_priors = self.subgoal_priors, long_space= self.long_space, lat_space= self.lat_space, max_speed=self.max_speed, robot_radius=self.radius,sim_state=state, start = self.current_trajectory[0])

        # Plan a trajectory
        optimal_controls = mpc_planner.plan([self_state.px,self_state.py])
        velocity = optimal_controls.reshape(-1, 2)
        action = ActionXY(velocity[0][0],velocity[0][1])

        return action
