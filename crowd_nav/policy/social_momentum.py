import numpy as np
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot
from crowd_nav.utils.sm_mpc import MPCLocalPlanner
import copy

class SMLegible(Policy):
    def __init__(self, config):
        super().__init__()
        self.trainable = False
        self.kinematics = 'unicycle'
        self.multiagent_training = True
        self.name = 'sm_legible'
        self.other_goals = config.env.other_goals
        self.radius = config.robot.radius


    def configure(self, config):
        self.max_speed = config.sm_legible.max_speed
        self.max_theta = config.sm_legible.max_theta
        self.horizon = config.sm_legible.mpc_horizon
        self.sm_weight = config.sm_legible.sm_weight
        self.dt = config.sm_legible.dt
        self.ob = np.zeros((1, 3))
        self.line_obs = []
        self.iter = 0
       

    def predict(self, state, border=None,radius=None, baseline=None):

        # if there are a different number of obstacles
        self_state = state.robot_state
        self.iter = self.iter + 1
        if len(self.ob) != len(state.human_states):
            self.ob = np.zeros((len(state.human_states), 3))

        arr1 = np.array([[-2, -5], [-2, 5]])
        arr2 = np.array([[2, -5], [2, 5]])
        arr3 = np.array([[-2, -5], [2, -5]])
        arr4 = np.array([[-2, 5], [2, 5]])


        array_list = [arr1, arr2, arr3, arr4]
        self.line_obs = array_list
        # update state of obstacles for the sim_config
        for idx, human_state in enumerate(state.human_states):
            self.ob[idx, :] = [human_state.position[0], human_state.position[1], human_state.radius]
        # Initialize MPCLocalPlanner
        mpc_planner = MPCLocalPlanner(self.horizon, self.dt, obstacles = self.ob, line_obstacles = self.line_obs, max_speed=self.max_speed, robot_radius=self.radius,sim_state=state, sm_weight=self.sm_weight, max_theta=self.max_theta)

        # Plan a trajectory
        optimal_controls = mpc_planner.plan([self_state.px,self_state.py, self_state.theta])
        velocity = optimal_controls.reshape(-1, 2)
        action = ActionRot(velocity[0][0],velocity[0][1])

        return action
