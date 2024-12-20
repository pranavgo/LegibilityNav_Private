import numpy as np
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_nav.utils.sm_mpc import MPCLocalPlanner
import copy

class SMLegible(Policy):
    def __init__(self, config):
        super().__init__()
        self.trainable = False
        self.kinematics =  'holonomic'
        self.multiagent_training = True
        self.name = 'sm_legible'
        self.radius = config.robot.radius
        if config.env.obstacle:
            self.static_obstacles = config.env.static_obstacles
        else:
            self.static_obstacles = None


    def configure(self, config):
        self.max_speed = config.sm_legible.max_speed
        self.max_theta = config.sm_legible.max_theta
        self.horizon = config.sm_legible.mpc_horizon
        self.sm_weight = config.sm_legible.sm_weight
        self.dt = config.sm_legible.dt
        self.ob = np.zeros((1, 3))
        self.current_trajectory = []
        self.iter = 0
        # self.static_obstacles = None

       

    def predict(self, state, border=None,radius=None, baseline=None):

        # if there are a different number of obstacles
        self_state = state.robot_state
        self.current_trajectory.append(np.array([self_state.px,self_state.py]))
        self.iter = self.iter + 1
        if len(self.ob) != len(state.human_states):
            self.ob = np.zeros((len(state.human_states), 3))

        # update state of obstacles for the sim_config
        for idx, human_state in enumerate(state.human_states):
            self.ob[idx, :] = [human_state.position[0], human_state.position[1], human_state.radius]
        # Initialize MPCLocalPlanner
        mpc_planner = MPCLocalPlanner(horizon=self.horizon, dt=self.dt, obstacles = self.ob, static_obstacles = self.static_obstacles, max_speed=self.max_speed, robot_radius=self.radius,sim_state=state, sm_weight=self.sm_weight, start=self.current_trajectory[0])

        # Plan a trajectory
        optimal_controls = mpc_planner.plan(np.array([self_state.px,self_state.py]))
        velocity = optimal_controls.reshape(-1, 2)
        action = ActionXY(velocity[0][0],velocity[0][1])

        return action
