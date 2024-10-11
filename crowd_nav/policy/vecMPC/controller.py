#!/usr/bin/env python
from time import time

import numpy as np

from crowd_sim.envs.policy.policy import Policy
from crowd_nav.policy.vecMPC.logger import BaseLogger
from crowd_sim.envs.utils.action import ActionXY

from .predictors import *


class vecMPC(Policy):

    def __init__(self, config):
        super().__init__()

        self.model_predictor = eval(config.MPC.mpc["model_predictor"])()
        self.logger = BaseLogger(config.MPC.save_path, config.MPC.model, config.MPC.exp_name)
        self.sim_heading = None
        self.trajectory = None
        self.model_predictor.vpref = config.robot.v_pref
        self.model_predictor.dt = config.env.time_step
        self.action_params = config.MPC.mpc['params']['action']
        self.span = np.deg2rad(self.action_params['span'])
        self.n_actions = self.action_params["n_actions"]
        if self.span == 2 * np.pi:
            self.span = 2 * np.pi - ((2 * np.pi) / self.n_actions)
        self.model_predictor.set_params(config.MPC.mpc['params'])
        self.logger.add_params(config.MPC.mpc['params'])

        self.kinematics = 'holonomic'
        self.speed_samples = 5
        self.rotation_samples = 16
        self.sampling = 'exponential'
        self.query_env = False
        self.rotation_constraint = np.pi / 3
        self.multiagent_training = config.MPC.multiagent_training

        dx_range = config.env.dx_range
        dy_range = config.env.dy_range
        self.border = [(dx_range[0], dy_range[1]), (dx_range[1], dy_range[1]), (dx_range[1], dy_range[0]), (dx_range[0], dy_range[0])]
        self.radius = 0.3

        self.name = 'vecmpc'

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
    
    def reset(self):
        self.trajectory = None
        self.sim_heading = None
        self.logger.reset()
        self.model_predictor.reset()
    
    @staticmethod
    def flatten_state(state):
        px = state.px
        py = state.py
        vx = state.vx
        vy = state.vy
        return np.array([px, py, vx, vy, state.radius])

    def update_trajectory(self, state):
        arr = [self.flatten_state(state.robot_state)]
        for state in state.human_states:
            arr.append(self.flatten_state(state))
        arr = np.stack(arr, axis=0)
        self.trajectory = np.concatenate((self.trajectory, arr[None]), axis=0) if self.trajectory is not None else arr[None]
        return arr

    # noinspection PyAttributeOutsideInit
    def predict(self, state, border=None, radius=None, baseline=None):
        start_t = time()
        self.pathbot_state = state.robot_state
        self.ego_state = state.robot_state

        # Initialize sim heading for MPC if first iteration
        if self.sim_heading is None:
            #self.sim_heading = self.pathbot_state.get_heading()
            self.sim_heading = 0.0
        #else:
            # Set the joint state sim_heading for rollouts
            #self.pathbot_state.set_sim_heading(self.sim_heading)
            #self.sim_heading = 0.0

        array_state = self.update_trajectory(state)
        self.logger.update_trajectory(self.trajectory, 0, 0, time())

        if self.reach_destination(state):
            return self.action_post_processing([0,0], start_t)

        goal = np.array([self.pathbot_state.gx, self.pathbot_state.gy])
        action_set = self.generate_action_set(self.pathbot_state, self.trajectory, goal, v_pref=self.pathbot_state.v_pref)
        # (N x S x T' x H x 2), (N x S), (2 x 1)
        predictions, costs, action_set, best_action = self.model_predictor.predict(self.trajectory, array_state, action_set, goal)

        self.logger.add_predictions(array_state, action_set, predictions, costs, goal)
        return self.action_post_processing(best_action[2:], start_t)
    
    def outside_check(self, position, radius, obstacle):
        left = position[0] - radius < obstacle[0][0]
        right = position[0] + radius > obstacle[1][0]
        below = position[1] - radius < obstacle[2][1]
        above = position[1] + radius > obstacle[1][1]
        if ((left or right) or (above or below)):
            return True

        return False
        
    def action_post_processing(self, action, start_t):
        global_action_xy = ActionXY(action[0], action[1])
        # Keep in global frame ActionXY
        action_xy = global_action_xy

        # Only update heading if we take an action
        if np.linalg.norm(action) > 0:
            self.sim_heading = np.arctan2(action[1], action[0])
        #else:
            # Randomly set heading to get us unstuck if not moving
            #if np.linalg.norm(np.array([self.pathbot_state.vx, self.pathbot_state.vy])) < 0.1:
                #self.sim_heading = np.random.rand() * 2 * np.pi
        self.logger.add_action(action)
        self.logger.add_time(time()-start_t)
        return action_xy

    def generate_action_set(self, state, trajectory, goal, v_pref=None):
        """To get actions"""

        pos = np.array([state.px, state.py])
        vel = [state.vx, state.vy]
        theta = [0.0, 0.0]
        theta_dot = [0.0, 0.0]

        #sim_heading = state.get_sim_heading()
        sim_heading = 0.0
        thetas = [sim_heading-(self.span / 2.0) + i * self.span / (self.n_actions - 1) for i in range(self.n_actions)]
        thetas = thetas if len(thetas) > 1 else np.arctan2(goal-pos)
        if v_pref is None:
            vpref = self.pathbot_state.v_pref
        else:
            vpref = v_pref
        pos_stack = pos[:, None]
        goals = pos_stack + (vpref * self.model_predictor.prediction_horizon *5) * np.stack((np.cos(thetas), np.sin(thetas)), axis=0) # (2, N)
        state = np.array([theta[0], theta_dot[0], vel[0], theta[1], theta_dot[1], vel[1]]) # (6, )

        return self.generate_cv_rollout(pos, state, goals, vpref, self.model_predictor.rollout_steps)

    def generate_cv_rollout(self, position, state, goals, vpref, length):
        """To get particular rollout"""

        rollouts = []
        state = np.repeat(state[:, None], goals.shape[1], axis=1)  # (6, N)
        position = np.repeat(position[:, None], goals.shape[1], axis=1)  # (2, N)
        for _ in range(length):
            ref_velocities = self.generate_cv_action(position, goals, vpref, multiplier=1.0) # (2, N)
            state, position, _ = self.step_dynamics(position, state, ref_velocities)
            rollouts.append( np.concatenate((position, ref_velocities), axis=0).transpose((1, 0)))
        rollouts = np.stack(rollouts, axis=1) # N x T x 4
        return rollouts
    
    def generate_cv_action(self, position, goal, vpref, multiplier=1.0):
        """To get particular action"""
        dxdy = goal-position # (2, N)
        thetas = np.arctan2(dxdy[1], dxdy[0]) # (N, )
        return np.stack((np.cos(thetas), np.sin(thetas)), axis=0) * vpref # (2, N)

    def step_dynamics(self, position, state, action): # (2, N), (6, N), (2, N)
        # Reference input (global)
        N = action.shape[1]
        U = np.concatenate((np.zeros((2, N)), action[0, None], np.zeros((2, N)), action[1, None]), axis=0) # (6, N)

        # Integrate with ballbot dynamics
        next_state = self.integrator(state, U) # (6, N)

        velocity = action
        position = position +  velocity * self.model_predictor.dt # (2, N)
        return next_state, position, velocity

    def integrator(self, S, U):
        M = 4
        dt_ = float(self.model_predictor.dt) / M
        S_next = np.array(S)
        for i in range(M):
            k1 = dt_ * self.state_dot(S, U)
            k2 = dt_ * self.state_dot(S + (0.5 * k1), U)
            k3 = dt_ * self.state_dot(S + (0.5 * k2), U)
            k4 = dt_ * self.state_dot(S + k3, U)
            S_next += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return S_next

    @staticmethod
    def state_dot(S0, U):
        S_dot = np.array(S0)
        S_dot[0] = S0[1]
        S_dot[1] = ((-38.73 * S0[0]) + (-11.84 * S0[1]) + (-6.28 * S0[2]) +
                    (51.61 * U[0]) + (11.84 * U[1]) + (6.28 * U[2]))

        S_dot[2] = ((13.92 * S0[0]) + (2.0 * S0[1]) + (1.06 * S0[2]) +
                    (-8.72 * U[0]) + (-2.0 * U[1]) + (-1.06 * U[2]))

        S_dot[3] = S0[4]
        S_dot[4] = ((-38.54 * S0[3]) + (-11.82 * S0[4]) + (-6.24 * S0[5]) +
                    (51.36 * U[3]) + (11.82 * U[4]) + (6.24 * U[5]))

        S_dot[5] = ((14.00 * S0[3]) + (2.03 * S0[4]) + (1.07 * S0[5]) +
                    (-8.81 * U[3]) + (-2.03 * U[4]) + (-1.07 * U[5]))
        return S_dot
    
    def get_control_type(self):
        return self.__str__() + "_" + self.model_predictor.__str__()