#!/usr/bin/env python
from time import time

import numpy as np

from crowd_sim.envs.policy.policy import Policy
from crowd_nav.policy.vecMPC.logger import BaseLogger
from crowd_sim.envs.utils.action import ActionXY, ActionRot

from .predictors import *
from .predictors.sgan_mppi import SGAN_MPPI
from .predictors.cv_mppi import CV_MPPI

import pytorch_mppi as mppi
import torch


class vecMPPI(Policy):

    def __init__(self, config):
        super().__init__()

        #self.model_predictor = eval(config.MPC.mpc["model_predictor"])()
        self.model_predictor = SGAN_MPPI()
        self.logger = BaseLogger(config.MPC.save_path, config.MPC.model, config.MPC.exp_name)
        self.sim_heading = None
        self.trajectory = None
        self.model_predictor.vpref = config.robot.v_pref
        self.model_predictor.dt = config.env.time_step
        self.model_predictor.set_params(config.MPC.mpc['params'])

        self.kinematics = 'differential'
        self.query_env = False
        self.rotation_constraint = np.pi / 3

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.radius = 0.3

        self.name = 'vecmppi'
        self.u_prev = None

        self.multiagent_training = config.MPC.multiagent_training

        self.mppi_settings = {
            'noise': 2.0,
            'samples': 100,
            'horizon': 6
        }

    def set_mppi_params(self, noise, samples, horizon):
        self.mppi_settings['noise'] = noise
        self.mppi_settings['samples'] = samples
        self.mppi_settings['horizon'] = horizon
        self.model_predictor.prediction_length = horizon + 1
        self.model_predictor.rollout_steps = horizon
        self.model_predictor.prediction_horizon = horizon * self.model_predictor.dt

    def dynamics(self, state, action, t=None):
        print("STATE SHAPE: ", state.shape)
        s = state[:,:2] + action * self.model_predictor.dt
        print("RETURN SHAPE: ", s.shape)
        return state[:,:2] + action * self.model_predictor.dt
    
    def normalize_angle(self, theta):
        """Normalize an angle or batch of angles to [-pi, pi]"""
        return torch.atan2(torch.sin(theta), torch.cos(theta))
    
    def dd_dynamics(self, s, a, t=None):
        print("DD !", s.shape)
        dt = self.model_predictor.dt  # length of transition duration in seconds

        s2_ego = torch.zeros((s.size()[0], 3))
        d_theta = a[:, 1] * dt
        turning_radius = a[:, 0] / a[:, 1]

        s2_ego[:, 0] = torch.where(
            a[:, 1] == 0, a[:, 0] * dt, turning_radius * torch.sin(d_theta)
        )
        s2_ego[:, 1] = torch.where(
            a[:, 1] == 0, 0.0, turning_radius * (1.0 - torch.cos(d_theta))
        )
        s2_ego[:, 2] = torch.where(a[:, 1] == 0, 0.0, d_theta)

        s2_global = torch.zeros_like(s)
        s2_global[:, 0] = (
            s[:, 0] + s2_ego[:, 0] * torch.cos(s[:, 2]) - s2_ego[:, 1] * torch.sin(s[:, 2])
        )
        s2_global[:, 1] = (
            s[:, 1] + s2_ego[:, 0] * torch.sin(s[:, 2]) + s2_ego[:, 1] * torch.cos(s[:, 2])
        )
        s2_global[:, 2] = self.normalize_angle(s[:, 2] + s2_ego[:, 2])

        return s2_global
    
    def cost(self, state, action, t):
        cost_goal = self.model_predictor.goal_cost(state, action, self.goal)
        cost_obstacle = self.model_predictor.obstacle_cost(state, action, self.predictions, t).squeeze()
        cost = cost_goal + cost_obstacle
        print("C SIZE: ", cost.shape)
        return cost
    
    def create_mppi(self):
        if self.u_prev is not None:
            # ctrl = mppi.MPPI(self.dd_dynamics, running_cost=self.cost, nx=5, noise_sigma= self.mppi_settings['noise'] * torch.eye(2), num_samples=self.mppi_settings['samples'], horizon=self.mppi_settings['horizon'],
            #         lambda_=1, device=self.device,
            #         u_min=torch.tensor([-1 * self.model_predictor.vpref, -1 * self.model_predictor.vpref], dtype=torch.double, device=self.device),
            #         u_max=torch.tensor([self.model_predictor.vpref, self.model_predictor.vpref], dtype=torch.double, device=self.device),
            #         U_init=self.u_prev,
            #         step_dependent_dynamics=True)
            ctrl = mppi.MPPI(self.dd_dynamics, running_cost=self.cost, nx=5, noise_sigma= self.mppi_settings['noise'] * torch.eye(2), num_samples=self.mppi_settings['samples'], horizon=self.mppi_settings['horizon'],
                    lambda_=1, device=self.device,
                    u_min=torch.tensor([-1 * 0.3, -1], dtype=torch.double, device=self.device),
                    u_max=torch.tensor([0.3, 0.3], dtype=torch.double, device=self.device),
                    U_init=self.u_prev,
                    step_dependent_dynamics=True)
        else:
            # ctrl = mppi.MPPI(self.dd_dynamics, running_cost=self.cost, nx=5, noise_sigma= self.mppi_settings['noise'] * torch.eye(2), num_samples=self.mppi_settings['samples'], horizon=self.mppi_settings['horizon'],
            #         lambda_=1, device=self.device,
            #         u_min=torch.tensor([-1 * self.model_predictor.vpref, -1 * self.model_predictor.vpref], dtype=torch.double, device=self.device),
            #         u_max=torch.tensor([self.model_predictor.vpref, self.model_predictor.vpref], dtype=torch.double, device=self.device),
            #         step_dependent_dynamics=True)
            ctrl = mppi.MPPI(self.dd_dynamics, running_cost=self.cost, nx=5, noise_sigma= self.mppi_settings['noise'] * torch.eye(2), num_samples=self.mppi_settings['samples'], horizon=self.mppi_settings['horizon'],
                    lambda_=1, device=self.device,
                    u_min=torch.tensor([-1 * 0.3, -1 * 0.3], dtype=torch.double, device=self.device),
                    u_max=torch.tensor([0.3, 0.3], dtype=torch.double, device=self.device),
                    step_dependent_dynamics=True)
        return ctrl
    
    def update_predictions(self):
        self.predictions = self.model_predictor.get_predictions(self.trajectory)

    def update_goal(self, goal):
        self.goal = goal

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
        print("PATHBOT STATE: ", self.pathbot_state.theta)
        self.ego_state = state.robot_state

        # Initialize sim heading for MPC if first iteration
        if self.sim_heading is None:
            self.sim_heading = 0.0
        #else:
            # Set the joint state sim_heading for rollouts
            #self.pathbot_state.set_sim_heading(self.sim_heading)
            #self.sim_heading = 0.0

        array_state = self.update_trajectory(state)
        self.logger.update_trajectory(self.trajectory, 0, 0, time())

        if self.reach_destination(state):
            return self.action_post_processing([0,0], start_t)

        goal = torch.Tensor([self.pathbot_state.gx, self.pathbot_state.gy])
        self.update_predictions()
        self.update_goal(goal)
        
        ctrl = self.create_mppi()
        
        mppi_action = None
        if self.u_prev == None:
            for i in range(5):
                mppi_action = ctrl.command(np.array([self.pathbot_state.px, self.pathbot_state.py, self.pathbot_state.theta]))
        else:
            mppi_action = ctrl.command(np.array([self.pathbot_state.px, self.pathbot_state.py, self.pathbot_state.theta]))
            self.u_prev = mppi_action

        return self.action_post_processing(mppi_action, start_t, rot=True)
        
    def action_post_processing(self, action, start_t, rot=True):
        if rot:
            print("ROTTTTTTTTTTTTTTTTTTTTTTT")
            return ActionRot(action[0], action[1])
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
        self.logger.add_time(time()-start_t)
        return action_xy