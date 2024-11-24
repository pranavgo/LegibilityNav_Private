import os
import logging
import copy
import torch
from tqdm import tqdm
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.action import ActionRot, ActionXY
import numpy as np


class Explorer(object):
    def __init__(self, env, robot, device, writer, memory=None, gamma=None, scenarios=None, goals=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.writer = writer
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.statistics = None
        self.scenarios = scenarios
        self.goals = goals

    def compute_path_irregularity(self, action, direct_action):
        a = np.arctan2(action.vy, action.vx)
        da = np.arctan2(direct_action.vy, direct_action.vx)
        return np.abs(a - da)

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None, epoch=None,
                       print_failure=False, baseline=None):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        min_distances = []
        min_distances_overall = []
        timesteps = []
        success = 0
        collision = 0
        timeout = 0
        discomfort = 0
        min_dist = []
        cumulative_rewards = []
        average_returns = []
        collision_cases = []
        timeout_cases = []
        average_accelerations = []
        average_path_irregularity = []
        average_human_path_irregularity = []
        average_human_accelerations = []

        if k != 1:
            pbar = tqdm(total=k)
        else:
            pbar = None

        for i in range(k):
            if self.scenarios is not None:
                ob = self.env.reset(phase, self.scenarios[i], self.goals[i])
            else:
                ob = self.env.reset(phase)
            self.env.scenario_num = self.env.scenario_num + 1
            done = False
            states = []
            actions = []
            rewards = []
            pis = []
            human_num = len(self.env.humans)
            humanpis = {key: [] for key in range(human_num)}
            human_path_length = {key: 0.0 for key in range(human_num)}
            path_length = 0.0
            while not done:
                hstates = {}
                prevhstates = {}
                for t,human in enumerate(self.env.humans):
                    prevhstates[t] = human.get_full_state()

                rstate = self.robot.get_full_state()
                prev_pos = [rstate.px, rstate.py]
                action = self.robot.act(ob, self.env.orca_border, baseline=baseline)
                rstate = self.robot.get_full_state()
                ob, reward, done, info = self.env.step(action)

                for t,human in enumerate(self.env.humans):
                    hstates[t] = human.get_full_state()
                    humandiff = 2*np.sqrt((hstates[t].gx - hstates[t].px)**2 + (hstates[t].gy - hstates[t].py)**2)
                    if hstates[t].vx == 0 and hstates[t].vy == 0:
                        humanpis[t].append(0.0)
                    else:
                        humanpis[t].append(self.compute_path_irregularity(ActionXY(hstates[t].vx,hstates[t].vy), ActionXY((hstates[t].gx - hstates[t].px) / humandiff, (hstates[t].gy - hstates[t].py) / humandiff)))
                    # print('real action ...................')
                    # print(ActionXY(hstates[t].vx,hstates[t].vy))
                    # print('direct action ...........')
                    # print(ActionXY((hstates[t].gx - hstates[t].px) / humandiff, (hstates[t].gy - hstates[t].py) / humandiff))
                    human_path_length[t] = human_path_length[t] + np.sqrt((hstates[t].px - prevhstates[t].px)**2 + (hstates[t].py - prevhstates[t].py)**2)

                new_pos = [self.robot.get_full_state().px, self.robot.get_full_state().py]
                path_length = path_length + np.sqrt((new_pos[0] - prev_pos[0])**2 + (new_pos[1] - prev_pos[1])**2)
                states.append(self.robot.policy.last_state)
                actions.append(action)
                rewards.append(reward)


                diff = np.sqrt((rstate.gx - rstate.px)**2 + (rstate.gy - rstate.py)**2)
                pis.append(self.compute_path_irregularity(action, ActionXY((rstate.gx - rstate.px) / diff, (rstate.gy - rstate.py) / diff)))

                if isinstance(info, Discomfort):
                    discomfort += 1
                    min_dist.append(info.min_dist)

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
                timesteps.append(self.env.num_steps)
                min_distances.append(self.env.min_dist_sum)
                min_distances_overall.append(self.env.min_dist_overall)
                average_accelerations.append(sum(self.env.robot_accelerations) / len(self.env.robot_accelerations))
                average_path_irregularity.append(sum(pis) / path_length)
                hpir = []
                hum_acc = []
                for t in range(human_num):
                    hpir.append(sum(humanpis[t])/human_path_length[t])
                    hum_acc.append(sum(self.env.human_accelerations[t])/len(self.env.human_accelerations[t]))
                average_human_path_irregularity.append((sum(hpir)/human_num))
                average_human_accelerations.append(sum(hum_acc)/human_num)

            elif isinstance(info, Collision):
                print("ADDING COLLISION")
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
                timesteps.append(self.env.num_steps)
            elif isinstance(info, Timeout):
                print("ADDING TIMEOUT")
                timeout += 1
                print(hstates[0].gy)
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))
            returns = []
            for step in range(len(rewards)):
                step_return = sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                   * reward for t, reward in enumerate(rewards[step:])])
                returns.append(step_return)
            average_returns.append(average(returns))

            if pbar:
                pbar.update(1)
        success_rate = success / k
        collision_rate = collision / k
        timeout_rate = timeout / k

        if success == 0:
            stats = 0.0, 0.0, 0.0, 0.0, 0.0
            exp_stats = 0.0, 0.0 ,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            return stats, exp_stats

        average_acceleration = sum(average_accelerations) / success
        average_human_acceleration = sum(average_human_accelerations) / success
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / success

        avg_min_dist = sum(min_distances) / sum(timesteps)
        avg_min_dist_overall = sum(min_distances_overall) / success
        avg_pi = sum(average_path_irregularity) / success
        avg_hpi = sum(average_human_path_irregularity)/success
        diff_accel = 0.0
        diff_dist = 0.0
        diff_time = 0.0
        diff_dist_overall = 0.0
        diff_pi = 0.0

        for i in range(len(success_times)):
            diff_accel = diff_accel + (average_accelerations[i] - average_acceleration)**2
            diff_dist = diff_dist + ((min_distances[i] / timesteps[i]) - avg_min_dist)**2
            diff_dist_overall = diff_dist_overall + (min_distances_overall[i] - avg_min_dist_overall)**2
            diff_pi = diff_pi + (average_path_irregularity[i] - avg_pi)**2
        avg_accel_std = np.sqrt(diff_accel / success)
        min_dist_std = np.sqrt(diff_dist / success)
        min_dist_overall_std = np.sqrt(diff_dist_overall / success)
        avg_pi_std = np.sqrt(diff_pi / success)
        for i in range(len(success_times)):
            diff_time = diff_time + (success_times[i] - avg_nav_time)**2
        nav_time_std = np.sqrt(diff_time / len(success_times)) if success_times else self.env.time_limit

        success_std = np.sqrt(success_rate * (1 - success_rate))
        collision_std = np.sqrt(collision_rate * (1 - collision_rate))
        timeout_std = np.sqrt(timeout_rate * (1 - timeout_rate))

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        extra_info = extra_info + '' if epoch is None else extra_info + ' in epoch {} '.format(epoch)

        if phase in ['val', 'test']:
            total_time = sum(success_times + collision_times + timeout_times)

        self.statistics = success_rate, collision_rate, avg_nav_time, average(cumulative_rewards), average(average_returns)
        self.exp_stats = success_rate, success_std, collision_rate, collision_std, timeout_rate, timeout_std, avg_nav_time, nav_time_std, avg_min_dist, min_dist_std, avg_min_dist_overall, min_dist_overall_std, avg_hpi, average_human_acceleration

        return self.statistics, self.exp_stats
    
    def calculate_metric_std(self, timesteps, min_distances, success_times):
        k = len(timesteps)


    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')
        
        for i, state in enumerate(states[:-1]):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                next_state = self.target_policy.transform(states[i+1])
                value = sum([pow(self.gamma, (t - i) * self.robot.time_step * self.robot.v_pref) * reward *
                             (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                next_state = states[i+1]
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    value = 0
            value = torch.Tensor([value]).to(self.device)
            reward = torch.Tensor([rewards[i]]).to(self.device)

            if self.target_policy.name == 'ModelPredictiveRL':
                self.memory.push((state[0], state[1], value, reward, next_state[0], next_state[1]))
            else:
                self.memory.push((state, value, reward, next_state))

    def log(self, tag_prefix, global_step):
        sr, cr, time, reward, avg_return = self.statistics
        self.writer.add_scalar(tag_prefix + '/success_rate', sr, global_step)
        self.writer.add_scalar(tag_prefix + '/collision_rate', cr, global_step)
        self.writer.add_scalar(tag_prefix + '/time', time, global_step)
        self.writer.add_scalar(tag_prefix + '/reward', reward, global_step)
        self.writer.add_scalar(tag_prefix + '/avg_return', avg_return, global_step)


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
