import logging
import argparse
import importlib.util
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils import utils
from crowd_sim.envs.policy.orca import ORCA
import random
import time
from numpy.linalg import norm
import json

def main_experiments(args):

    # configure logging and device
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    if args.model_dir is not None:
        if args.config is not None:
            config_file = args.config
        else:
            config_file = os.path.join(args.model_dir, 'config.py')
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
            logging.info('Loaded IL weights')
        elif args.rl:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                model_weights = os.path.join(args.model_dir, sorted(os.listdir(args.model_dir))[-1])
            logging.info('Loaded RL weights')
        else:
            model_weights = os.path.join(args.model_dir, 'best_val.pth')
            logging.info('Loaded RL weights with best VAL')

    else:
        config_file = args.config

    spec = importlib.util.spec_from_file_location('config', config_file)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    ec = config.ExperimentsConfig(args.debug)
    scenarios, goals = utils.random_sequence(ec)

    baseline = None

    # configure policy
    policy_config = config.PolicyConfig(args.debug)
    if args.policy == 'vecmpc' or args.policy == 'vecmppi' or args.policy == 'legible' or args.policy == 'sm_legible':
        env_config = config.EnvConfig(args.debug)
        policy = policy_factory[args.policy](env_config)
    elif args.policy == 'orca' or args.policy == 'sfm' or args.policy == 'cv' or args.policy == 'reactive':
        policy = policy_factory[policy_config.name]()
        baseline = args.policy
    else:
        policy = policy_factory[policy_config.name]()
        baseline = args.policy

    if args.planning_depth is not None:
        policy_config.model_predictive_rl.do_action_clip = True
        policy_config.model_predictive_rl.planning_depth = args.planning_depth
    if args.planning_width is not None:
        policy_config.model_predictive_rl.do_action_clip = True
        policy_config.model_predictive_rl.planning_width = args.planning_width
    if args.sparse_search:
        policy_config.model_predictive_rl.sparse_search = True

    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.load_model(model_weights)

    if args.visualize:
        # configure environment
        env_config = config.EnvConfig(args.debug)
        if args.human_num is not None:
            env_config.sim.human_num = args.human_num
        env = gym.make('CrowdSim-v0')
        e = 1
        se = 1

        env_config.env.dx_range = ec.exp.dx[e][se]
        env_config.env.dy_range = ec.exp.dy[e][se]
        env_config.sim.test_scenario = ec.exp.scenarios[e][se]
        env.configure(env_config)

        robot = Robot(env_config, 'robot')
        env.set_robot(robot)
        robot.time_step = env.time_step
        robot.set_policy(policy)
        explorer = Explorer(env, robot, device, None, gamma=0.9)

        train_config = config.TrainConfig(args.debug)
        epsilon_end = train_config.train.epsilon_end
        # if not isinstance(robot.policy, ORCA):
        #     robot.policy.set_epsilon(epsilon_end)

        policy.set_phase(args.phase)
        policy.set_device(device)
        # set safety space for ORCA in non-cooperative simulation
        if isinstance(robot.policy, ORCA):
            if robot.visible:
                robot.policy.safety_space = args.safety_space
            else:
                robot.policy.safety_space = args.safety_space
            logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

        policy.set_env(env)
        robot.print_info()

        rewards = []
        time_start = time.time()
        ob = env.reset(args.phase, scenarios[e][se][1], goals[e][se][1])
        done = False
        last_pos = np.array(robot.get_position())
        while not done:
            action = robot.act(ob, border=env.orca_border, baseline=baseline)
            ob, _, done, info = env.step(action)
            rewards.append(_)
            current_pos = np.array(robot.get_position())
            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos
        gamma = 0.9
        cumulative_reward = sum([pow(gamma, t * robot.time_step * robot.v_pref)
            * reward for t, reward in enumerate(rewards)])
        
        time_end = time.time()

        if args.traj:
            env.render('traj', args.video_file)
        else:
            if args.video_file is not None:
                if policy_config.name == 'gcn':
                    args.video_file = os.path.join(args.video_dir, policy_config.name + '_' + policy_config.gcn.similarity_function)
                else:
                    args.video_file = os.path.join(args.video_dir, args.video_file)
                #args.video_file = args.video_file + '_' + args.phase + '_' + str(args.test_case) + '.mp4'
            env.render('video', args.video_file)
        logging.info('It takes %.2f seconds to finish. Final status is %s, cumulative_reward is %f', env.global_time, info, cumulative_reward)
        if robot.visible and info == 'reach goal':
            human_times = env.get_human_times()
            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
    else:
        exp_stats_list = []
        if ec.exp.parameter_sweep is not None:
            if ec.exp.parameter_sweep == 'sigma':
                num_p1 = len(ec.exp.sigma)
                num_p2 = len(ec.exp.sigma)
                num_p3 = len(ec.exp.sigma)
            elif ec.exp.parameter_sweep == 'q':
                num_p1 = len(ec.exp.q)
                num_p2 = len(ec.exp.q)
                num_p3 = len(ec.exp.q)
            elif ec.exp.parameter_sweep == 'mppi':
                num_p1 = len(ec.exp.noise)
                num_p2 = len(ec.exp.samples)
                num_p3 = len(ec.exp.horizon)
            for p1 in range(num_p1):
                for p2 in range(num_p2):
                    for p3 in range(num_p3):
                        for e in range(len(ec.exp.dx)):
                            if e != 0:
                                continue
                            for se in range(len(ec.exp.dx[e])):
                                if se != 0:
                                    continue
                                # configure environment
                                env_config = config.EnvConfig(args.debug)
                                env_config.env.dx_range = ec.exp.dx[e][se]
                                env_config.env.dy_range = ec.exp.dy[e][se]
                                env_config.env.randomize_attributes = ec.exp.randomize_attributes[e][se]
                                env_config.humans.num_sf = ec.exp.num_sf[e][se]
                                env_config.humans.num_orca = ec.exp.num_orca[e][se]
                                env_config.humans.num_static = ec.exp.num_static[e][se]
                                env_config.humans.num_linear = ec.exp.num_linear[e][se]
                                env_config.sim.test_scenario = ec.exp.scenarios[e][se]

                                if args.human_num is not None:
                                    env_config.sim.human_num = args.human_num
                                env = gym.make('CrowdSim-v0')
                                env.configure(env_config)

                                robot = Robot(env_config, 'robot')
                                env.set_robot(robot)
                                robot.time_step = env.time_step
                                if ec.exp.parameter_sweep == 'sigma':
                                    param_list = ec.exp.sigma
                                    policy.model_predictor.sigma_h = param_list[p1]
                                    policy.model_predictor.sigma_s = param_list[p2]
                                    policy.model_predictor.sigma_r = param_list[p3]
                                elif ec.exp.parameter_sweep == 'q':
                                    param_list = ec.exp.q
                                    policy.model_predictor.q_obs = param_list[p1]
                                    policy.model_predictor.q_goal = param_list[p2]
                                    policy.model_predictor.q_wind = param_list[p3]
                                elif ec.exp.parameter_sweep == 'mppi':
                                    policy.set_mppi_params(ec.exp.noise[p1], ec.exp.samples[p2], ec.exp.horizon[p3])
                                robot.set_policy(policy)
                                explorer = Explorer(env, robot, device, None, gamma=0.9)

                                train_config = config.TrainConfig(args.debug)
                                epsilon_end = train_config.train.epsilon_end
                                if not isinstance(robot.policy, ORCA):
                                    robot.policy.set_epsilon(epsilon_end)

                                policy.set_phase(args.phase)
                                policy.set_device(device)
                                # set safety space for ORCA in non-cooperative simulation
                                if isinstance(robot.policy, ORCA):
                                    if robot.visible:
                                        robot.policy.safety_space = args.safety_space
                                    else:
                                        robot.policy.safety_space = args.safety_space

                                policy.set_env(env)
                                time_start = time.time()
                                stats, exp_stats = explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True, baseline=baseline)
                                time_end = time.time()
                                exp_stats_list.append(exp_stats)
                                if args.plot_test_scenarios_hist:
                                    test_angle_seeds = np.array(env.test_scene_seeds)
                                    b = [i * 0.01 for i in range(101)]
                                    n, bins, patches = plt.hist(test_angle_seeds, b, facecolor='g')
                                    plt.savefig(os.path.join(args.model_dir, 'test_scene_hist.png'))
                                    plt.close()

                                logging.info("Parameter settings %.2d %.2d %.2d with success rate %.2f average time %.2f minimum distance %.2f and path irregularity %.2f", p1, p2, p3, exp_stats[0], exp_stats[6], exp_stats[12], exp_stats[14])

        else:
            env_config = config.EnvConfig(args.debug)
            for e in range(len(ec.exp.dx)):
                #if e != 0:
                #    continue
                for se in range(len(ec.exp.dx[e])):
                    #if e == 0 and se == 6:
                    #    continue
                    # configure environment
                    env_config.env.dx_range = ec.exp.dx[e][se]
                    env_config.env.dy_range = ec.exp.dy[e][se]
                    env_config.env.randomize_attributes = ec.exp.randomize_attributes[e][se]
                    env_config.humans.num_sf = ec.exp.num_sf[e][se]
                    env_config.humans.num_orca = ec.exp.num_orca[e][se]
                    env_config.humans.num_static = ec.exp.num_static[e][se]
                    env_config.humans.num_linear = ec.exp.num_linear[e][se]
                    env_config.sim.test_scenario = ec.exp.scenarios[e][se]

                    if args.human_num is not None:
                        env_config.sim.human_num = args.human_num
                    env = gym.make('CrowdSim-v0')
                    env.configure(env_config)

                    robot = Robot(env_config, 'robot')
                    env.set_robot(robot)
                    robot.time_step = env.time_step
                    robot.set_policy(policy)
                    if scenarios is not None:
                        explorer = Explorer(env, robot, device, None, gamma=0.9, scenarios=scenarios[e][se], goals=goals[e][se])
                    else:
                        explorer = Explorer(env, robot, device, None, gamma=0.9)

                    train_config = config.TrainConfig(args.debug)
                    epsilon_end = train_config.train.epsilon_end
                    if not isinstance(robot.policy, ORCA):
                        robot.policy.set_epsilon(epsilon_end)

                    policy.set_phase(args.phase)
                    policy.set_device(device)
                    # set safety space for ORCA in non-cooperative simulation
                    if isinstance(robot.policy, ORCA):
                        if robot.visible:
                            robot.policy.safety_space = args.safety_space
                        else:
                            robot.policy.safety_space = args.safety_space
                        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

                    policy.set_env(env)

                    stats, exp_stats = explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True, baseline=baseline)
                    exp_stats_list.append(exp_stats)
                    print(exp_stats_list)
                    if args.plot_test_scenarios_hist:
                        test_angle_seeds = np.array(env.test_scene_seeds)
                        b = [i * 0.01 for i in range(101)]
                        n, bins, patches = plt.hist(test_angle_seeds, b, facecolor='g')
                        plt.savefig(os.path.join(args.model_dir, 'test_scene_hist.png'))
                        plt.close()

            with open(args.results_file, "w") as results:
                json.dump(exp_stats_list, results)

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--policy', type=str, default=None)
    parser.add_argument('-m', '--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--rl', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('-v', '--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('-c', '--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--scenario', type=str, default='circle')
    parser.add_argument('--video_file', type=str, default='legible.mp4')
    parser.add_argument('--video_dir', type=str, default='./')
    parser.add_argument('--results_file', type=str, default='results.json')
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--human_num', type=int, default=None)
    parser.add_argument('--safety_space', type=float, default=0.2)
    parser.add_argument('--test_scenario', type=str, default=None)
    parser.add_argument('--plot_test_scenarios_hist', default=True, action='store_true')
    parser.add_argument('-d', '--planning_depth', type=int, default=None)
    parser.add_argument('-w', '--planning_width', type=int, default=None)
    parser.add_argument('--sparse_search', default=False, action='store_true')

    sys_args = parser.parse_args()

    main_experiments(sys_args)
