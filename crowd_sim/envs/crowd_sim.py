import logging
import random
import math

import gym
import matplotlib.lines as mlines
from matplotlib import patches
from matplotlib.collections import PatchCollection
from matplotlib import colors
import numpy as np
from numpy.linalg import norm

from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.state import tensor_to_joint_state, JointState
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
import crowd_sim.envs.utils.utils as utils
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.policy.linear import Linear
from crowd_sim.envs.policy.socialforce import SocialForce


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.robot_sensor_range = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_scenario = None
        self.test_scenario = None
        self.current_scenario = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        self.nonstop_human = None
        self.centralized_planning = None
        self.centralized_planner = None

        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        self.robot_actions = None
        self.rewards = None
        self.As = None
        self.Xs = None
        self.feats = None
        self.trajs = list()
        self.panel_width = 10
        self.panel_height = 10
        self.panel_scale = 1
        self.test_scene_seeds = []
        self.dynamic_human_num = []
        self.human_starts = []
        self.human_goals = []

        self.phase = None

    def sample_uniform(self, range):
        return random.randint(range[0], range[1])

    def set_num_policies(self):
        config = self.config
        if config.humans.num_sf_orca is not None:
            num_orca = self.sample_uniform([0, config.humans.num_sf_orca])
            num_sf = config.humans.num_sf_orca - num_orca
        else:
            if len(config.humans.num_orca) > 1:
                num_orca = self.sample_uniform(config.humans.num_orca)
            else:
                num_orca = config.humans.num_orca[0]
            
            if len(config.humans.num_sf) > 1:
                num_sf = self.sample_uniform(config.humans.num_sf)
            else:
                num_sf = config.humans.num_sf[0]

        if config.humans.num_linear_static is not None:
            num_linear = self.sample_uniform([0, config.humans.num_linear_static])
            num_static = config.humans.num_linear_static - num_linear
        else:
            if len(config.humans.num_linear) > 1:
                num_linear = self.sample_uniform(config.humans.num_linear)
            else:
                num_linear = config.humans.num_linear[0]

            if len(config.humans.num_static) > 1:
                num_static = self.sample_uniform(config.humans.num_static)
            else:
                num_static = config.humans.num_static[0]

        self.num_policies = {
            'orca' : num_orca,
            'socialforce' : num_sf,
            'linear' : num_linear,
            'static' : num_static
        }


    def configure(self, config):
        self.config = config
        self.time_limit = config.env.time_limit
        self.time_step = config.env.time_step
        self.randomize_attributes = config.env.randomize_attributes
        self.robot_sensor_range = config.env.robot_sensor_range
        self.success_reward = config.reward.success_reward
        self.collision_penalty = config.reward.collision_penalty
        self.discomfort_dist = config.reward.discomfort_dist
        self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': config.env.train_size, 'val': config.env.val_size,
                          'test': config.env.test_size}
        self.train_val_scenario = config.sim.train_val_scenario
        self.test_scenario = config.sim.test_scenario
        self.square_width = config.sim.square_width
        self.circle_radius = config.sim.circle_radius
        self.human_num = config.sim.human_num
        self.random_seed = config.sim.random_seed
        self.other_goals = config.env.other_goals

        self.nonstop_human = config.sim.nonstop_human
        self.centralized_planning = config.sim.centralized_planning
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        self.multi_policy = config.sim.multi_policy
        human_policy = config.humans.policy
        if self.centralized_planning:
            if human_policy == 'socialforce':
                logging.warning('Current socialforce policy only works in decentralized way with visible robot!')
            self.centralized_planner = policy_factory['centralized_' + human_policy]()

        self.orca_border = self.create_border_orca(config.env.dx_range, config.env.dy_range)
        self.sfm_border = self.create_border_sfm(config.env.dx_range, config.env.dy_range)
        self.x_width = (config.env.dx_range[1] - config.env.dx_range[0]) - (2 * config.robot.radius + 1e-2)
        self.y_width = (config.env.dy_range[1] - config.env.dy_range[0]) - (2 * config.robot.radius + 1e-2)
        self.min_dist_sum = 0.0
        self.min_dist_overall = 1e6
        self.num_steps = 0
        self.robot_velocities = []
        self.robot_accelerations = []

        self.current_scenario = self.test_scenario
        self.scenarios = None
        self.scenario_num = 0

        self.collided = False

    def set_robot(self, robot):
        self.robot = robot

    def create_border_orca(self, dx_range, dy_range):
        return [(dx_range[0], dy_range[1]), (dx_range[1], dy_range[1]), (dx_range[1], dy_range[0]), (dx_range[0], dy_range[0])]
    
    def create_border_sfm(self, dx_range, dy_range):
        lower = [dx_range[0], dx_range[1], dy_range[0] - 1, dy_range[0]]
        upper = [dx_range[0], dx_range[1], dy_range[1], dy_range[1] + 1]
        left = [dx_range[0] - 1, dx_range[0], dy_range[0], dy_range[1]]
        right = [dx_range[1], dx_range[1] + 1, dy_range[0], dy_range[1]]
        return [lower, upper, left, right]
    
    def min_dist_to_human(self):
        min_dist = 1e6
        robot_state = self.robot.get_full_state()
        for human in self.humans:
            state = human.get_full_state()
            dist = np.sqrt((robot_state.px - state.px)**2 + (robot_state.py - state.py)**2)
            if dist < min_dist:
                min_dist = dist
            
        return min_dist
    
    def generate_human_from_state(self, policy, state, human=None):
        if human is None:
            if self.multi_policy:
                if policy == 'static':
                    human = Human(self.config, 'humans', policy=policy_factory['linear']())
                else:
                    human = Human(self.config, 'humans', policy=policy_factory[policy]())
            else:
                human = Human(self.config, 'humans')
            if self.randomize_attributes:
                human.sample_random_attributes()
        human.set(*state)
        if policy == 'static':
            human.set(0.5, 3, 2, 4, 0, 0, np.pi / 2)
            human.v_pref = 1e-4
        if policy == 'socialforce':
            human.set(1.8, 4, 0, -4, 0, 0, np.pi / 2)


        return human

    def reset(self, phase='test', scenario=None, goals=None, test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        if self.robot is None:
            raise AttributeError('Robot has to be set!')

        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0

        base_seed = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                     'val': 0, 'test': self.case_capacity['val']}
        
        if self.robot.policy.name == 'vecmpc' or self.robot.policy.name == 'vecmppi':
            self.robot.policy.reset()

        self.collided = False

        self.min_dist_overall = 1e6
        self.robot_velocities = []

        self.robot.set(0, -4, 2, 4, 0, 0, np.pi / 2)
        if self.case_counter[phase] >= 0:
            seed = base_seed[phase] + self.case_counter[phase] + 0
            if self.random_seed:
                seed = random.randint(1000, 10000)
            if phase == 'test':
                logging.debug('current test seed is:{}'.format(base_seed[phase] + self.case_counter[phase]))
            if not self.robot.policy.multiagent_training and phase in ['train', 'val']:
                # only CADRL trains in circle crossing simulation
                human_num = 1
                self.current_scenario = 'circle_crossing'
            else:
                self.current_scenario = self.test_scenario
                human_num = self.human_num
            self.humans = []
            if self.multi_policy:
                print()
                if scenario is not None:
                    #print("RESET TIME")
                    for policy in scenario:
                        for n in range(len(scenario[policy])):
                            human = self.generate_human_from_state(policy, scenario[policy][n])
                            self.humans.append(human)
                            human.goals = goals[policy][n]
                            #print("STATE: ", scenario[policy][n])
                else:
                    self.set_num_policies()
                    for policy in self.num_policies:
                        for _ in range(self.num_policies[policy]):
                            self.humans.append(self.generate_human(human=None, policy=policy))
            else:
                for _ in range(human_num):
                    self.humans.append(self.generate_human())

            # case_counter is always between 0 and case_size[phase]
            self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
        else:
            if self.random_seed:
                seed = random.randint(1000, 10000)

            assert phase == 'test'
            if self.case_counter[phase] == -1:
                # for debugging purposes
                self.human_num = 3
                self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
            else:
                raise NotImplementedError
        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        if self.centralized_planning:
            self.centralized_planner.time_step = self.time_step

        self.states = list()
        self.robot_actions = list()
        self.rewards = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()
        if hasattr(self.robot.policy, 'get_matrix_A'):
            self.As = list()
        if hasattr(self.robot.policy, 'get_feat'):
            self.feats = list()
        if hasattr(self.robot.policy, 'get_X'):
            self.Xs = list()
        if hasattr(self.robot.policy, 'trajs'):
            self.trajs = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob = self.compute_observation_for(self.robot)
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        return ob

    def onestep_lookahead(self, action):
        return self.step(action, update=False)
    
    def outside_check(self, agent_state, obstacle, action):
        #print("KINEMATICS: ", self.robot.kinematics)
        if agent_state.kinematics == 'holonomic':
            px = agent_state.px + action.vx * self.time_step
            py = agent_state.py + action.vy * self.time_step

            left = px - agent_state.radius < obstacle[0][0]
            right = px + agent_state.radius > obstacle[1][0]
            if left or right:
                vx = 0.0
            else:
                vx = action.vx

            below = py - agent_state.radius < obstacle[2][1]
            above = py + agent_state.radius > obstacle[1][1]
            if below or above:
                vy = 0.0
            else:
                vy = action.vy

            return ActionXY(vx, vy)
        else:
            theta = agent_state.theta + action.r
            px = agent_state.px + np.cos(theta) * action.v * self.time_step
            py = agent_state.py + np.sin(theta) * action.v * self.time_step

            left = px - agent_state.radius < obstacle[0][0]
            right = px + agent_state.radius > obstacle[1][0]
            if left or right:
                v = 0.0
            else:
                v = action.v

            below = py - agent_state.radius < obstacle[2][1]
            above = py + agent_state.radius > obstacle[1][1]
            if below or above:
                v = 0.0
            else:
                v = action.v

            return ActionRot(v, action.r)

    def step(self, action, update=True, baseline=None):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        self.num_steps = self.num_steps + 1
        if self.centralized_planning:
            agent_states = [human.get_full_state() for human in self.humans]
            agent_policies = [human.policy for human in self.humans]

            if self.robot.visible:
                agent_states.append(self.robot.get_full_state())
                if self.multi_policy:
                    human_actions, robot_actions = self.centralized_planner.predict(agent_states, agent_policies, self.orca_border, self.orca_border)
                else:
                    human_actions = self.centralized_planner.predict(agent_states)[:-1]
            else:
                if self.multi_policy:
                    human_actions, robot_actions = self.centralized_planner.predict(agent_states, agent_policies, self.orca_border, self.orca_border)
                else:
                    human_actions = self.centralized_planner.predict(agent_states)
        else:
            human_actions = []
            i = 0
            for human in self.humans:
                i = i + 1
                ob = self.compute_observation_for(human)
                human_actions.append(human.act(ob))

        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                vx = human.vx - action.vx
                vy = human.vy - action.vy
            else:
                vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            #closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            closest_dist = np.sqrt(px**2 + py**2) - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                logging.info("Collision: distance between robot and p{} is {:.2E} at time {:.2E}".format(human.id, closest_dist, self.global_time))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius
        #print("HOW CLOSE ARE WE: ", norm(end_position - np.array(self.robot.get_goal_position())), self.robot.radius)
        self.min_dist_sum = self.min_dist_sum + self.min_dist_to_human()
        if self.min_dist_overall > self.min_dist_to_human():
            self.min_dist_overall = self.min_dist_to_human()

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
            self.collided = True
            print("COLLISION AHHHHHHHHHHHHHHHHHHHHHH")
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Discomfort(dmin)
        else:
            reward = 0
            done = False
            info = Nothing()

        if update:
            # store state, action value and attention weights
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())
            if hasattr(self.robot.policy, 'get_matrix_A'):
                self.As.append(self.robot.policy.get_matrix_A())
            if hasattr(self.robot.policy, 'get_feat'):
                self.feats.append(self.robot.policy.get_feat())
            if hasattr(self.robot.policy, 'get_X'):
                self.Xs.append(self.robot.policy.get_X())
            if hasattr(self.robot.policy, 'traj'):
                self.trajs.append(self.robot.policy.get_traj())

            # update all agents
            self.robot_velocities.append([self.robot.vx, self.robot.vy])
            if len(self.robot_velocities) > 2:
                vel1 = self.robot_velocities[len(self.robot_velocities) - 1]
                vel2 = self.robot_velocities[len(self.robot_velocities) - 2]
                ax = (vel1[0] - vel2[0]) / self.time_step
                ay = (vel1[1] - vel2[1]) / self.time_step
                self.robot_accelerations.append(np.sqrt(ax**2 + ay**2))
            if baseline is not None:
                if baseline == 'orca':
                    action = robot_actions[1]
                elif baseline == 'sfm':
                    action = robot_actions[0]
            action = self.outside_check(self.robot, self.orca_border, action)
            self.robot.step(action)
            for human, action in zip(self.humans, human_actions):
                action = self.outside_check(human, self.orca_border, action)
                human.step(action)

                if human.reached_destination():
                    if self.nonstop_human:
                        if human.v_pref > 1e-2:
                            agents = [human.get_set_state() for human in self.humans]
                            agents.append(self.robot.get_set_state())
                            #print("PREVIOUS GOAL: ", human.gx, human.gy) 
                            #self.generate_human_from_state(policy=human.policy.name, state=utils.generate_human_state(agents, self.x_width, self.y_width, self.discomfort_dist, None, self.current_scenario, start=(human.px, human.py)), human=human)
                            #print("HUMAN CURRENT GOAL: ", human.current_goal, len(human.goals), len(human.goals[human.current_goal]))
                            human.gx = human.goals[human.current_goal][0]
                            human.gy = human.goals[human.current_goal][1]
                            human.current_goal = human.current_goal + 1
                            #print("NEXT GOAL: ", human.gx, human.gy, human.current_goal)
                    else:
                        human.v_pref = 1e-2
                        human.vx = 0.0
                        human.vy = 0.0
                        human.policy = policy_factory['linear']()

            self.global_time += self.time_step
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans],
                                [human.id for human in self.humans]])
            self.robot_actions.append(action)
            self.rewards.append(reward)

            # compute the observation
            if self.robot.sensor == 'coordinates':
                ob = self.compute_observation_for(self.robot)
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        else:
            if self.robot.sensor == 'coordinates':
                ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
            
        #print("DONE FIRST STEP")

        return ob, reward, done, info

    def compute_observation_for(self, agent):
        if agent == self.robot:
            ob = []
            for human in self.humans:
                ob.append(human.get_observable_state())
        else:
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != agent]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
        return ob

    def render(self, mode='video', output_file=None):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        x_offset = 0.2
        y_offset = 0.4
        cmap = plt.cm.get_cmap('hsv', 3)
        robot_color = 'black'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)
        display_numbers = True

        if mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))

            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add human start positions and goals
            #human_colors = [cmap(i) for i in range(len(self.humans))]
            human_colors = []
            for h in self.humans:
                if isinstance(h.policy, SocialForce):
                    human_colors.append(cmap(0))
                elif isinstance(h.policy, ORCA):
                    human_colors.append(cmap(1))
                elif isinstance(h.policy, Linear):
                    human_colors.append(cmap(2))

            for i in range(len(self.humans)):
                human = self.humans[i]
                human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                           color=human_colors[i],
                                           marker='*', linestyle='None', markersize=15)
                ax.add_artist(human_goal)
                human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                            color=human_colors[i],
                                            marker='o', linestyle='None', markersize=15)
                ax.add_artist(human_start)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]

            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=False, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)

                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 1)]
                    for time in times:
                       ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
            ax.tick_params(bottom=False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            #ax.set_xlabel('x(m)', fontsize=14)
            #ax.set_ylabel('y(m)', fontsize=14)
            show_human_start_goal = False
            display_numbers = False

            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([self.robot.gx], [self.robot.gy],
                                 color=robot_color, marker='*', linestyle='None',
                                 markersize=15, label='Goal')
            other_goals = mlines.Line2D([self.other_goals[:,0]], [self.other_goals[:,1]],
                                color='blue', marker='*', linestyle='None',
                                markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            # sensor_range = plt.Circle(robot_positions[0], self.robot_sensor_range, fill=False, ls='dashed')
            ax.add_artist(robot)
            ax.add_artist(goal)
            ax.add_artist(other_goals)
            for i in range(3):
                ax.plot([self.orca_border[i][0], self.orca_border[i+1][0]], [self.orca_border[i][1], self.orca_border[i+1][1]], color='black')
            ax.plot([self.orca_border[3][0], self.orca_border[0][0]], [self.orca_border[3][1], self.orca_border[0][1]], color='black')

            human_colors = []
            for h in self.humans:
                if isinstance(h.policy, SocialForce):
                    human_colors.append('#4575b4')
                elif isinstance(h.policy, ORCA):
                    human_colors.append('#1a9850')
                elif isinstance(h.policy, Linear):
                    human_colors.append('#998ec3')

            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius)
                      for i in range(len(self.humans))]
            humans_patch = PatchCollection(humans, cmap=plt.cm.tab10, alpha=1.0)
            ax.add_collection(humans_patch)
            global_step = 0

            frames = len(self.states)
            if self.collided:
                frames = len(self.states) - 1
            collide = False

            def update(frame_num):
                nonlocal global_step
                nonlocal frames
                nonlocal collide
                global_step = frame_num

                robot.center = robot_positions[frame_num]

                patches = []
                for i in range(len(human_positions[frame_num])):
                    circle = plt.Circle(human_positions[frame_num][i], self.humans[i].radius)
                    patches.append(circle)

                humans_patch.set_paths(patches)
                colors_list = []
                for i in range(len(human_colors)):
                    colors_list.append(colors.to_rgb(human_colors[i]))
                    if np.sqrt((robot.center[0] - human_positions[frame_num][i][0])**2 + (robot.center[1] - human_positions[frame_num][i][1])**2) < 0.6:
                        colors_list[i] = colors.to_rgb('#ff0000')
                        if not collide:
                            frames = frame_num
                            collide = True
                color_arr = np.array(colors_list)
                humans_patch.set_facecolor(color_arr)

            anim = animation.FuncAnimation(fig, update, frames=frames, interval=self.time_step * 500, blit=False)

            if output_file is not None:
                # save as video
                ffmpeg_writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                # writer = ffmpeg_writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=ffmpeg_writer)

                # save output file as gif if imagemagic is installed
                # anim.save(output_file, writer='imagemagic', fps=12)
            plt.show()
        else:
            raise NotImplementedError
