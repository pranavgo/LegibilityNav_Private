import numpy as np
import os
import yaml

class Config(object):
    def __init__(self):
        pass

class BaseExperimentsConfig(object):
    exp = Config()

# change the ninth param below.
    exp.num_orca = [[[2], [5], [7], [10], [12], [15], [17]], [[1], [0], [2], [3], [3], [4], [4]], [[7], [7], [7], [7], [7]], [[0], [15], [7], [5], [4]]]
    exp.num_sf = [[[3], [5], [8], [10], [13], [15], [18]], [[2], [0], [3], [3], [4], [4], [5]], [[8], [8], [8], [8], [8]], [[15], [0], [8], [5], [4]]]

    exp.num_linear = [[[0], [0], [0], [0], [0], [0], [0]], [[0], [0], [0], [0], [0], [0], [0]], [[0], [0], [0], [0], [0]], [[0], [0], [0], [2], [4]]]
    exp.num_static = [[[0], [0], [0], [0], [0], [0], [0]], [[0], [1], [0], [0], [0], [0], [0]], [[0], [0], [0], [0], [0]], [[0], [0], [0], [3], [3]]]

    exp.randomize_attributes = [[False, False, False, False, False, False, False], [False, False, False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False]]
    
    exp.scenarios = [['passing_crossing', 'passing_crossing', 'passing_crossing', 'passing_crossing', 'passing_crossing', 'passing_crossing', 'passing_crossing'],
                             ['passing_crossing', 'passing_crossing', 'passing_crossing', 'passing_crossing', 'passing_crossing', 'passing_crossing', 'passing_crossing'],
                             ['passing', 'crossing', 'passing_crossing', 'random', 'circle_crossing'],
                             ['passing_crossing', 'passing_crossing', 'passing_crossing', 'passing_crossing', 'passing_crossing']]
    
    exp.parameter_sweep = None
    exp.sigma = [0.3, 0.6, 0.9]
    exp.q = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    
    exp.dx = [[[-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5]], [[-0.75, 0.75], [-5, 5], [-1.25, 1.25], [-1.5, 1.5], [-1.75, 1.75], [-2, 2], [-2.25, 2.25]], [[-5, 5], [-5, 5], [-5, 5], [-5, 5], [-6, 6]], [[-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5]]]
    exp.dy = [[[-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5]], [[-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5]], [[-5, 5], [-5, 5], [-5, 5], [-5, 5], [-6, 6]], [[-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5]]]
    
    exp.parameter_sweep = None
    exp.sigma = [0.3, 0.6, 0.9]
    exp.q = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    exp.noise = [1.0, 2.0, 3.0, 4.0, 5.0]
    exp.samples = [100, 250, 500, 1000]
    exp.horizon = [3, 4, 5, 6, 7]

    exp.random_seed = True
    
    def __init__(self, debug=False):
        pass

class BaseEnvConfig(object):
    env = Config()
    env.time_limit = 50
    env.time_step = 0.25
    env.val_size = 500
    env.test_size = 500
    env.train_size = np.iinfo(np.uint32).max - 2000
    env.randomize_attributes = False #TODO: Make this work with pre-generating scenarios
    env.robot_sensor_range = 5
    env.dx_range = [-5, 5]
    env.dy_range = [-5, 5]
    env.other_goals = np.array([[-2,4]])

    reward = Config()
    reward.success_reward = 1
    reward.collision_penalty = -0.25
    reward.discomfort_dist = 0.2
    reward.discomfort_penalty_factor = 0.5

    sim = Config()
    sim.train_val_scenario = 'passing_crossing'
    sim.test_scenario = 'passing_crossing'
    sim.square_width = 10
    sim.circle_radius = 4
    sim.human_num = 5
    sim.nonstop_human = True
    sim.centralized_planning = True
    sim.multi_policy = True
    sim.random_seed = True #Should match the setting of exp.random_seed

    humans = Config()
    humans.visible = True
    humans.policy = 'multipolicy'
    humans.radius = 0.3
    humans.v_pref = 1
    humans.sensor = 'coordinates'
    humans.num_sf = [0]
    humans.num_orca = [4]
    humans.num_static = [0]
    humans.num_linear = [0]
    humans.num_sf_orca = None
    humans.num_linear_static = None

    robot = Config()
    robot.visible =True
    robot.policy = 'none'
    robot.radius = 0.3
    robot.v_pref = 1.0
    robot.sensor = 'coordinates'
    robot.reactive = True

    # MPC = Config()
    # MPC.model = 'sgan' #Change to sgan/cv for mpc methods
    # MPC.path = '/home/socnav/arstr/RelationalGraphLearning/crowd_nav/configs/params/sgan_mppi.yaml' #Change to sgan/cv for mpc methods
    # with open(MPC.path, "r") as fin:
    #     MPC.mpc = yaml.safe_load(fin)
    # print("MPC: ", MPC.mpc)
    # MPC.mpc['params']['dt'] = 0.25
    # MPC.mpc['params']['prediction_length'] = 1.75
    # MPC.save_path = "REAL_WORLD"
    # MPC.exp_name = "REAL_WORLD"
    # MPC.multiagent_training = True

    def __init__(self, debug=False):
        if debug:
            self.env.val_size = 1
            self.env.test_size = 1


class BasePolicyConfig(object):
    rl = Config()
    rl.gamma = 0.9

    om = Config()
    om.cell_num = 4
    om.cell_size = 1
    om.om_channel_size = 3

    action_space = Config()
    action_space.kinematics = 'holonomic'
    action_space.speed_samples = 5
    action_space.rotation_samples = 16
    action_space.sampling = 'exponential'
    action_space.query_env = False
    action_space.rotation_constraint = np.pi / 3

    cadrl = Config()
    cadrl.mlp_dims = [150, 100, 100, 1]
    cadrl.multiagent_training = False

    lstm_rl = Config()
    lstm_rl.global_state_dim = 50
    lstm_rl.mlp1_dims = [150, 100, 100, 50]
    lstm_rl.mlp2_dims = [150, 100, 100, 1]
    lstm_rl.multiagent_training = True
    lstm_rl.with_om = False
    lstm_rl.with_interaction_module = True

    srl = Config()
    srl.mlp1_dims = [150, 100, 100, 50]
    srl.mlp2_dims = [150, 100, 100, 1]
    srl.multiagent_training = True
    srl.with_om = True

    sarl = Config()
    sarl.mlp1_dims = [150, 100]
    sarl.mlp2_dims = [100, 50]
    sarl.attention_dims = [100, 100, 1]
    sarl.mlp3_dims = [150, 100, 100, 1]
    sarl.multiagent_training = True
    sarl.with_om = True
    sarl.with_global_state = True

    gcn = Config()
    gcn.multiagent_training = True
    gcn.num_layer = 2
    gcn.X_dim = 32
    gcn.wr_dims = [64, gcn.X_dim]
    gcn.wh_dims = [64, gcn.X_dim]
    gcn.final_state_dim = gcn.X_dim
    gcn.gcn2_w1_dim = gcn.X_dim
    gcn.planning_dims = [150, 100, 100, 1]
    gcn.similarity_function = 'embedded_gaussian'
    gcn.layerwise_graph = True
    gcn.skip_connection = False

    gnn = Config()
    gnn.multiagent_training = True
    gnn.node_dim = 32
    gnn.wr_dims = [64, gnn.node_dim]
    gnn.wh_dims = [64, gnn.node_dim]
    gnn.edge_dim = 32
    gnn.planning_dims = [150, 100, 100, 1]

    dwa = Config()
    dwa.acc_max = 0.5
    dwa.ang_acc_max = 1.04
    dwa.ang_acc_res_deg = 0.5
    dwa.max_d = 2.0


    def __init__(self, debug=False):
        pass


class BaseTrainConfig(object):
    trainer = Config()
    trainer.batch_size = 100
    trainer.optimizer = 'Adam'

    imitation_learning = Config()
    imitation_learning.il_episodes = 2000
    imitation_learning.il_policy = 'orca'
    imitation_learning.il_epochs = 50
    imitation_learning.il_learning_rate = 0.001
    imitation_learning.safety_space = 0.15

    train = Config()
    train.rl_train_epochs = 1
    train.rl_learning_rate = 0.001
    # number of batches to train at the end of training episode il_episodes
    train.train_batches = 100
    # training episodes in outer loop
    train.train_episodes = 10000
    # number of episodes sampled in one training episode
    train.sample_episodes = 1
    train.target_update_interval = 1000
    train.evaluation_interval = 1000
    # the memory pool can roughly store 2K episodes, total size = episodes * 50
    train.capacity = 100000
    train.epsilon_start = 0.5
    train.epsilon_end = 0.1
    train.epsilon_decay = 4000
    train.checkpoint_interval = 1000

    train.train_with_pretend_batch = False

    def __init__(self, debug=False):
        if debug:
            self.imitation_learning.il_episodes = 10
            self.imitation_learning.il_epochs = 5
            self.train.train_episodes = 1
            self.train.checkpoint_interval = self.train.train_episodes
            self.train.evaluation_interval = 1
            self.train.target_update_interval = 1
