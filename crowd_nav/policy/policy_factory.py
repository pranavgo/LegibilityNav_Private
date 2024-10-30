from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl import SARL
from crowd_nav.policy.gcn import GCN
from crowd_nav.policy.dwa import DynamicWindowApproach
from crowd_nav.policy.legible import Legible
from crowd_nav.policy.model_predictive_rl import ModelPredictiveRL
from crowd_nav.policy.vecMPC.controller import vecMPC
from crowd_nav.policy.vecMPC.controller_mppi import vecMPPI
from crowd_nav.policy.social_momentum import SMLegible
from crowd_nav.policy.lpsnav import LPSNavPolicy

policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['gcn'] = GCN
policy_factory['model_predictive_rl'] = ModelPredictiveRL
policy_factory['vecmpc'] = vecMPC
policy_factory['vecmppi'] = vecMPPI
policy_factory['dwa'] = DynamicWindowApproach
policy_factory['legible'] = Legible
policy_factory['sm_legible'] = SMLegible
policy_factory['lpsnav'] = LPSNavPolicy
