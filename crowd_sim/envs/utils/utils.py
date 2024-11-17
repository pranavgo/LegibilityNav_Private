import numpy as np
from numpy.linalg import norm
import random
from shapely.geometry import Point
from shapely.geometry import MultiPolygon, Polygon

def point_to_segment_dist(x1, y1, x2, y2, x3, y3):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

    """
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return np.linalg.norm((x3-x1, y3-y1))

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return np.linalg.norm((x - x3, y-y3))

def set_params_crossing_fixed(radius, agents, x_width, y_width, discomfort_dist):
    if np.random.random() > 0.5:
            sign = -1
    else:
        sign = 1
    while True:
        px = np.random.random() * x_width * 0.5 * sign
        py = (np.random.random() - 0.5) * y_width
        collide = False
        for agent in agents:
            if norm((px - agent[0], py - agent[1])) < radius + 0.3 + discomfort_dist:
                collide = True
                break
        if not collide:
            break
    while True:
        gx = np.random.random() * x_width * 0.5 * - sign
        gy = (np.random.random() - 0.5) * y_width
        collide = False
        for agent in agents:
            if norm((gx - agent[2], gy - agent[3])) < radius + 0.3 + discomfort_dist:
                collide = True
                break
        if not collide:
            break
    return px, py, gx, gy, 0, 0, 0

def set_params_passing_fixed(radius, agents, x_width, y_width, discomfort_dist):
    if np.random.random() > 0.5:
            sign = -1
    else:
        sign = 1
    while True:
        py = np.random.random() * y_width * 0.5 * sign
        px = (np.random.random() - 0.5) * x_width
        collide = False
        for agent in agents:
            if norm((px - agent[0], py - agent[1])) < radius + 0.3 + discomfort_dist:
                collide = True
                break
        if not collide:
            break
    while True:
        gy = np.random.random() * y_width * 0.5 * - sign
        if px > 0:
            gx = (np.random.random() * 0.5) * x_width
        else:
            gx = -1 * (np.random.random() * 0.5) * x_width
        collide = False
        for agent in agents:
            if norm((gx - agent[2], gy - agent[3])) < radius + 0.3 + discomfort_dist:
                collide = True
                break
        if not collide:
            break
    return px, py, gx, gy, 0, 0, 0

def get_goal_sequence(num_goals, g, current_scenario, x_width, y_width):
    goals = []
    #goals.append(g)
    prev_goal = g
    for i in range(num_goals):
        #print("I ", i)
        if current_scenario == 'passing':
            if prev_goal[0] > 0:
                gx = (np.random.random() * 0.5) * x_width
            else:
                gx = -1 * (np.random.random() * 0.5) * x_width
            if prev_goal[1] > 0:
                gy = np.random.random() * y_width * 0.5 * -1
            else:
                gy = np.random.random() * y_width * 0.5
            goals.append((gx, gy))
            prev_goal = (gx, gy)
        elif current_scenario == 'crossing':
            if prev_goal[0] > 0:
                gx = np.random.random() * x_width * 0.5 * -1
            else:
                gx = np.random.random() * x_width * 0.5
            gy = (np.random.random() - 0.5) * y_width
            goals.append((gx, gy))
            prev_goal = (gx, gy)
        elif current_scenario == 'circle_crossing':
            gx = -1 * prev_goal[0]
            gy = -1 * prev_goal[1]
            goals.append((gx, gy))
            prev_goal = (gx, gy)
        elif current_scenario == 'random':
            gy = (np.random.random() - 0.5) * y_width
            gx = (np.random.random() - 0.5) * x_width
            goals.append((gx, gy))
            prev_goal = (gx, gy)

    #print("LEN GOALS: ", len(goals), current_scenario)

    return goals

def generate_human_state(agents, x_width, y_width, discomfort_dist, policy=None, current_scenario='passing_crossing', obstacles = None):
        params = None
        if current_scenario == 'circle_crossing':
            while True:
                angle = np.random.random() * np.pi/2
                # add some noise to simulate all the possible cases robot could meet with human
                px_noise = (np.random.random() - 0.5) * 1.0
                py_noise = (np.random.random() - 0.5) * 1.0
                px = 4.0 * np.cos(angle) + px_noise
                py = 4.0 * np.sin(angle) + py_noise
                if np.random.random() > 0.5:
                    px = px * -1

                if np.random.random() > 0.5:
                    py = py * -1

                collide = False
                for agent in agents:
                    min_dist = 0.3 + 0.3 + discomfort_dist
                    if norm((px - agent[0], py - agent[1])) < min_dist or \
                            norm((px - agent[2], py - agent[3])) < min_dist:
                        collide = True
                        break
                if not collide:
                    break
            params = px, py, -px, -py, 0, 0, 0

        elif current_scenario == 'passing':
            params = set_params_passing_fixed(0.3, agents, x_width, y_width, discomfort_dist)

        elif current_scenario == 'crossing':
            params = set_params_crossing_fixed(0.3, agents, x_width, y_width, discomfort_dist)

        elif current_scenario == 'passing_crossing':
            if np.random.random() > 0.5:
                sign = -1
            else:
                sign = 1

            if sign == 1:
                params = set_params_passing_fixed(0.3, agents, x_width, y_width, discomfort_dist)
                current_scenario = 'passing'
            else:
                params = set_params_crossing_fixed(0.3, agents, x_width, y_width, discomfort_dist)
                current_scenario = 'crossing'

        elif current_scenario == 'random':
            while True:
                py = (np.random.random() - 0.5) * y_width
                px = (np.random.random() - 0.5) * x_width
                collide = False
                for agent in agents:
                    if norm((px - agent[0], py - agent[1])) < 0.3 + 0.3 + discomfort_dist:
                        collide = True
                        break
                if not collide:
                    break
            while True:
                gy = (np.random.random() - 0.5) * y_width
                gx = (np.random.random() - 0.5) * x_width
                collide = False
                for agent in agents:
                    if norm((gx - agent[2], gy - agent[3])) < 0.3 + 0.3 + discomfort_dist:
                        collide = True
                        break
                if not collide:
                    break
            params = px, py, gx, gy, 0, 0, 0

        if policy == 'static':
            params = list(params)
            params[2] = params[0] + 1e-2
            params[3] = params[1] + 1e-2
            params = tuple(params)

        return params, current_scenario

def generate_scenarios_fixed(length, radius, x_width, y_width, discomfort_dist, num_orca, num_sf, num_linear, num_static, current_scenario, obstacles):
        states = []
        goals = []
        num_policies = {
            'orca' : num_orca,
            'socialforce' : num_sf,
            'linear' : num_linear,
            'static' : num_static
        }
        for i in range(length):
            states_i = {}
            goals_i = {}
            agents = [(0, -4, 0, 4, 0, 0, np.pi / 2)]
            for policy in num_policies:
                for _ in range(num_policies[policy]):
                    if policy in states_i:
                        params, current_scenario_mod = generate_human_state(agents, x_width, y_width, discomfort_dist, policy=policy, current_scenario=current_scenario, obstacles=obstacles)
                        
                        states_i[policy].append(params)
                        goal_list = get_goal_sequence(100, (params[2], params[3]), current_scenario_mod, x_width, y_width)
                        goals_i[policy].append(goal_list)
                        agents.append(params)
                    else:
                        params, current_scenario_mod = generate_human_state(agents, x_width, y_width, discomfort_dist, policy=policy, current_scenario=current_scenario, obstacles=obstacles)
                        states_i[policy] = [params]
                        goals_i[policy] = [get_goal_sequence(100, (params[2], params[3]), current_scenario_mod, x_width, y_width)]
            states.append(states_i)
            goals.append(goals_i)

        return states, goals

def random_sequence(ec,en, length=500, radius=0.3, discomfort_dist=0.2): #Does not work with randomized radii yet, fixed at 0.3m for now
    if ec.exp.random_seed:
        seed = np.random.randint(1000, 10000)
    else:
        seed = 1000
    np.random.seed(seed)
    random.seed(seed)

    scenarios = []
    goals = []

    for e in range(len(ec.exp.dx)):
        scenarios_e = []
        goals_e = []
        for se in range(len(ec.exp.dx[e])):
            # configure environment
            x_width = (ec.exp.dx[e][se][1] - ec.exp.dx[e][se][0]) - (2 * radius + 1e-2)
            y_width = (ec.exp.dy[e][se][1] - ec.exp.dy[e][se][0]) - (2 * radius + 1e-2)
            num_sf = ec.exp.num_sf[e][se][0]
            num_orca = ec.exp.num_orca[e][se][0]
            num_static = ec.exp.num_static[e][se][0]
            num_linear = ec.exp.num_linear[e][se][0]
            test_scenario = ec.exp.scenarios[e][se]
            if en.env.obstacle:
                s, g = generate_scenarios_fixed(length, radius, x_width, y_width, discomfort_dist, num_orca, num_sf, num_linear, num_static, test_scenario,en.env.static_obstacles)
            else:
                s, g = generate_scenarios_fixed(length, radius, x_width, y_width, discomfort_dist, num_orca, num_sf, num_linear, num_static, test_scenario,None)
            scenarios_e.append(s)
            goals_e.append(g)
        
        scenarios.append(scenarios_e)
        goals.append(goals_e)

    return scenarios, goals
