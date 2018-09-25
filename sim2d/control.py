"""
Finite-State MDP Implementation for TransportAI

states:
    up, down, left, right for position
    wall
    up, down, left, right for velocity
    ^ 9 total states
    for each, close or far
    2^9 = 512 states

actions:
    4 accelerations
    3 directions
    12 actions
"""

import math
import numpy as np
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
utils_dir = os.path.join(parent_dir, "utils")
sys.path.append(utils_dir)
from constants import ENVS, CAR_RADIUS, NUM_CARS, TIME_STEP  # TODO tell srinu using CAR_RADIUS
# env_dir = os.path.join(parent_dir, "env")
# sys.path.append(env_dir)
from car import Car
from env import Environment

MAX_ACC = 8  # max acceleration
MAX_DEC = -16 # 16  # max deceleration
NUM_ACCS = 4 # 4
ACC_DIFF = (MAX_ACC - MAX_DEC) / (NUM_ACCS - 1)

# MAX_ROT = math.pi / 6
# MIN_ROT = -math.pi / 6
MAX_ROT = 0.0
MIN_ROT = 0.0

NUM_ROT = 3
rot_dict = {  # maps index to rotation angle
    0: 0.0,  # 0.0
    1: MAX_ROT,  # MAX_ROT
    2: MIN_ROT
}

NUM_ACTIONS = NUM_ACCS * NUM_ROT  # todo automate computation
NUM_ACTION_I = [NUM_ACCS, NUM_ROT]

NUM_POS_STATE = 3  # number of possible values for position info; note: if these
NUM_WALL_STATE = 2
NUM_VEL_STATE = 3  # values are changed, should also change get_.*{2} functions
NUM_STATE_I = [NUM_POS_STATE,
               NUM_POS_STATE,
               NUM_POS_STATE,
               NUM_POS_STATE,
               NUM_WALL_STATE,
               NUM_VEL_STATE,
               NUM_VEL_STATE,
               NUM_VEL_STATE,
               NUM_VEL_STATE]
NUM_STATES = np.prod(NUM_STATE_I)

TIME_STEP = 0.1  # in seconds
MAX_TIME = 300
GAMMA = 0.995 # 0.9 # 0.995
TOLERANCE = 0.00000001 # max change in value function to declare convergence
NO_LEARNING_THRESHOLD = 50 # 20  # max number of no learning instances before

def load_policy(file_name='policy.npy'):
    if not os.path.exists(file_name):
        raise ValueError("No policy.npy found")

    return np.load(file_name)


def get_accel_list(policy, cars):
    """
    Args:
        cars: list of Car objects
    """
    return [get_action(i) for i in get_actions(cars, policy)]


def index_to_action(index):
    """maps integer in range [0, NUM_ACTIONS) to size-2 tuple action"""
    a = [0, 0]
    # print("List", NUM_ACTION_I)
    for i in reversed(range(2)):
        a[i] = index % (NUM_ACTION_I[i])
        index = int(index / (NUM_ACTION_I[i]))
        # print("a is", a)
    return tuple(a)


def action_to_index(a):
    """maps size-2 tuple action to integer in range[0, NUM_ACTIONS)"""
    index = 1
    multiplier = 1
    for i, ai in reversed(list(enumerate(a))):
        index += ai * multiplier
        multiplier *= NUM_ACTION_I[i]
    return index-1


def get_action(index):
    (ai, ri) = index_to_action(index)
    # print("ai is", ai)
    return MAX_ACC - ai * ACC_DIFF, rot_dict[ri]


def get_rot(index):
    return rot_dict[index]


def pos_to_i(x, y):
    """converts position to indices in env mat"""
    return int(x), int(y)


def get_closest_walls(xi, yi, env_mat):
    """north, south, east, west;
    a returned pair is (-1, -1) if there's nothing in that direction
    returns size-4 tuple of pairs each representing loc in env_mat"""

    # get north, decrement y
    xs = -1
    ys = -1
    y = yi
    while y >= 0:
        if env_mat[xi][y] == 0:
            xs = xi
            ys = y
            break
        y -= 1

    # get south, increment y
    xn = -1
    yn = -1
    y = yi
    while y < len(env_mat):
        if env_mat[xi][y] == 0:
            xn = xi
            yn = y
            break
        y += 1

    # get east, increment x
    xe = -1
    ye = -1
    x = xi
    while x < len(env_mat[0]):
        if env_mat[x][yi] == 0:
            xe = x
            ye = yi
            break
        x += 1

    # get west, decrement x
    xw = -1
    yw = -1
    x = xi
    while x >= 0:
        if env_mat[x][yi] == 0:
            xw = x
            yw = yi
            break
        x -= 1

    return (xn, yn), (xs, ys), (xe, ye), (xw, yw)


def get_closest_cars(xi, yi, env_mat, car_is):
    """
    north, south, east, west;
    car_is is a list of (xi, yi) tuples (including the tuple for given car)
    scans range [index-CAR_RADIUS, index+CAR_RADIUS]

    Return:
        size-4 tuple of indices, each representing index in car_is
        a returned index is -1 if there is no car in that direction
    """
    env_len = len(env_mat)
    env_width = len(env_mat[0])

    half_range = 2*CAR_RADIUS
    x_min = max(0, xi - half_range)
    x_max = min(xi + half_range, env_width - 1)
    y_min = max(0, yi - half_range)
    y_max = min(yi + half_range, env_len - 1)

    # get north, decrement y
    cn = -1
    y = yi
    not_found = True
    while y >= 0 and not_found:
        for x in range(x_min, x_max + 1):
            if (x, y) in car_is:
                cs = car_is.index((x, y))
                not_found = False
                break
        y -= 1

    # get south, increment y
    cs = -1
    y = yi
    not_found = True
    while y < env_len and not_found:
        for x in range(x_min, x_max + 1):
            if (x, y) in car_is:
                cs = car_is.index((x, y))
                not_found = False
                break
        y += 1

    # get east, increment x
    ce = -1
    x = xi
    not_found = True
    while x < env_width and not_found:
        for y in range(y_min, y_max + 1):
            if (x, y) in car_is:
                ce = car_is.index((x, y))
                not_found = False
                break
        x += 1

    # get west, decrement x
    cw = -1
    x = xi
    not_found = True
    while x >= 0 and not_found:
        for y in range(y_min, y_max + 1):
            if (x, y) in car_is:
                cs = car_is.index((x, y))
                not_found = False
                break
        x -= 1

    return cn, cs, ce, cw


def get_pc(dist1, dist2):
    """get car1 relative dist to car2 (along an axis), returns a number"""
    rel_dist = abs(dist2 - dist1)
    if rel_dist >= 4:
        return 2
    if rel_dist >= 2:
        return 1
    return 0


def get_pw(wall_distance):
    """get car pos w.r.t. wall, returns a number"""
    if wall_distance >= 2:
        return 1
    return 0


def get_v(v_car, v_other):  # get car rel vel
    rel_v = v_car - v_other
    if rel_v >= 2:
        return 2
    if rel_v <= -2:
        return 0
    return 1


def get_state(car, env_mat, cars):
    """get the state of the given car

    Return:
        a size-9 tuple state"""
    xi, yi = pos_to_i(car.x, car.y)  # note: can also apply pos_to_i on this
    xi = min(xi, len(env_mat[0]) - 1)
    yi = min(yi, len(env_mat) - 1)
    vx = car.v_x
    vy = car.v_y
    car_is = [pos_to_i(c.x, c.y) for c in cars]
    ((xn, yn), (xs, ys), (xe, ye), (xw, yw)) = get_closest_walls(xi, yi, env_mat)
    (cn, cs, ce, cw) = get_closest_cars(xi, yi, env_mat, car_is)
    vn = 0
    vs = 0
    ve = 0
    vw = 0
    bn = CAR_RADIUS - abs(xi - car_is[cn][0])/2  # account for car boundary
    if cn != -1 and (yn == -1 or car_is[cn][1] + bn > yn):
        vn = cars[cn].v_y
        yn = car_is[cn][1] + bn
    bs = CAR_RADIUS - abs(xi - car_is[cs][0])/2  # account for car boundary
    if cs != -1 and (ys == -1 or car_is[cs][1] - bs < ys):
        vs = cars[cs].v_y
        ys = car_is[cs][1] - bs
    be = CAR_RADIUS - abs(yi - car_is[ce][1])/2  # account for car boundary
    if ce != -1 and (ye == -1 or car_is[ce][0] - be < xe):
        ve = cars[ce].v_x
        xe = car_is[ce][0] - be
    bw = CAR_RADIUS - abs(yi - car_is[cw][1])/2  # account for car boundary
    if cw != -1 and (yw == -1 or car_is[cw][0] + bw < xw):
        vw = cars[cw].v_x
        xw = car_is[cw][0] + bw
    return (get_pc(yi, yn),
            get_pc(yi, ys),
            get_pc(xi, xe),
            get_pc(xi, xw),
            get_pw(car.wall_distance),
            get_v(vy, vn),
            get_v(vy, vs),
            get_v(vx, ve),
            get_v(vx, vw))


def get_states(cars, env_mat):
    """

    Assuming that there are at least 2 cars

    Return:
        length-len(cars) list of size-5 tuple states
    """
    return [get_state(c, env_mat, cars) for c in cars]


def get_actions(cars, policy):  # TODO check if need
    """Computes actions given car states and policy

    Assuming that there are at least 2 cars

    Args:
        cars: list of Cars
        policy: 1-dim numpy array with length NUM_STATES

    Return:
        list of indices (index can be turned into acceleration value via get_acc)
    """
    state_indices = [state_to_index(s) for s in get_states(cars, ENVS["FOUR_WAY_1"])]
    return [policy[i] for i in state_indices]


def top_state(states):  # TODO check if need
    """Returns state, probability

    Args:
        states: dictionary with states as keys and probabilities as values
    """
    max_s = None
    max_p = -1
    for s in states:
        if states[s] > max_p:
            max_p = states[s]
            max_s = s
    return max_s, max_p


def index_to_state(index):
    """maps integer in range [0, NUM_STATES) to size-9 tuple state"""
    s = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in reversed(range(9)):
        s[i] = index % (NUM_STATE_I[i])
        index /= (NUM_STATE_I[i])
    return s


def state_to_index(s):
    """maps size-9 tuple state to integer in range [0, NUM_STATES)"""
    index = 1
    multiplier = 1
    for i, si in reversed(list(enumerate(s))):
        index += si * multiplier
        multiplier *= NUM_STATE_I[i]
    return index-1


def get_policy(P, V):
    """Returns optimal policy based on P and V

    Args:
        P: transition probability matrix,
            np array with shape (NUM_STATES, NUM_ACTIONS, NUM_STATES)
        V: value function, np array with shape (NUM_STATES)
    Return:
        np array of ints in the range [0, NUM_ACTIONS) with shape (NUM_STATES)
    """
    return np.argmax(np.matmul(P, V), axis=1)


def update_sa_counts(sa_counts, sas_counts, state_history, action_history):
    for s, a in zip(state_history[0], action_history[0]):
        if (s, a) in sa_counts:
            sa_counts[(s, a)] += 1
        else:
            sa_counts[(s, a)] = 1
    for i in range(1, len(state_history)):
        for j in range(len(state_history[0])):
            if (state_history[i][j], action_history[i][j]) in sa_counts:
                sa_counts[(state_history[i][j], action_history[i][j])] += 1
            else:
                sa_counts[(state_history[i][j], action_history[i][j])] = 1
            if (state_history[i - 1][j], action_history[i - 1][j]) in sas_counts:
                if state_history[i][j] in sas_counts[(state_history[i - 1][j], action_history[i - 1][j])]:
                    sas_counts[(state_history[i - 1][j], action_history[i - 1][j])][state_history[i][j]] += 1
                else:
                    sas_counts[(state_history[i - 1][j], action_history[i - 1][j])][state_history[i][j]] = 1
            else:
                sas_counts[(state_history[i - 1][j], action_history[i - 1][j])] = {state_history[i][j]: 1}
    return sa_counts, sas_counts


def update_P(P, sa_counts, sas_counts):
    """Updates transition probability matrix P given historical data

    Args:
        P: transition probability matrix,
            np array with shape (NUM_STATES, NUM_ACTIONS, NUM_STATES)
        sa_counts: list of length-num_cars integer lists, where ints are in
                       the range [0, NUM_STATES)
        sas_counts: list of length-num_cars integer lists, where ints are in
                        the range [0, NUM_ACTIONS)

    Return:
        new transition probability matrix P,
            np array with shape (NUM_STATES, NUM_ACTIONS, NUM_STATES)
    """
    for (s, a) in sa_counts:
        if (s, a) in sas_counts:
            for s2 in sas_counts[(s, a)]:
                P[s][a][s2] = sas_counts[(s, a)][s2] / sa_counts[(s, a)]
    return P


def update_sr_counts(s_counts, sr_counts, state_history, reward_history):
    for i in range(len(state_history)):
        for j in range(len(state_history[0])):
            if state_history[i][j] in s_counts:
                s_counts[state_history[i][j]] += 1
            else:
                s_counts[state_history[i][j]] = 1
            if state_history[i][j] in sr_counts:
                if reward_history[i] in sr_counts[state_history[i][j]]:
                    sr_counts[state_history[i][j]][reward_history[i]] += 1
                else:
                    sr_counts[state_history[i][j]][reward_history[i]] = 1
            else:
                sr_counts[state_history[i][j]] = {reward_history[i]: 1}
    return s_counts, sr_counts


def update_R(R, s_counts, sr_counts):
    for s in s_counts:
        if s in sr_counts:
            R[s] = 0
            for r in sr_counts[s]:
                R[s] += float(r) * sr_counts[s][r] / s_counts[s]
    return R


def update_V(V, P, R):
    """updates V using value iteration
    
    Return:
        num_iter: number of iterations that V took to converge
        V: np array with shape (NUM_STATES)
    """
    converged = False
    num_iter = 0
    while not converged:
        V_prev = np.copy(V)
        # print(np.amax(np.matmul(P, V), axis=1))
        V = R + GAMMA * np.amax(np.matmul(P, V), axis=1)
        num_iter += 1
        # print(np.amax(V - V_prev))
        # print(V)
        # print(V - V_prev)
        if np.amax(V - V_prev) < TOLERANCE:
            converged = True
    # print(V-V_prev)
    print(num_iter)
    # print(np.amax(V - V_prev))
    # print(num_iter)
    return num_iter, V


def train(display=False):
    # initialize model parameters
    P_size = NUM_STATES * NUM_ACTIONS * NUM_STATES

    sa_counts = {}
    sas_counts = {}

    # Generate Random Sum to 1
    P = np.random.uniform(9, 10, P_size)
    P_sum = np.sum(P)
    P /= P_sum
    P = np.reshape(P, (NUM_STATES, NUM_ACTIONS, NUM_STATES))

    s_counts = {}
    sr_counts = {}
    R = np.full(NUM_STATES, 0.1, dtype=float)
    # V_old = np.random.rand(NUM_STATES)
    # _, V = update_V(V_old / 10, P, R)
    _, V = update_V(np.full(NUM_STATES, 0.1, dtype=float), P, R)

    policy = get_policy(P, V)

    # initialize state
    state = Environment()  # max_cars
    time = 0
    state_history = []  # list of length-max_cars integer lists
    action_history = []  # list of length-max_cars integer lists
    reward_history = []

    num_iters = [] # TODO temp

    # train model - each iteration steps through 1 time step
    consecutive_no_learning_trials = 0
    num_updates = 0 # TODO temp
    while consecutive_no_learning_trials < NO_LEARNING_THRESHOLD:
        num_updates += 1 # TODO temp

        '''
        if num_updates % 10000 == 0: # 100
            sa_counts = {}
            sas_counts = {}
            s_counts = {}
            sr_counts = {}
            state_history = []  # list of length-max_cars integer lists
            action_history = []  # list of length-max_cars integer lists
            reward_history = []
        '''

        # print(counter) # TODO temp
        # simulate and get new state
        state_history.append([state_to_index(s) for s in get_states(state.cars, ENVS["FOUR_WAY_1"])])
        acc_indices = get_actions(state.cars, policy)
        action_history.append(acc_indices)
        time += TIME_STEP
        curr_R = state.update([get_action(i) for i in acc_indices], TIME_STEP)
        reward_history.append(curr_R)

        # Recompute MDP model and reset environment when crash occurs or all
        # cars reach finish line
        if curr_R < 0 or curr_R >= 100 or time > MAX_TIME:
            # print(num_updates)
            # counter = 0 # TODO temp
            if display:
                print("success" if curr_R >= 100 else "fail", "; time:", time)

            # update P
            sa_counts, sas_counts = update_sa_counts(sa_counts, sas_counts, state_history, action_history)
            P = update_P(P, sa_counts, sas_counts)

            # update R
            s_counts, sr_counts = update_sr_counts(s_counts, sr_counts, state_history, reward_history)
            R = update_R(R, s_counts, sr_counts)

            # update V via value iteration
            num_iter, V = update_V(V, P, R)
            num_iters.append(num_iter)

            # update policy
            policy = get_policy(P, V)
            # print(policy)

            # reset state
            state = Environment()  # max_cars
            time = 0
            state_history = []
            action_history = []
            reward_history = []
            cars_safe = True

            if num_iter == 1:
                consecutive_no_learning_trials += 1

    return P, V, policy


if __name__ == "__main__":
    P, V, policy = train(True)
    # print("policy updated")
    # print(policy)
    # acc_policy = [get_acc(p) for p in policy]
    np.save('policy.npy', policy)
