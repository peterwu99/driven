'''
Finite-State MDP Implementation for TransportAI

The state of each car is represented by (p1, p2, p, v1, v2), where each
variable provides the following information:
    p1: distance to car behind
    p2: distance to car ahead
    p: distance to its wall
    v1: relative speed with respect to car behind
    v2: relative speed with respect to car ahead
'''

import math
import numpy as np
import os

from constants import CAR_LENGTH, NUM_CARS, TIME_STEP
from env import Car, Environment

MAX_ACC = 8   # max acceleration
MAX_DEC = -16 # max deceleration
NUM_ACTIONS = 4
ACC_DIFF = (MAX_ACC - MAX_DEC) / (NUM_ACTIONS-1)

NUM_POS_STATE = 2 # number of possible values for position info; note: if these
NUM_VEL_STATE = 3 # values are changed, should also change get_.*{2} functions
NUM_STATE_I = [NUM_POS_STATE, NUM_POS_STATE, NUM_POS_STATE, NUM_VEL_STATE, NUM_VEL_STATE]
NUM_STATES = int(math.pow(NUM_POS_STATE, 3)*math.pow(NUM_VEL_STATE, 2))

MAX_TIME = 300
GAMMA = 0.995
TOLERANCE = 0.01 # max change in value function to declare convergence
NO_LEARNING_THRESHOLD = 20 # max number of no learning instances before
                           # declaring model has converged

def load_policy(file_name='policy.npy'):
    '''Returns policy

    If 'policy.npy' doesn't exist, trains policy, saves it to 'policy.npy',
    and returns it.

    Return:
        a size-NUM_STATES numpy array
    '''
    if not os.path.exists(file_name):
        raise ValueError("No policy.npy found")

    return np.load(file_name)


def get_accel_list(policy, cars):
    '''Returns accelerations for each of the given cars, based on given policy

    Args:
        policy: size-NUM_STATES numpy array of accelerations, i.e. 'policy.npy'
        cars: list of cars
    '''
    return [get_acc(i) for i in get_actions(cars, policy)]

def get_acc(index):
    return MAX_ACC-index*ACC_DIFF

def get_pc(dist1, dist2):
    return 1 if dist2-dist1-CAR_LENGTH > 3 else 0

def get_pw(car_pos, wall_pos):
    return 1 if car_pos-wall_pos-CAR_LENGTH/2 > 3 else 0

def get_v(v_car, v_other):
    diff = v_car - v_other
    if diff > 2: return 2
    elif diff >= -2: return 1
    else: return 0

def get_states(cars):
    '''
    
    Assuming that there are at least 2 cars
    
    Return:
        length-len(cars) list of size-5 tuple states 
    '''
    data = [(c.position, c.wall_position, c.velocity) for c in cars]
    data.sort()
    car_states = [(0,0,0,0,0) for _ in cars]
    car_states[0] = (NUM_POS_STATE-1, get_pc(data[1][0],data[0][0]), \
                     get_pw(data[0][0],data[0][1]), NUM_VEL_STATE-1, \
                     get_v(data[0][2],data[1][2]))
    for i in range(1,len(car_states)-1):
        car_states[i] = (get_pc(data[i-1][0],data[i][0]), \
                         get_pc(data[i][0],data[i+1][0]), \
                         get_pw(data[i][0],data[i][1]), \
                         get_v(data[i][2],data[i-1][2]), \
                         get_v(data[i][2],data[i+1][2]))
    car_states[-1] = (get_pc(data[-2][0],data[-1][0]), NUM_POS_STATE-1, \
                      get_pw(data[-1][0],data[-1][1]), \
                      get_v(data[-1][2],data[-2][2]), 0)
    return car_states

def get_actions(cars, policy):
    '''Computes actions given car states and policy
    
    Assuming that there are at least 2 cars
    
    Args:
        cars: list of Cars
        policy: 1-dim numpy array with length NUM_STATES
    
    Return:
        list of indices (index can be turned into acceleration value via get_acc)
    '''
    state_indices = [state_to_index(s) for s in get_states(cars)]
    return [policy[i] for i in state_indices]

def top_state(states):
    '''Returns state, probability

    Args:
        states: dictionary with states as keys and probabilities as values
    '''
    max_s = None
    max_p = -1
    for s in states:
        if states[s] > max_p:
            max_p = states[s]
            max_s = s
    return max_s, max_p

def index_to_state(index):
    '''maps integer in range [0, NUM_STATES) to size-5 tuple state'''
    s = (0,0,0,0,0)
    for i in reversed(range(5)):
        s[i] = index%(NUM_STATE_I[i])
        index /= (NUM_STATE_I[i])
    return s

def state_to_index(s):
    '''maps size-5 tuple state to integer in range [0, NUM_STATES)'''
    index = 1
    multiplier = 1
    for i, si in reversed(list(enumerate(s))):
        index += si*multiplier
        multiplier *= NUM_STATE_I[i]
    return index

def get_policy(P, V):
    '''Returns optimal policy based on P and V
    
    Args:
        P: transition probability matrix,
            np array with shape (NUM_STATES, NUM_ACTIONS, NUM_STATES)
        V: value function, np array with shape (NUM_STATES)
    Return:
        np array of ints in the range [0, NUM_ACTIONS) with shape (NUM_STATES)
    '''
    return np.argmax(np.matmul(P, V), axis=1)

def update_sa_counts(sa_counts, sas_counts, state_history, action_history):
    for s,a in zip(state_history[0], action_history[0]):
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
            if (state_history[i-1][j], action_history[i-1][j]) in sas_counts:
                if state_history[i][j] in sas_counts[(state_history[i-1][j], action_history[i-1][j])]:
                    sas_counts[(state_history[i-1][j], action_history[i-1][j])][state_history[i][j]] += 1
                else:
                    sas_counts[(state_history[i-1][j], action_history[i-1][j])][state_history[i][j]] = 1
            else:
                sas_counts[(state_history[i-1][j], action_history[i-1][j])] = {state_history[i][j]: 1}
    return sa_counts, sas_counts

def update_P(P, sa_counts, sas_counts):
    '''Updates transition probability matrix P given historical data

    Args:
        P: transition probability matrix,
            np array with shape (NUM_STATES, NUM_ACTIONS, NUM_STATES)
        state_history: list of length-num_cars integer lists, where ints are in
                       the range [0, NUM_STATES)
        action_history: list of length-num_cars integer lists, where ints are in
                        the range [0, NUM_ACTIONS)
    
    Return:
        new transition probability matrix P,
            np array with shape (NUM_STATES, NUM_ACTIONS, NUM_STATES)
    '''
    for (s, a) in sa_counts:
        if (s,a) in sas_counts:
            for s2 in sas_counts[(s,a)]:
                P[s][a][s2] = sas_counts[(s,a)][s2]/sa_counts[(s,a)]
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
                R[s] += float(r)*sr_counts[s][r]/s_counts[s]
    return R

def update_V(V, P, R):
    '''updates V using value iteration
    
    Return:
        num_iter: number of iterations that V took to converge
        V: np array with shape (NUM_STATES)
    '''
    converged = False
    num_iter = 0
    while not converged:
        V_prev = np.copy(V)
        V = R + GAMMA * np.argmax(np.matmul(P, V), axis=1)
        num_iter += 1
        if np.amax(V - V_prev) < TOLERANCE:
            converged = True
    return num_iter, V

def train(max_cars=NUM_CARS, display=False):
    # initialize model parameters
    sa_counts = {}
    sas_counts = {}
    P = np.ones((NUM_STATES, NUM_ACTIONS, NUM_STATES), dtype=float)/NUM_STATES
    s_counts = {}
    sr_counts = {}
    R = np.zeros(NUM_STATES, dtype=float)
    _, V = update_V(np.zeros(NUM_STATES, dtype=float), P, R)
    policy = get_policy(P, V)

    # initialize state
    state = Environment(max_cars)
    time = 0
    state_history = [] # list of length-max_cars integer lists
    action_history = [] # list of length-max_cars integer lists
    reward_history = []
    cars_safe = True

    # train model - each iteration steps through 1 time step
    consecutive_no_learning_trials = 0
    while consecutive_no_learning_trials < NO_LEARNING_THRESHOLD:
        # simulate and get new state
        state_history.append([state_to_index(s) for s in get_states(state.cars)])
        acc_indices = get_actions(state.cars, policy)
        action_history.append(acc_indices)
        time += TIME_STEP
        curr_R = state.update([get_acc(i) for i in acc_indices], TIME_STEP)    
            # -1 for crash, 0 for nothing crashed, 1 for car just passed
            # finish line,, 2 for all cars passed finish line
        reward_history.append(curr_R)
    
        # Recompute MDP model and reset environment when crash occurs or all
        # cars reach finish line
        if curr_R == -1 or curr_R == 2 or time > MAX_TIME:
            if display:
                print("success" if curr_R == 2 else "fail", "; time:",time)

            # update P
            sa_counts, sas_counts = update_sa_counts(sa_counts, sas_counts, state_history, action_history)
            P = update_P(P, sa_counts, sas_counts)

            # update R
            s_counts, sr_counts = update_sr_counts(s_counts, sr_counts, state_history, reward_history)
            R = update_R(R, s_counts, sr_counts)
                
            # update V via value iteration
            num_iter, V = update_V(V, P, R)

            # update policy
            policy = get_policy(P, V)

            # reset state
            state = Environment(max_cars)
            time = 0
            state_history = []
            action_history = []
            reward_history = []
            cars_safe = True
        
            if num_iter == 1:
                consecutive_no_learning_trials += 1

    acc_policy = [get_acc(p) for p in policy]
    np.save('policy.npy', policy)

    return P, V, policy

if __name__ == "__main__":
    P, V, policy = train(NUM_CARS, True)
