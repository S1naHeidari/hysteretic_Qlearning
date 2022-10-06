import numpy as np
from helper import *


def create_table():
    dict = {}
    discretePos = np.round(list(np.linspace(-1, 1, 100)), decimals=3)
    discreteV = np.round(list(np.linspace(-3, 3, 50)), decimals=3)
    actions = np.round(list(np.linspace(-1, 1, 15)), decimals=3)
   
    for p in discretePos:
        for v in discreteV:
            dict[(p, v)] = {}
            for a in actions:
                dict[(p, v)][a] = 0.0
    
    return dict


def reward(x, v):
    return 0.8 * np.exp(-(np.power(x, 2) / np.power(0.25, 2))) + 0.2 * np.exp(-(np.power(v, 2) / np.power(0.25, 2)))


def dynamic(h1, h2, v):
    m = 0.5
    g = 9.8
    l = 2
    c = 0.01
    ball_acceleration = ((-c * v) + (m * g * ((h1 - h2) / l))) / m
    return ball_acceleration


def nextPosition(a, t, v, x_0):
    
    return 1 / 2 * a * t * t + v * t + x_0


def nextSpeed(a, t, v_0):
    return a * t + v_0


def nextState(h1, h2, v, t, x_0, v_0):
    a = dynamic(h1, h2, v)
    nextX = nextPosition(a, t, v, x_0)
    nextS = nextSpeed(a, t, v_0)
    return nextX, nextS


def nextAction(states, actions, qTables, trial, numOfEps=40,trials=5000):


    if trial is not None and numOfEps > 0:
        epsilons = np.linspace(0.8, 0.1, numOfEps)
        index = int(trial // (trials / numOfEps))
        eps = epsilons[index]
    else:
        eps = 0.1

    numberOfAgents = len(qTables)
    new_actions = [0] * numberOfAgents
    for q in range(len(qTables)):
        if actions is not None and np.random.uniform() < eps:
            action = np.random.choice(actions)
        else:
            action = getKey(qTables[q][states], max(qTables[q][states].values()))
        new_actions[q] = action
    return new_actions


def isValidState(states):
    discretePos = np.round(list(np.linspace(-1, 1, 100)), decimals=3)
    position_space = 1 / 50
    position_index = np.abs(int(np.round(states[0] / position_space, decimals=0)))
    if states[0] > 0:
        position_index += 50
    else:
        position_index = 50 - position_index

    if position_index != 100 and position_index != 0:
        if position_index == 99:
            possible_positions = [discretePos[position_index - 1], discretePos[position_index]]
        else:
            possible_positions = [discretePos[position_index - 1], discretePos[position_index],
                                  discretePos[position_index + 1]]
    elif position_index == 100:
        possible_positions = [discretePos[position_index - 1], discretePos[position_index - 2]]
    else:
        possible_positions = [discretePos[position_index], discretePos[position_index + 1]]




    discreteV = np.round(list(np.linspace(-3, 3, 50)), decimals=3)
    velocity_space = 3 / 25
    velocity_index = np.abs(int(np.round(states[1] / velocity_space, decimals=0)))
    
    if states[1] > 0:
        velocity_index += 25
    else:
        velocity_index = 25 - velocity_index


    if velocity_index != 50 and velocity_index != 0:
        if velocity_index == 49:
            possible_velocities = [discreteV[velocity_index - 1], discreteV[velocity_index]]
        else:
            possible_velocities = [discreteV[velocity_index - 1], discreteV[velocity_index],
                                   discreteV[velocity_index + 1]]
    elif velocity_index == 50:
        possible_velocities = [discreteV[velocity_index - 1], discreteV[velocity_index - 2]]
    else:
        possible_velocities = [discreteV[velocity_index], discreteV[velocity_index + 1]]



    new_states = (min(possible_positions, key=lambda x: abs(x - states[0])),
                  min(possible_velocities, key=lambda x: abs(x - states[1])))
    
    return new_states


my_dict = create_table()
