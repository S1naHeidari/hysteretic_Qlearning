from hysteretic import *
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
import pickle as pkl

alpha = 0.9
beta = 0.1
gamma = 0.9
samplingTime = 0.03
decimals = 3
trials = 5000
actions = np.round(np.linspace(-1, 1, 15), decimals=decimals)




def trainHysteretic():
    # create q-Table
    qTable1 = create_table()
    qTable2 = create_table()
    qTables = [qTable1, qTable2]

    iterationRewards = []
    for i in range(5):
        rewardSumInTrial = []
        for trial in range(trials):
            progress(trial, trials, prefix='Iteration: ' + str(i))
            # Initialize states (x,xbar)
            states = (0.495, 1.041)  
            rewardSum = 0

            for t in np.arange(0, 20, samplingTime):

                new_actions = nextAction(states, actions, qTables, trial, numOfEps=40, trials=trials)

                x, v = nextState(h1=new_actions[0], h2=new_actions[1], v=states[1], t=samplingTime, x_0=states[0],
                                     v_0=states[1])

                # deal with velocities more than 3 and less than -3
                if v > 3: v = 3
                if v < -3: v = -3

                if np.abs(x) > 1: break

                thisReward = reward(states[0], states[1])
                rewardSum = rewardSum + thisReward

                new_states = (np.round(x, decimals=decimals), np.round(v, decimals=decimals))
                # check if the new states are in discretized states
                new_states = isValidState(new_states)

                qTables = hysteretic(qTables, states, new_actions, alpha, beta, thisReward, gamma, new_states)
                states = new_states

            rewardSumInTrial.append(rewardSum)
        
        iterationRewards.append(rewardSumInTrial)
    mean_output = np.mean(iterationRewards, axis=0)
    plt.plot(list(range(trials)), mean_output, '-', color="black")
    plt.title('Hysteretic')
    plt.savefig('./plots/Hysteretic.png')
    plt.clf()

    pkl.dump(qTables[0], open('tables/q-table1.p', 'wb'))
    pkl.dump(qTables[1], open('tables/q-table2.p', 'wb'))

def main():
    trainHysteretic()

if __name__ == '__main__':
    main()



