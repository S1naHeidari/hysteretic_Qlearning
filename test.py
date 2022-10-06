from functions import *
import matplotlib.pyplot as plt
import pickle as pkl


def test(qTables):
    spaces, velocities = [], []
    states = (0.495, 1.041)  
    spaces.append(states[0])
    velocities.append(states[1])

    for i in range(30):  
        actions = nextAction(states, None, qTables, None)
        x, v = nextState(h1=actions[0], h2=actions[1], v=states[1], t=0.03, x_0=states[0], v_0=states[1])
        spaces.append(x)
        velocities.append(v)
        # if the ball falls from the flat surface
        if np.abs(x) > 1: 
            break  
        new_states = (np.round(x, decimals=3), np.round(v, decimals=3))
        new_states = isValidState(new_states)
        states = new_states
    
    plt.plot(spaces, '-', label="Space", color='black')
    plt.plot(velocities, '-', label="Speed", color='red')
    plt.legend()
    plt.title("Hysteretic")
    plt.savefig('./plots/' + "Hysteretic" + '_test.png')
    plt.clf()


def main():
    with open('./tables/q-table1.p', 'rb') as file:
        qTable1 = pkl.load(file)
    with open('./tables/q-table2.p', 'rb') as file:
        qTable2 = pkl.load(file)
    qTables = [qTable1, qTable2]
    test(qTables)


if __name__ == '__main__':
    main()


