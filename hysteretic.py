


def hysteretic(qTables, states, actions, alpha, beta, r, gamma, new_states):
    
    for action, table in zip(actions, qTables):
        maximum = 0 if not table[new_states] else max(table[new_states].values())
        delta = r + gamma * maximum - table[states][action]
        if delta >= 0:
            table[states][action] += delta * alpha
        else:
            table[states][action] += delta * beta
    return qTables
