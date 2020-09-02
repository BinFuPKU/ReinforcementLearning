
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def init_plot():
    # basic
    fig = plt.figure(figsize=(5,5))
    ax = plt.gca()
    plt.title("maze_QLearning")

    # plot red line - 'stop'
    plt.plot([1,1],[0,1], color='red', linewidth=2)
    plt.plot([1,2],[2,2], color='red', linewidth=2)
    plt.plot([2,2],[2,1], color='red', linewidth=2)
    plt.plot([2,3],[1,1], color='red', linewidth=2)

    # textual hints
    plt.text(0.5, 2.5, 'S0', size=14, ha='center')
    plt.text(1.5, 2.5, 'S1', size=14, ha='center')
    plt.text(2.5, 2.5, 'S2', size=14, ha='center')
    plt.text(0.5, 1.5, 'S3', size=14, ha='center')
    plt.text(1.5, 1.5, 'S4', size=14, ha='center')
    plt.text(2.5, 1.5, 'S5', size=14, ha='center')
    plt.text(0.5, 0.5, 'S6', size=14, ha='center')
    plt.text(1.5, 0.5, 'S7', size=14, ha='center')
    plt.text(2.5, 0.5, 'S8', size=14, ha='center')
    plt.text(0.5, 2.3, 'START', size=12, ha='center')
    plt.text(2.5, 0.3, 'GOAL', size=12, ha='center')

    # range
    ax.set_xlim(0,3)
    ax.set_ylim(0,3)

    # label
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off',
                    right='off', left='off', labelleft='off')
    # green ball
    line, = ax.plot([0.5], [2.45], marker='o', color='g', markersize=60)
    step_text = ax.text(0.02, 0.02, '', size=10, transform=ax.transAxes)
    # plt.show()

    return fig, plt, line, step_text, ax

# policy: action direction (up, right, down, left)
theta_0 = np.array([[np.nan, 1, 1, np.nan],
                    [np.nan, 1, np.nan, 1],
                    [np.nan, np.nan, 1, 1],
                    [1, 1, 1, np.nan],
                    [np.nan, np.nan, 1, 1],
                    [1, np.nan, np.nan, np.nan],
                    [1, np.nan, np.nan, np.nan],
                    [1, 1, np.nan, np.nan]])

# probability: average
def simple_convert_into_pi_from_theta(theta):
    m,n = theta.shape
    pi = np.zeros((m,n))
    for i in range(m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])
    pi = np.nan_to_num(pi)
    return pi

def get_action(state, Q, epsilon, pi):
    direction = ['up', 'right', 'down', 'left']

    if np.random.rand() < epsilon:
        next_direction = np.random.choice(direction, p=pi[state, :])
    else:
        next_direction = direction[np.nanargmax(Q[state, :])]
    action = None
    if next_direction=='up':
        action = 0
    elif next_direction=='right':
        action = 1
    elif next_direction=='down':
        action = 2
    elif next_direction == 'left':
        action = 3
    return action

def get_next_state(state, action):
    direction = ['up', 'right', 'down', 'left']
    next_direction = direction[action]
    next_state = state
    if next_direction=='up':
        next_state -= 3
    elif next_direction=='right':
        next_state +=1
    elif next_direction=='down':
        next_state += 3
    elif next_direction == 'left':
        next_state -=1
    return next_state

# TD error
def QLearning(state, action, reward, next_state, Q, eta, gamma):
    if next_state==8:
        Q[state, action] = Q[state, action] + eta * (reward - Q[state, action])
    else:
        Q[state, action] = Q[state, action] + eta * \
                           (reward + gamma * np.nanmax(Q[next_state, :]) - Q[state, action])
    return Q

def goal_maze(Q, epsilon, eta, gamma, pi):
    state = 0
    action = next_action = get_action(state, Q, epsilon, pi)
    state_action_history = [[0, np.nan]]
    while 1:
        action = next_action
        state_action_history[-1][1] = action

        next_state = get_next_state(state, action)
        state_action_history.append([next_state, np.nan])

        if next_state==8:
            reward = 1.
            next_action = np.nan
        else:
            reward = 0.
            next_action = get_action(next_state, Q, epsilon, pi)
        Q = QLearning(state, action, reward, next_state, Q, eta, gamma)
        if next_state==8:
            break
        else:
            state = next_state
    return state_action_history, Q

def value_strategy(pi_0):
    eta = 0.1
    gamma = 0.9
    epsilon = 0.5

    m, n = theta_0.shape
    Q = np.random.rand(m, n) * theta_0 * 0.1

    v = np.nanmax(Q, axis=1)
    episode = 1

    V = []
    V.append(np.nanmax(Q, axis=1))

    while 1:
        epsilon /= 2.
        action_state_history, Q = goal_maze(Q, epsilon, eta, gamma, pi_0)
        new_v = np.nanmax(Q, axis=1)
        print('iter=', episode, 'changeValue=', np.sum(np.abs(new_v-v)))
        v = new_v
        V.append(v)
        episode+=1
        if episode>100:
            break
    return Q, V, epsilon, eta, gamma

def plot_history(fig, plt, line, step_text, ax, state_action_history, V):
    from matplotlib import animation
    from IPython.display import HTML

    def init():
        line.set_data([],[])
        step_text.set_text('')
        return (line, step_text)

    def animate(i):
        step_text.set_text('step=%d/%d' %(i+1, len(state_action_history)))

        # state, action = state_action_history[i]
        # x = (state % 3) + 0.5
        # y = 2.5 - int(state/3.)
        # line.set_data(x, y)

        line, = ax.plot([0.5], [2.5], marker='s', color=cm.jet(V[i][0]), markersize=85)
        line, = ax.plot([1.5], [2.5], marker='s', color=cm.jet(V[i][1]), markersize=85)
        line, = ax.plot([2.5], [2.5], marker='s', color=cm.jet(V[i][2]), markersize=85)
        line, = ax.plot([0.5], [1.5], marker='s', color=cm.jet(V[i][3]), markersize=85)
        line, = ax.plot([1.5], [1.5], marker='s', color=cm.jet(V[i][4]), markersize=85)
        line, = ax.plot([2.5], [1.5], marker='s', color=cm.jet(V[i][5]), markersize=85)
        line, = ax.plot([0.5], [0.5], marker='s', color=cm.jet(V[i][6]), markersize=85)
        line, = ax.plot([1.5], [0.5], marker='s', color=cm.jet(V[i][7]), markersize=85)
        line, = ax.plot([2.5], [0.5], marker='s', color=cm.jet(1.0), markersize=85)

        return (line, step_text, ax)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(state_action_history),
                                   interval=1000, repeat=False)
    anim.save('maze_QLearning_heat.gif', writer='pillow') # imagemagick
    # HTML(anim.to_jshtml())
    plt.show()

if __name__ == '__main__':
    fig, plt, line, step_text, ax = init_plot()

    pi_0 = simple_convert_into_pi_from_theta(theta_0)
    Q, V, epsilon, eta, gamma = value_strategy(pi_0)
    state_action_history, Q = goal_maze(Q, epsilon, eta, gamma, pi_0)
    print('test:', state_action_history)
    plot_history(fig, plt, line, step_text, ax, state_action_history, V)
