
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def init_plot():
    # basic
    fig = plt.figure(figsize=(5,5))
    ax = plt.gca()
    plt.title("maze_policygradient")

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

    return fig, plt, line, step_text

# policy: action direction (up, right, down, left)
theta_0 = np.array([[np.nan, 1, 1, np.nan],
                    [np.nan, 1, np.nan, 1],
                    [np.nan, np.nan, 1, 1],
                    [1, 1, 1, np.nan],
                    [np.nan, np.nan, 1, 1],
                    [1, np.nan, np.nan, np.nan],
                    [1, np.nan, np.nan, np.nan],
                    [1, 1, np.nan, np.nan]])

# probability: softmax
def softmax_convert_into_pi_from_theta(theta):
    beta=1.0
    m,n = theta.shape
    pi = np.zeros((m,n))
    exp_theta = np.exp(beta * theta)
    for i in range(m):
        pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])
    pi = np.nan_to_num(pi)
    return pi

def get_action_next_state(pi, state):
    direction = ['up', 'right', 'down', 'left']

    next_direction = np.random.choice(direction, p=pi[state, :])
    next_state = state
    action = None
    if next_direction=='up':
        action = 0
        next_state -= 3
    elif next_direction=='right':
        action = 1
        next_state +=1
    elif next_direction=='down':
        action = 2
        next_state += 3
    elif next_direction == 'left':
        action = 3
        next_state -=1
    return action, next_state

def goal_maze(pi):
    s = 0
    state_action_history = []
    while 1:
        action, next_state = get_action_next_state(pi, s)
        state_action_history.append([next_state, action])

        if next_state==8:
            break
        else:
            s = next_state
    return state_action_history

# policy gradient
def update_theta(theta, pi, state_action_history):
    eta = 0.1
    T = len(state_action_history)-1

    m, n = theta.shape
    delta_theta = theta.copy()

    # policy gradient: theta(s,a) = theta(s_old,a_old) + eta * delta_theta(s,a)
    # delta_theta(s,a) = (N(s,a) - pi(a|s) * N(s))/T
    for i in range(m):
        for j in range(n):
            if np.isnan(theta[i,j]):
                continue
            N_s = sum([1 for s,a in state_action_history if s==i])
            N_s_a = sum([1 for s,a in state_action_history if s==i and a==j])

            delta_theta[i,j] = (N_s - N_s_a * pi[i,j])/T
    new_theta = theta + eta * delta_theta
    return new_theta

# reinforcement learning: policy
def policy_gradient():
    epsilon = 0.00001

    theta = theta_0
    pi = softmax_convert_into_pi_from_theta(theta_0)

    while 1:
        state_action_history = goal_maze(pi)
        new_theta = update_theta(theta, pi, state_action_history)
        new_pi = softmax_convert_into_pi_from_theta(new_theta)

        change = np.sum(np.abs(new_pi-pi))
        print('change=', change)

        if change<epsilon:
            break
        else:
            theta = new_theta
            pi = new_pi
    print('final pi=', pi)
    return theta, pi

def plot_history(fig, plt, line, step_text, state_action_history):
    from matplotlib import animation
    from IPython.display import HTML

    def init():
        line.set_data([],[])
        step_text.set_text('')
        return (line, step_text)

    def animate(i):
        state, action = state_action_history[i]
        x = (state % 3) + 0.5
        y = 2.5 - int(state/3)
        line.set_data(x, y)
        step_text.set_text('step=%d/%d' %(i+1, len(state_action_history)))
        return (line, step_text)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(state_action_history),
                                   interval=1000, repeat=False)
    anim.save('maze_policygradient.gif', writer='pillow')
    # HTML(anim.to_jshtml())
    plt.show()

if __name__ == '__main__':
    fig, plt, line, step_text = init_plot()

    theta, pi = policy_gradient()
    state_action_history = goal_maze(pi)
    print('test:', state_action_history)
    plot_history(fig, plt, line, step_text, state_action_history)
