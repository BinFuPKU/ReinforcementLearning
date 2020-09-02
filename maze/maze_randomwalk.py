
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def init_plot():
    # basic
    fig = plt.figure(figsize=(5,5))
    ax = plt.gca()

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

# probability: average
def simple_convert_into_pi_from_theta(theta):
    m,n = theta.shape
    pi = np.zeros((m,n))
    for i in range(m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])
    pi = np.nan_to_num(pi)
    return pi

def get_next_state(pi, state):
    direction = ['up', 'right', 'down', 'left']

    next_direction = np.random.choice(direction, p=pi[state, :])
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

def goal_maze(pi):
    s = 0
    state_history = []
    while 1:
        next_state = get_next_state(pi, s)
        state_history.append(next_state)

        if next_state==8:
            break
        else:
            s = next_state
    return state_history

def plot_history(fig, plt, line, step_text, state_history):
    from matplotlib import animation
    from IPython.display import HTML

    def init():
        line.set_data([],[])
        step_text.set_text('')
        return (line, step_text)

    def animate(i):
        state = state_history[i]
        x = (state % 3) + 0.5
        y = 2.5 - int(state/3)
        line.set_data(x, y)
        step_text.set_text('step=%d' %(i+1))
        return (line, step_text)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(state_history),
                                   interval=1000, repeat=False)
    anim.save('maze_randomwalk.gif', writer='pillow') # imagemagick
    # HTML(anim.to_jshtml())
    plt.show()

if __name__ == '__main__':
    fig, plt, line, step_text = init_plot()

    pi_0 = simple_convert_into_pi_from_theta(theta_0)
    # print(pi_0)
    state_history = goal_maze(pi_0)
    print(state_history)
    plot_history(fig, plt, line, step_text, state_history)
