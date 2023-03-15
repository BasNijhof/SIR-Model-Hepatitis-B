import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total population
N = 1000

# Initial number of infected (I0) and recovered (R0) individuals
I0, R0 = 1, 0

# Initial number of susceptible individuals
S0 = N - I0 - R0

# Infectiousness, c, and mean recovery rate, d (in 1/days)
c, d = 0.2, 1. / 100

# t is a grid of time points (in days)
t = np.linspace(0, 365, num=365)


def get_derivatives(y, t, N, c, d):
    S, I, R = y
    dS = -c * S * I / N
    dI = c * S * I / N - d * I
    dR = d * I
    return dS, dI, dR


if __name__ == '__main__':
    # Initial conditions vector
    y0 = S0, I0, R0

    # Create the model from the SIR equations
    ret = odeint(get_derivatives, y0, t, args=(N, c, d))
    S, I, R = ret.T

    # Plot the model
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(t, S, 'b', label='Susceptible')
    ax.plot(t, I, 'r', label='Infected')
    ax.plot(t, R, 'g', label='Recovered')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Number of people')
    ax.set_ylim(0, N * 1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()
