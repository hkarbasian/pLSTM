# Pythion for Lorenz system
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


##########################################################################

## Lorenz system of equaitons
def vanderpol(u, t, s):

    [A, omega, mu] = s

    #u = u.T
    x, y = u[0], u[1]

    dxdt = y
    dydt = mu * (1 - x**2) * y - x + A * np.sin(omega * t)


    return np.array([dxdt, dydt]).T


## Runge-Kutta 4th order
def rk44(f, u, t, s, dt):

    k1 = f(u, t, s)
    k2 = f(u + 0.5 * dt * k1, t, s)
    k3 = f(u + 0.5 * dt * k2, t, s)
    k4 = f(u + 1.0 * dt * k3, t, s)	

    return u + (k1 + 2.0 * k2 + 2.0 * k3 + k4)*(dt/6.0)


##########################################################################


dir = '/root directory/data'

dt=0.05
m0 = 5000
mt = 8 * 50
mskip = 3
n_sample = 20

# generate samples
mu = np.round( np.random.uniform(low=0, high=3, size=(n_sample,)), 2)
Amp = np.round( np.random.uniform(low=0, high=1, size=(n_sample,)), 2)
Omega = np.round( np.random.uniform(low=0, high=1, size=(n_sample,)), 2)

# create para.txt file
with open(dir + '/para.txt', 'w') as fh:
    comment = 'ID, m1, m2, nsnap, parameters (Amp, omega, mu)\n'
    fh.write(comment)


# loop over samples
for j in range(n_sample): 

    # design parameters
    s = [Amp[j], Omega[j], mu[j]]

    # add case to para.txt file
    with open(dir + '/para.txt', 'a') as fh:
        comment = 'vanderpol{}, 0, {}, {}, {}, {}, {}\n'.format(j, mt-1, mt, s[0], s[1], s[2])
        fh.write(comment)

    X = np.zeros((2, mt))
    force = np.zeros(mt)
    u0= np.random.rand(2)
    c = 0
    for i in range(m0 + mskip * mt):
        t = i * dt
        u0 = rk44(vanderpol, u0, t, s, dt)
        if i >= m0 and i%mskip == 0:
            X[:, c] = u0
            force[c] = Amp[j] * np.sin(Omega[j] * t)
            c += 1

    np.savetxt("{}/q_C{}_{}.txt".format(dir, j, "R1"), X, delimiter=',')
    np.savetxt("{}/f_C{}_{}.txt".format(dir, j, "R1"), force, delimiter=',')		
    print(f"# {j} - case {s}")

    # do some plotting
    plt.clf()
    plt.plot(X[0, :],X[1, :], '-k', alpha=1)
    plt.savefig("{}/q_C{}_{}.png".format(dir, j, "R1"))
    plt.close()








