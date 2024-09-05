# Pythion for Lorenz system
import numpy as np
import matplotlib.pyplot as plt


##########################################################################

## Lorenz system of equaitons
def lorenz(u, rho, sigma=10.0, beta=8.0/3.0):

    u = u.T
    x, y, z = u[0], u[1], u[2]

    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    return np.array([dxdt, dydt, dzdt]).T


## Runge-Kutta 4th order
def rk44(f, u, s, dt):

    k1 = f(u, s)
    k2 = f(u + 0.5 * dt * k1, s)
    k3 = f(u + 0.5 * dt * k2, s)
    k4 = f(u + 1.0 * dt * k3, s)	

    return u + (k1 + 2.0 * k2 + 2.0 * k3 + k4)*(dt/6.0)


##########################################################################


dir = '/root directory/data'

dt=0.01
m0 = 4000
mt = 2000
mskip = 5
n_sample = 3

# generate samples
rho = np.round( np.random.uniform(low=35, high=45, size=(n_sample,)), 0)

# create para.txt file
with open(dir + '/para.txt', 'w') as fh:
    comment = 'ID, m1, m2, nsnap, parameters (rho)\n'
    fh.write(comment)

# loop over samples
for j, s  in enumerate(rho): 

    # add case to para.txt file
    with open(dir + '/para.txt', 'a') as fh:
        comment = 'lorenz{}, 0, {}, {}, {}\n'.format(j, mt-1, mt, s)
        fh.write(comment)

    X = np.zeros((3, mt))
    u0= np.random.rand(3)
    c = 0
    for i in range(m0 + mskip * mt):
        u0 = rk44(lorenz, u0, s, dt)
        if i >= m0 and i%mskip == 0:
            X[:, c] = u0
            c += 1

    np.savetxt("{}/q_C{}_{}.txt".format(dir, j, "R1"), X, delimiter=',')	
    print(f"# {j} - case {s}")



plt.plot(X[0, :],X[2, :], '-g')
plt.show()







