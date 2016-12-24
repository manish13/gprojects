import pandas as pd
import numpy as np
import matplotlib.pylab as mpl
import seaborn as sb
import scipy.io as sio
import numba


class HMM():
    def __init__(self, x_states, z_states, pi, A, B):
        self.x_states = x_states
        self.z_states = z_states
        self.pi = pi
        self.A = A
        self.B = B

    @numba.jit
    def normalize(self, u):
        Z = np.sum(u)
        return u/Z, Z

    @numba.jit
    def forward_algo(self, x):
        # calculates alpha and sum_Z
        zt = np.zeros((len(x)+1, len(self.z_states)))
        zt[0,:] = self.pi

        i = hmm.x_states.index(x[0])
        a, Z = self.normalize(self.A[:,i] * self.pi)
        zt[1,:] = a
        lz = np.log(Z)

        for t in np.arange(1,len(x)):
            i = hmm.x_states.index(x[t])
            a, Z = self.normalize(self.A[:,i] * np.dot(a,self.B.T))
            zt[t+1,:] = a
            lz += np.log(Z)
        return zt, lz

    @numba.jit
    def backward_algo(self, x):
        # calculates beta
        bt = np.zeros((len(x), len(self.z_states)))
        bt[-1,:] = 1

        for t in np.arange(len(x)-2,-1,-1):
            i = hmm.x_states.index(x[t+1])
            bt[t,:] = np.dot(self.B.T, self.A[:,i]*bt[t+1,:] )

        return bt

    @numba.jit
    def viterbi_algo(self, x):
        lp = np.log(self.pi)
        lA = np.log(self.A)
        lB = np.log(self.B)

        d = np.zeros((len(x),len(self.z_states)))
        a = np.zeros((len(x),len(self.z_states)),dtype=int)

        # forward run
        i = hmm.x_states.index(x[0])
        d[0,:] = lp+lA[:,i]
        for t in np.arange(1,len(x)):
            i = hmm.x_states.index(x[t])
            da = d[t-1,:]+lB
            d[t,:] = np.max(da,1)+lA[:,i]
            a[t,:] = np.argmax(da,1)

        # backward run
        mpz = np.zeros(len(x))
        mpz[-1] = np.argmax(d[-1,:])
        for t in np.arange(len(x)-2,-1,-1):
            mpz[t] =  a[t+1,mpz[t+1]]

        return mpz

if __name__=='__main__':
    # comes with the arrray of x and returns z and nc

    dirn = '/media/manish/Data/D/COURSES/Murphy/'
    X = sio.loadmat(dirn+'casinoData.mat')

    x = X['C'][0]

    x_states = [1,2,3,4,5,6]
    z_states = ['L','F']
    pi = np.array([0.5, 0.5])
    A = np.array([[1/6,1/6,1/6,1/6,1/6,1/6],
                [1/10,1/10,1/10,1/10,1/10,5/10]])
    B = np.array([[0.95,0.05],[0.1,0.9]])

    # x_states = [1,2,3]
    # z_states = [1,2]
    # pi = np.array([0.5,0.5])
    # A = np.array([[0.8, 0.1, 0.1],[0.1, 0.1, 0.8]])
    # B = np.array([[0.5, 0.2],[0.5, 0.8]])
    # x = np.array([1,2,3])

    hmm = HMM(x_states, z_states, pi, A, B)
    a, nc = hmm.forward_algo(x)
    b = hmm.backward_algo(x)
    z = hmm.viterbi_algo(x)

    print(a)
    print(b)
    print(z)

    #z = pd.DataFrame(a,columns=z_states)
    #ax = z['F'].plot(style='k-',lw=3)
    #pd.Series(x).plot(ax=ax, secondary_y=True, lw=0.5)


