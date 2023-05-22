import pickle
import numpy as np
import ripser as rp
import persim
from persim import plot_diagrams
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Ori = np.load('./sample_new_3/sample_new/gen_5000.npy')
R_ori = rp.ripser(Ori, maxdim=2, coeff=2, distance_matrix=False, do_cocycles=False)
dgm1 = R_ori['dgms'][1]
barcodes = R_ori['dgms']
plt.figure(figsize=(8, 8), dpi=100)
plot_diagrams(barcodes, size=40, lifetime=False, show=False)
plt.title("Persistence Diagram")
plt.savefig('./plots/dgm_r_ori.png')
plt.show()
sim_bn = []
sim_w = []
x = []

for i in range(0, 5000):
    x.append(i)
    X = np.load('./sample_new_3/sample_new/gen_%d.npy' % i)
    R = rp.ripser(X, maxdim=2, coeff=2, distance_matrix=False, do_cocycles=False)
    dgm2 = R['dgms'][1]
    d = persim.bottleneck(dgm1, dgm2, matching=False)
    sim_bn.append(d)
    d = persim.sliced_wasserstein(dgm1, dgm2)
    sim_w.append(d)
    if i%500 == 0:
        barcodes = R['dgms']
        plt.figure(figsize=(8, 8), dpi=100)
        plot_diagrams(barcodes, size=40, lifetime=False, show=False)
        plt.title("Persistence Diagram")
        plt.savefig('./plots/dgm_%d.png' % i)
        plt.show()
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Ori[:, 0], Ori[:, 1], Ori[:, 2], label="clean data")
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], label="noisy data")
        plt.savefig('./plots/3d_%d.png' % i)
        plt.show()

plt.clf()
plt.plot(x, sim_bn, ls='-')
plt.savefig('./plots/sim_bn.png')
plt.show()
plt.clf()
plt.plot(x, sim_w, ls='-')
plt.savefig('./plots/sim_w.png')
plt.show()
