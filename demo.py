import numpy as np
import matplotlib.pyplot as plt
import PR

elevangle = np.loadtxt('data/elevangle') # 1 x 400
depth = np.loadtxt('data/depth') # 1193 x 400

means = np.mean(depth, axis=0)
stds = np.std(depth, axis=0)
plt.plot(elevangle, means, linewidth=0.5, color='k')
plt.plot(elevangle, means-stds, alpha=0.3, linewidth=0.5, color='k')
plt.plot(elevangle, means+stds, alpha=0.3, linewidth=0.5, color='k')
plt.fill_between(elevangle, means-stds, means+stds, alpha=0.15, color='k')
plt.xlim([-12, 0])
plt.ylim([0, 60])
plt.xlabel('Elevation angle (deg)')
plt.ylabel('Depth (m)')
plt.savefig('images/data.svg')
#plt.show()

cost, weight, depth_pred = PR.PRLearn(elevangle, means, 3)
plt.plot(elevangle, depth_pred, linewidth=0.5, color='r')
plt.savefig('images/prediction.svg')
plt.show()

plt.clf()
Ks = range(1, 6)
M = 5
J = []
for K in Ks:
    print(K)
    J.append(PR.CrossValidation(elevangle, depth, K, M))
plt.plot(Ks, J)
plt.xticks(range(1,6))
plt.xlabel('K')
plt.ylabel('J')
plt.savefig('images/crossvalidation.svg')
plt.show()