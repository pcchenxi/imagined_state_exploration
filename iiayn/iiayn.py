from density_estimator import Gaussian_Density_Estimator
from sklearn.neighbors.kde import KernelDensity
import numpy as np

class IIAYN:
    def __init__(self):
        self.obs_hist = None 
        self.obs_next_hist = None
        self.dones = None

        # self.density_estimator = Gaussian_Density_Estimator()
        self.density_estimator = KernelDensity(kernel='gaussian', bandwidth=0.1)
        self.density_estimator_q = KernelDensity(kernel='gaussian', bandwidth=0.1)

    def update_history(self, obs, obs_next, dones):
        if self.obs_hist is None:
            self.obs_hist = obs 
            self.obs_next_hist = obs_next
            self.done_hist = dones
        else:
            self.obs_hist = np.concatenate((self.obs_hist, obs), axis=0)
            self.obs_next_hist = np.concatenate((self.obs_next_hist, obs_next), axis=0)
            self.done_hist = np.concatenate((self.done_hist, dones), axis=0)


    def train_density_estimator(self, hist_index=None):
        if hist_index is None:
            self.density_estimator.fit(self.obs_hist)
        else:
            data = self.obs_hist[hist_index]
            self.density_estimator.fit(data)


    def test_density_estimator(self, test_data):
        y = np.exp(self.density_estimator.score_samples(test_data))
        return y

    def get_pvisited(self, obs_test=None):
        if obs_test is None:
            pvisited = np.exp(self.density_estimator.score_samples(self.obs_hist))
        else:
            pvisited = np.exp(self.density_estimator.score_samples(obs_test))
        return pvisited


    def get_preach(self, obs, obs_neighbour):
        return 1


    def get_preal(self, obs_test):
        return 1


    def get_hist_entropy(self, hist_index=None):
        if hist_index is None:
            pvisited = self.get_pvisited()
            entropy = np.sum(-pvisited * np.log(pvisited))/len(self.obs_hist)

        else:
            data = self.obs_hist[hist_index]
            pvisited = self.get_pvisited(data)
            entropy = np.sum(-pvisited * np.log(pvisited))/len(hist_index)

        return entropy


    def sample_states(self, obs, sample_state_num):
        # mu = np.zeros(len(obs))
        # sigma = np.ones(len(obs))
        # samples = np.random.normal(mu, sigma, sample_state_num) + obs

        return [obs]


    def compute_reward(self, obs, sample_state_num):
        imagined_states = self.sample_states(obs, sample_state_num)
        sum_reward = 0
        for i_state in imagined_states:
            preal = self.get_preal(i_state)
            pvisit = self.get_pvisited(i_state)
            preach = self.get_preach(obs, i_state)

            r_si = preal*preach*pvisit
            sum_reward += r_si 
        
        avg_rew = sum_reward/len(imagined_states)

        return avg_rew 


import numpy as np
import matplotlib.pyplot as plt

# data = np.random.randn(2**6).reshape(-1, 1)
iiayn = IIAYN()
data = np.random.normal(0, 1, 128).reshape(-1, 1)
dones = np.zeros(len(data)).reshape(-1, 1)
iiayn.update_history(data, data, dones)
print(iiayn.obs_hist.shape, iiayn.obs_next_hist.shape)

iiayn.train_density_estimator()
y = iiayn.get_pvisited(iiayn.obs_hist)
print(y.shape)

hist_entropy = iiayn.get_hist_entropy()
l = np.arange(len(iiayn.obs_hist))
sub_entorpy = np.zeros(len(l))

for i in range(1, len(l)-1, 1):
    index_list = np.concatenate((l[:i],l[i+1:]))

    # iiayn.train_density_estimator(hist_index=index_list)
    # sub_entropy = iiayn.get_hist_entropy(hist_index=index_list)
    # y_sub = iiayn.get_pvisited(iiayn.obs_hist)


    # sub_entorpy[i] = (hist_entropy - sub_entropy)*10
    # print(hist_entropy - sub_entropy, y[i])

    # plt.clf()
    # plt.scatter(iiayn.obs_hist[:,0], y, c='r')
    # plt.scatter(iiayn.obs_hist[:,0], y_sub, c='b', s = 1)
    # plt.scatter(iiayn.obs_hist[i,0], y[i], c='g', s=10)
    # # plt.scatter(iiayn.obs_hist[:,0], sub_entorpy, c='b') #; plt.tight_layout()
    # plt.show()

    iiayn.density_estimator_q.fit([iiayn.obs_hist[i]])
    y_q = np.exp(iiayn.density_estimator_q.score_samples(iiayn.obs_hist))

    kl = (y * np.log(y/(y_q+0.00001))).sum()
    sub_entorpy[i] = kl
    print(kl, y[i])


print(hist_entropy, sub_entorpy.max())

plt.scatter(iiayn.obs_hist[:,0], (y-y.min())/(y.max()-y.min()), c='b') #; plt.tight_layout()
plt.scatter(iiayn.obs_hist[:,0], (sub_entorpy-sub_entorpy.min())/(sub_entorpy.max()-sub_entorpy.min()), c='r')
# plt.scatter(sub_entorpy, y) #; plt.tight_layout()

plt.show()

