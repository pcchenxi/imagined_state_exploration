# from density_estimator import Gaussian_Density_Estimator
from sklearn.neighbors.kde import KernelDensity
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
# import matplotlib
# gui_env = ['TKAgg','GTK3Agg','Qt4Agg','WXAgg']
# matplotlib.use(gui_env[1],warn=False, force=True)
# from matplotlib import pyplot as plt
# for gui in gui_env:
#     try:
#         print( "testing", gui)
#         matplotlib.use(gui,warn=False, force=True)
#         from matplotlib import pyplot as plt
#         break
#     except:
#         continue
# print ("Using:",matplotlib.get_backend())

class IIAYN_density:
    def __init__(self):
        self.obs_hist = None 
        self.obs_next_hist = None
        self.dones = None

        self.obs_in_use = None
        self.dist_weight = None
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



class IIAYN_coverage:
    def __init__(self, sigma=0.05):
        self.obs_hist = None 
        self.obs_next_hist = None
        self.dones = None

        self.obs_in_use = None
        self.dist_weight = None
        self.beta = 1/(sigma**2 * 2)
        files = glob.glob('coverage_*.jpg')
        for f in files:
            os.remove(f)   



    def activate_buffer(self):
        print('activate buffer', len(self.obs_hist))
        self.obs_in_use = self.obs_hist.copy()
        obs_min = np.min(self.obs_in_use, axis=0)
        obs_max = np.max(self.obs_in_use, axis=0)
        obs_max_diff = obs_max - obs_min
        self.dist_weight = obs_max_diff.max()/obs_max_diff
        print(obs_max_diff)
        print(self.dist_weight)


        self.plot_coverage(len(self.obs_in_use))
        # plt.clf()
        # self.plot_coverage_ori()


    # def update_history(self, obs, obs_next, dones):
    def update_history(self, obs):
        if self.obs_hist is None:
            self.obs_hist = obs 
            # self.obs_next_hist = obs_next
            # self.done_hist = dones
        else:
            self.obs_hist = np.concatenate((self.obs_hist, obs), axis=0)
            # self.obs_next_hist = np.concatenate((self.obs_next_hist, obs_next), axis=0)
            # self.done_hist = np.concatenate((self.done_hist, dones), axis=0)


    def get_pvisited(self, obs_test=None):
        if obs_test is None:
            obs_test = self.obs_in_use.copy()

        p_visited = np.zeros(len(obs_test))

        # print(self.obs_in_use)
        for i in range(len(obs_test)):
            obs = obs_test[i]
            obs_diff = self.obs_in_use - obs
            for d in range(len(obs_diff[0])):
                obs_diff[:,d] *= self.dist_weight[d]

            diff_norm = LA.norm(obs_diff, axis=1)
            min_dist = diff_norm.min()

            # min_dist = np.array([np.linalg.norm(x+y) for (x,y) in self.obs_in_use-obs]).min()

            pv = np.exp(-min_dist*min_dist*self.beta)
            p_visited[i] = 1-pv


        return p_visited


    def get_preach(self, obs, obs_neighbour):
        return 1


    def get_preal(self, obs_test):
        return 1


    def sample_states(self, obs, sample_state_num):
        mu = obs
        sigma = 1/self.dist_weight * 0.1
        samples = np.random.normal(mu, sigma, size=(sample_state_num, len(obs)))

        # print('in sampling', mu.shape, sigma.shape, samples.shape)

        # print(obs, samples)

        return samples


    def compute_reward(self, obs_test, sample_state_num=10):
        if self.obs_in_use is None:
            return [0]

        # rewards = np.zeros(len(obs_test))
        # for i in range(len(obs_test)):
        #     obs = obs_test[i]
        #     imagined_states = self.sample_states(obs, sample_state_num)
        #     p_visited = self.get_pvisited(imagined_states)
            
        #     rewards[i] = p_visited.mean()

        # return rewards #self.get_pvisited(obs) 
        return self.get_pvisited(obs_test) 


    def plot_coverage(self, index=0):
        xbins=200j
        ybins=200j
        # x,y = self.obs_hist[-1000:,1], self.obs_hist[-1000:,0]
        # xx, yy = np.mgrid[self.obs_hist[:,1].min():self.obs_hist[:,1].max():xbins,
        #               self.obs_hist[:,0].min():self.obs_hist[:,0].max():ybins]
        xx, yy = np.mgrid[-6:6:xbins,
                      -12:4:ybins]        
        xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
        # xy_train  = np.vstack([y, x]).T
        # self.estimator.fit(xy_train)
        # z = np.exp(self.estimator.score_samples(xy_sample))
        z = self.compute_reward(xy_sample)
        zz= np.reshape(z, xx.shape)

        z_ori = self.get_pvisited(xy_sample)
        zz_ori = np.reshape(z_ori, xx.shape)

        # print('in plot', xy_sample[0], xy_sample[-1])
        print(z.max(), z.min())

        fig,(ax1, ax2)  = plt.subplots(2,1)


        im = ax1.pcolormesh(yy, xx, zz, vmin=0, vmax=1)
        # ax1.scatter(x, y, s=2, facecolor='white')
        # ax1.colorbar(im)

        im = ax2.pcolormesh(yy, xx, zz_ori, vmin=0, vmax=1)
        # ax2.scatter(x, y, s=2, facecolor='white')
        # ax2.colorbar(im)

        plt.savefig('coverage_' + str(index)+'.jpg')
        print('coverage:', z_ori.mean())


# # ########## for testing ###########
# import numpy as np
# import matplotlib.pyplot as plt

# # data = np.random.randn(2**6).reshape(-1, 1)
# # data_test = np.linspace(-1, 1, 200).reshape(-1, 1)

# iiayn = IIAYN_coverage()
# # data = np.random.normal(0, 1, 1280).reshape(-1, 1)
# # data = np.array([[0]])
# data = np.array([[0,0], [0, 0.5]])
# dones = np.zeros(len(data)).reshape(-1, 1)
# iiayn.update_history(data)
# iiayn.activate_buffer()
# # print(iiayn.obs_hist.shape)

# data_test = np.array([[0.1, 0.1]])
# coverage_y = iiayn.compute_reward(data_test)

# print(coverage_y)
# # plt.scatter(data_test[:,0], coverage_y, c='b')
# # plt.scatter(data_test[:,0], -coverage_y, c='r')

# plt.show()