from KDEpy import FFTKDE


class Gaussian_Density_Estimator:

    def __init__(self, kernel='gaussian', bw='silverman'):
        self.estimator = FFTKDE(kernel=kernel, bw=bw)

    def train(self, data, weights=None):
        self.estimator.fit(data, weights=weights)

    def score_samples(self, input_x=None):
        if input_x is None:
            x, y = self.estimator.evaluate()
            return x, y
        else:
            y = self.estimator.evaluate(input_x)
            return y


# import numpy as np
# import matplotlib.pyplot as plt

# data = np.random.randn(2**6)
# density_estimator = Gaussian_Density_Estimator()
# density_estimator.train(data)
# # x, y = density_estimator.score_samples()
# # print(x.shape, y.shape)

# x, y = density_estimator.score_samples(10)
# print(y)

# plt.plot(x, y); plt.tight_layout()
# plt.show()