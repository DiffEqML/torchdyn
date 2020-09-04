import torch


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        iters = max(1, iters)
        self.val = maxval / iters
        self.maxval = maxval
        self.iters = iters

    def step(self):
        self.val = min(self.maxval, self.val + self.maxval / self.iters)

    def __call__(self):
        return self.val


class EMAMetric(object):
    """
    Exponential Moving Average Metric
    """
    def __init__(self, gamma=.99):
        super(EMAMetric, self).__init__()
        self.prev_metric = 0.
        self.gamma = gamma

    def step(self, x):
        with torch.no_grad():
            self.prev_metric = (1. - self.gamma) * self.prev_metric + self.gamma * x

    def val(self):
        return self.prev_metric
