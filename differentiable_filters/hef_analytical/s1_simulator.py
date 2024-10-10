import pdb
from typing import Optional

import numpy as np

from differentiable_filters.hef_analytical.s1_distributions import S1, S1Gaussian, S1MultimodalGaussian
from differentiable_filters.hef_analytical.base_fft import FFTBase
from differentiable_filters.hef_analytical.simulator_base import Simulator


class S1Simulator(Simulator):
    def __init__(self, step, theta_initial,
                 samples: Optional[np.ndarray] = None,
                 fft: Optional[FFTBase] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.step = step
        self.samples = samples
        self.fft = fft
        self.theta = theta_initial
        # If motion noise or measurement noise are zero, default to 0.1 and 0.4 respectively
        self.motion_cov = self.motion_noise ** 2 if self.motion_noise != 0.0 else 0.1
        self.measurement_cov = self.measurement_noise ** 2 if self.measurement_noise != 0.0 else 0.4

    def motion(self) -> S1:
        """
        Simulate motion
        :return: S1 distribution of the predicted motion.
        """
        self.theta = (self.theta + self.step)
        # Jitter step with noise and wrap theta between 0 and 2pi
        noisy_prediction = (self.step + np.random.normal(0.0, self.motion_noise, self.theta.shape)) % (2 * np.pi)
        return S1Gaussian(mu_theta=noisy_prediction,
                          cov=self.motion_cov,
                          samples=self.samples,
                          fft=self.fft)

    def measurement(self) -> S1:
        """
        Simulate measurement
        :return: S1 Distribution of measurement model.
        """
        # Jitter measurement with noise and make sure theta is between 0 and 2pi
        noisy_measurement = (self.theta + np.random.normal(0.0, self.measurement_noise, self.theta.shape)) % (2 * np.pi)
        return S1Gaussian(mu_theta=noisy_measurement,
                          cov=self.measurement_cov,
                          samples=self.samples,
                          fft=self.fft)
