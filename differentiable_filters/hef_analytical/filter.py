from typing import Type, List
from copy import deepcopy
import numpy as np
from numpy.fft import ifftshift
from scipy.interpolate import RegularGridInterpolator
from einops import rearrange

from differentiable_filters.hef_analytical.distribution_base import HarmonicExponentialDistribution
from lie_learn.spectral.SE2FFT import SE2_FFT


class BayesFilter:
    def __init__(
            self,
            distribution: Type[HarmonicExponentialDistribution],
            prior: HarmonicExponentialDistribution,
    ):
        """
        :param distribution: Type of distribution to filter.
        :param prior: a prior distribution of type "distribution".
        """
        self.distribution = distribution
        self.prior: HarmonicExponentialDistribution = prior

    def prediction(
            self, motion_model: HarmonicExponentialDistribution
    ) -> HarmonicExponentialDistribution:
        """
        Prediction step
        :param motion_model: motion model for prediction step
        :return unnormalized belief distribution
        """
        # Convolve prior and motion model
        predict = self.distribution.convolve(motion_model, self.prior)
        # There is no need to normalize the prediction as it is already normalized
        # See line 62 in distributions/distribution_base.py
        # predict.normalize()
        # Update prior
        self.prior = deepcopy(predict)

        return predict

    def update(
            self, measurement_model: HarmonicExponentialDistribution
    ) -> HarmonicExponentialDistribution:
        """
        Update step
        :param measurement_model: measurement model for update step
        :return unnormalized posterior distribution
        """
        # Product of belief and measurement model
        update = self.distribution.product(self.prior, measurement_model)
        update.normalize()
        # Update prior with new belief
        self.prior = deepcopy(update)

        return update, measurement_model
