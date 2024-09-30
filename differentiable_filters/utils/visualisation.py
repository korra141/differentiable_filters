
from typing import List, Type

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import tensorflow as tf

from lie_learn.spaces.Tn import linspace
import math


def plot_s1_func(f, legend=None, ax=None, plot_type: str = 'polar'):
    if ax is None:
        _, ax = plt.subplots(1, 1)

    if legend is None:
        legend = [rf'$prob_{i}$' for i, _ in enumerate(f)]

    # Working on unit circle
    radii = 1.0
    bandwidth = f[0].shape[0]

    # First plot the support of the distributions S^1
    tensor_start = tf.constant(0, dtype=tf.float32)
    tensor_stop = tf.constant(2 * math.pi, dtype=tf.float32)
    theta = tf.linspace(tensor_start, tensor_stop, bandwidth)
    theta = tf.concat([theta, theta[0, None]], 0)

    ct = tf.math.cos(theta)
    st = tf.math.sin(theta)

    theta_1 = tf.linspace(tensor_start, tensor_stop, 100)
    theta_1 = tf.concat([theta_1, theta_1[0, None]], 0)

    ct_1 = tf.math.cos(theta_1)
    st_1= tf.math.sin(theta_1)



    # First plot circle
    ax.plot(ct_1, st_1, 'k-', lw=3, alpha=0.6)

    # Plot functions in polar coordinates
    for i, f_bar in enumerate(f):
        # Concat first element to close function
        f_bar = tf.concat([f_bar, f_bar[0,None]],0)
        # Use only real components of the function and offset to unit radius
        f_real = tf.math.real(f_bar) * 0.5 + radii
        f_x = ct * f_real
        f_y = st * f_real
        # Plot circle using x and y coordinates
        ax.plot(f_x, f_y, '-', lw=3, alpha=0.5, label=legend[i])
    # Only set axis off for polar plot
    plt.axis('off')
    # Set aspect ratio to equal, to create a perfect circle
    ax.set_aspect('equal')
    # Annotate axes in circle
    ax.text(1.05, 0, rf'0', style='italic', fontsize=15)
    ax.text(-1.15, 0, r'$\pi$', style='italic', fontsize=15)
    ax.text(0, 1.12, r'$\frac{\pi}{2}$', style='italic', fontsize=20)
    ax.text(0, -1.12, r'$-\frac{\pi}{2}$', style='italic', fontsize=20)
    return ax

def plot_s1_energy(energy_samples_list,
                     legend=None,
                     ax=None,
                     plot_type: str = 'polar'):
    """Process multiple functions at once for plotting"""

    f = []
    for energy_samples in energy_samples_list:

        maximum = tf.math.reduce_max(energy_samples)
        moments = tf.signal.rfft(tf.exp(energy_samples - maximum))
        ln_z_ = tf.math.real(tf.math.log(moments[0] / math.pi))  + maximum
        prob = tf.math.exp(energy_samples - ln_z_)
        f.append(prob)
    return plot_s1_func(f, legend, ax, plot_type)