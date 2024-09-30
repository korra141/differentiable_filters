# !/usr/bin/env python3
"""
Example code for training a differentiable filter on a simulated disc tracking
task.
"""

import tensorflow as tf
import numpy as np
import os
import argparse
import time
import math

from differentiable_filters.contexts.s1_simulation_context import S1ToyContext
from differentiable_filters.utils import recordio as tfr
import pdb
import wandb
from sklearn.model_selection import KFold
import random
from differentiable_filters.utils.visualisation import plot_s1_energy
import matplotlib.pyplot as plt
import keras.backend as K


def run_example(filter_type, loss, out_dir, batch_size, grid_size, use_gpu, debug, trajectory_length, motion_noise,
                measurement_noise, train_size, initial_cov, seed, epochs, n_traj):
    """
    Exemplary code to set up and train a differentiable filter for the
    simulated disc tracking task described in the paper "How to train your
    Differentiable FIlter"

    Parameters
    ----------
    filter_type : str
        Defines which filtering algorithm is used. Can be ekf, ukf, mcukf or pf
    loss : str
        Which loss to use for training the filter. This can be "nll" for the
        negative log likelihood, "mse" for the mean squared error or "mixed"
        for a combination of both
    out_dir : str
        Path to the directory where results and data should be written to.
    batch_size : int
        Batch size for training and testing.
    hetero_q : bool
        If true, heteroscedastic process noise is learned, else constant.
    hetero_r : bool
        If true, heteroscedastic observation noise is learned, else constant.
    learned_process : bool
        If true, a neural network is used as process model in the filter, else
        an analytical process model is used.
    image_size : int
        Width and height of the image observations
    use_gpu : bool
        If true, the training and testing is run on GPU (if one is available)
    debug : bool
        Turns on additional debug output and prints.

    Returns
    -------
    None.

    """
    if use_gpu:
        # limit tensorflows gpuy memory consumption
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
    else:
        # Hide GPU from visible devices to run on cpu
        tf.config.set_visible_devices([], 'GPU')

    uuid = wandb.util.generate_id()
    # prepare the output directories
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(os.path.join(out_dir, uuid)):
        os.makedirs(os.path.join(out_dir, uuid))
    train_dir = os.path.join(out_dir, uuid + '/train')
    # data_dir = os.path.join(out_dir + '/data')
    fig_dir = os.path.join(out_dir , uuid + '/fig')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    # if not os.path.exists(data_dir):
    #     os.makedirs(data_dir)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)



    model = FilterApplication(filter_type, loss, batch_size, grid_size, initial_cov, motion_noise,
                              debug=debug)
    val_size = 72
    test_size = 30
    step = 0.3

    n_samples = train_size + val_size + test_size
    starting_positions = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)

    true_trajectories = np.ndarray((n_samples, trajectory_length))
    measurements = np.ndarray((n_samples, trajectory_length))

    for i in range(n_samples):
        theta = starting_positions[i]  # starting position
        for j in range(trajectory_length):
            true_trajectories[i][j] = theta % (2 * np.pi)
            measurements[i][j] = (theta + np.random.normal(0.0, measurement_noise, 1).item()) % (2 * np.pi)
            theta = theta + step
    measurements_ = tf.expand_dims(tf.convert_to_tensor(measurements), 2)
    ground_truth_ = tf.expand_dims(tf.convert_to_tensor(true_trajectories), 2)
    poses_train_val, observations_train_val = true_trajectories[:train_size + val_size], measurements[
                                                                                         :train_size + val_size]

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (ground_truth_[train_size + val_size:], measurements_[train_size + val_size:]))
    test_set = test_dataset.batch(batch_size, drop_remainder=True)

    kfold = KFold(n_splits=2, shuffle=True)

    # prepare the training
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    custom_step = 0

    train_summary_writer = tf.summary.create_file_writer(train_dir + '/' +
                                                         uuid)
    tf.summary.experimental.set_step(custom_step)
    group_name = "s1_hef_filter-" + uuid

    # Wandb settings

    for i, (train_idx, val_idx) in enumerate(kfold.split(poses_train_val, observations_train_val)):

        print(f"Training for Kfold:{i}")

        wandb.init(
            project="differential-hef",
            entity="korra141",
            config={"fold": i}
        )
        reset_weights(model)
        wandb.watch(model, log_freq=50)

        poses_train = tf.expand_dims(tf.convert_to_tensor(poses_train_val[np.array([train_idx])].squeeze()), 2)
        measurements_train = tf.expand_dims(
            tf.convert_to_tensor(observations_train_val[np.array([train_idx])].squeeze()), 2)
        poses_val = tf.expand_dims(tf.convert_to_tensor(poses_train_val[np.array([val_idx])].squeeze()), 2)
        measurements_val = tf.expand_dims(
            tf.convert_to_tensor(observations_train_val[np.array([val_idx])].squeeze()), 2)
        train_dataset = tf.data.Dataset.from_tensor_slices((poses_train, measurements_train))
        train_set = train_dataset.shuffle(train_size, seed=seed).batch(batch_size, drop_remainder=True)

        val_dataset = tf.data.Dataset.from_tensor_slices((poses_val, measurements_val))
        val_set = val_dataset.batch(batch_size, drop_remainder=True)

        for epoch in range(epochs):
            print("\nStart of epoch %d \n" % (epoch))
            print("Validating ...")
            dict_val = evaluate(model, val_set, "validate", batch_size, trajectory_length, n_traj, None)
            running_loss = 0
            for (x_batch_train, y_batch_train) in train_set:

                start = time.time()

                with tf.GradientTape() as tape:
                    out = model(x_batch_train)

                    loss_value, metrics, metric_names = \
                        model.context.get_loss(x_batch_train, y_batch_train, out)

                    # pdb.set_trace()
                    running_loss += loss_value.numpy().item()

                    # log summaries of the metrics every 50 steps
                    if(custom_step%50 == 0):
                        dict = {}
                        for i, name in enumerate(metric_names):
                            dict[f'train/{name}'] = tf.reduce_mean(metrics[i])
                        dict['custom_step'] = custom_step
                        dict["fold"] =  i
                        wandb.log(dict)

                    # Use the gradient tape to automatically retrieve the
                    # gradients of the trainable variables with respect to the loss.
                    grads = tape.gradient(loss_value, model.trainable_weights)

                    # wandb.log({"gradients": wandb.Histogram(grads)})

                    # Run one step of gradient descent by updating
                    # the value of the variables to minimize the loss.
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))
                end = time.time()

                if custom_step % 50 == 0:
                    print("Training loss at step %d: %.4f (took %.3f seconds) " %
                          (custom_step, float(loss_value), float(end - start)))
                    # wandb.log("Training loss at step %d: %.4f (took %.3f seconds) " %
                    #       (step, float(loss_value), float(end-start)))
                custom_step += 1
                tf.summary.experimental.set_step(custom_step)
            train_loss = running_loss / len(train_dataset)
            dict_epoch = {"fold":i,
                          "epoch": epoch,
                          "train_loss": train_loss}
            dict_epoch.update(dict_val)
            wandb.log(dict_epoch)

        model.save_weights(os.path.join(train_dir,f'model_fold_{i}_.weights.h5'))

    # test the trained model on the held out data
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    print("\n Testing")
    for f in os.listdir(train_dir):
        reset_weights(model)
        if f.endswith('.weights.h5'):
            fold = f.split('_')[2]
            model = model.load_weights(f)
            test_dict = evaluate(model, test_set, "test", batch_size, trajectory_length, n_traj, fig_dir)
            test_dict["fold"] = fold
        wandb.log(test_dict)
    wandb.finish()


def reset_weights(model):

    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session)

def evaluate(model, dataset, type, batch_size, trajectory_length, n_traj, folder_name=None):
    """
    Evaluates the model on the given dataset (without training)

    Parameters
    ----------
    model : tf.keras.Model
        The model to evaluate
    dataset : tf.data.Dataset
        The dataset on which to evaluate the model
    type : string
        if the function is used for validating or testing

    Returns
    -------
    None.

    """
    outputs = {}
    metric_names = []
    for step, (x_batch, y_batch) in enumerate(dataset):

        out = model(x_batch, training=False)

        # if type == "test":
        plotting_figure(out, y_batch, x_batch, n_traj, batch_size, folder_name, step, trajectory_length,
                        frequency_iter=3)

        # Compute the loss and metrics for this minibatch.
        loss_value, metrics, metric_names = \
            model.context.get_loss(x_batch, y_batch, out)

        if step == 0:
            for ind, k in enumerate(metric_names):
                outputs[k] = [metrics[ind]]
        else:
            for ind, k in enumerate(metric_names):
                outputs[k].append(metrics[ind])

    print(f'{type} Result: ')
    # pdb.set_trace()
    dict = {}
    for ind, k in enumerate(metric_names):
        tf.print(k, ": ", tf.reduce_mean(outputs[k]))
        dict[type + "/" + k] = tf.reduce_mean(outputs[k])
    tf.print('\n')

    return dict


def plotting_figure(prediction, label, input, n_traj, batch_size, folder_name, batch_no, trajectory_length,
                    frequency_iter=3):
    # pdb.set_trace()
    list_idx = random.sample(range(0, batch_size), n_traj)
    posterior_state, z_pred, pred_state = prediction
    measurement = label
    trajectory = input
    for i in range(n_traj):
        for traj in range(trajectory_length):
            if traj % frequency_iter == 0:
                pdb.set_trace()
                ax = plot_s1_energy(
                    [posterior_state[list_idx[i], traj], z_pred[list_idx[i], traj], pred_state[list_idx[i], traj]])
                ax.plot(tf.math.cos(trajectory[list_idx[i], traj]), tf.math.sin(trajectory[list_idx[i], traj]), 'o',
                        label="pose data")
                ax.plot(tf.math.cos(measurement[list_idx[i], traj]), tf.math.sin(measurement[list_idx[i], traj]), 'o',
                        label="measurement data")
                plt.savefig(f"{folder_name}/s1_hef_{batch_no}_traj_{list_idx[i]}_iter{traj}.png", format='png', dpi=300)
                plt.close()


class FilterApplication(tf.keras.Model):
    def __init__(self, filter_type='ekf', loss='nll', batch_size=32,
                 grid_size=20, initial_cov=0.1, motion_noise=0.5, debug=False, **kwargs):
        """
        Tf.keras.Model that combines a differentiable filter and a problem
        context to run filtering on this problem.

        Parameters
        ----------
        filter_type : str, optional
            A string that defines which filter to use, can be ekf, ukf, mcukf
            or pf. Default is ekf.
        batch_size : int, optional
            Batch size. Default is 32
        loss : str, optional
            The loss function to use, can be nll, mse or mixed. Default is nll.
        hetero_q : bool, optional
            Learn heteroscedastic process noise? Default is False.
        hetero_e : bool, optional
            Learn heteroscedastic observation noise? Default is True.
        learned_process : bool, optional
            Learn the process model or use an analytical one? Default is True.
        image_size : int, optional
            Width and height of the image observations. Default is 120.
        debug : bool, optional
            Print debug output? Default is False.

        Raises
        ------
        ValueError
            If the desired filter class (filter_type) is not implemented

        Returns
        -------
        None.

        """
        super(FilterApplication, self).__init__(**kwargs)

        # -------------------------- (1) --------------------------------------
        # Construct the context class that describes the problem on which
        # we want to run a differentiable filter
        # ----------------------------------------------------------------------
        self.grid_size = grid_size
        self.context = S1ToyContext(batch_size, filter_type, self.grid_size, motion_noise,loss)

        # -------------------------- (2) --------------------------------------
        # Instantiate the desired filter cell
        # ----------------------------------------------------------------------
        problem = 'simple'
        if filter_type == 'ekf':
            from differentiable_filters.filters import ekf_cell as ekf
            self.cell = ekf.EKFCell(self.context, problem, debug=debug)
        elif filter_type == 'ukf':
            from differentiable_filters.filters import ukf_cell as ukf
            self.cell = ukf.UKFCell(self.context, problem, debug=debug)
        elif filter_type == 'mcukf':
            from differentiable_filters.filters import mcukf_cell as mcukf
            self.cell = mcukf.MCUKFCell(self.context, problem, debug=debug)
        elif filter_type == 'pf':
            from differentiable_filters.filters import pf_cell as pf
            self.cell = pf.PFCell(self.context, problem, debug=debug)
        elif filter_type == 'hef':
            from differentiable_filters.filters import hef_cell as hef
            self.cell = hef.HEFCell(self.context, problem, self.grid_size, debug=debug)
        else:
            self.log.error('Unknown filter type: ' + filter_type)
            raise ValueError('Unknown filter type: ' + filter_type)

        # -------------------------- (3) --------------------------------------
        # wrap the Filter cell in a keras RNN Layer
        # ---------------------------------------------------------------------
        self.rnn_layer = tf.keras.layers.RNN(self.cell, return_sequences=True,
                                             unroll=False)

        # store some shape related information
        self.batch_size = self.context.batch_size
        # self.image_size = image_size
        self.filter_type = filter_type

        # -------------------------- (4) --------------------------------------
        # Define the covariance matrix for the initial belief of the filter
        # ---------------------------------------------------------------------

        self.cov = initial_cov

    def compute_energy(self, theta, mu):
        x = self.theta_to_2D(theta)
        angle = tf.math.acos(tf.squeeze(tf.linalg.matmul(x, mu, transpose_b=True), axis=-1))
        return -0.5 * tf.math.pow(angle, 2) / self.cov

    def theta_to_2D(self, theta):
        ct = tf.math.cos(theta)
        st = tf.math.sin(theta)
        out = tf.stack([ct, st], axis=-1)
        return out

    def __call__(self, inputs, training=True):
        """
        Run one step of prediction with the model

        Parameters
        ----------
        inputs : list of tensors
            the input tensors include the sequence of raw sensory observations,
            the true initial satte of the system and a noise vector to perturb
            this initial state before passing it to the filter
        training : bool
            if the model is run in training or test mode

        Returns
        -------
        res : list of tensors
            the prediction output

        """

        mu = self.theta_to_2D(inputs[:, 0])
        tensor_start = tf.constant(0, dtype=tf.float64)
        tensor_stop = tf.constant(2 * math.pi, dtype=tf.float64)
        samples_ = tf.linspace(tensor_start, tensor_stop, self.grid_size * self.batch_size)
        samples = tf.reshape(samples_, [self.batch_size, self.grid_size])
        init_state = (tf.reshape(tf.convert_to_tensor(self.compute_energy(samples, mu)),
                                 [self.batch_size, -1]), tf.zeros([self.batch_size, 1]))

        outputs = self.rnn_layer(inputs, training=training, initial_state=init_state)

        return outputs


def main():
    parser = argparse.ArgumentParser('run example')
    parser.add_argument('--out-dir', dest='out_dir', type=str,
                        required=True, help='where to store results')
    parser.add_argument('--filter', dest='filter', type=str,
                        default='hef', choices=['ekf', 'ukf', 'mcukf', 'pf'],
                        help='which filter class to use')
    parser.add_argument('--loss', dest='loss', type=str,
                        default='nll', choices=['nll', 'mse', 'mixed'],
                        help='which loss function to use')
    parser.add_argument('--batch-size', dest='batch_size',
                        type=int, default=16, help='batch size for training')
    # parser.add_argument('--image-size', dest='image_size',
    #                     type=int, default=120,
    #                     help='width and height of image observations')
    # parser.add_argument('--hetero-q', dest='hetero_q', type=int,
    #                     choices=[0, 1], default=1,
    #                     help='learn heteroscedastic process noise?')
    # parser.add_argument('--hetero-r', dest='hetero_r', type=int,
    #                     choices=[0, 1], default=1,
    #                     help='learn heteroscedastic observation noise?')
    parser.add_argument('--grid_size', dest='grid_size', type=int,
                        default=20,
                        help='bandwidth of the harmonic exponential distribution')
    parser.add_argument('--gpu', dest='gpu',
                        type=int, choices=[0, 1], default=1,
                        help='if true, the code is run on gpu if one is found')
    parser.add_argument('--debug', dest='debug',
                        type=int, choices=[0, 1], default=0,
                        help='turns debugging on/off ')
    parser.add_argument('--trajectory_length', dest='trajectory_length', type=int,
                        default=30,
                        help='length of the trajectory used to create dataset in S1')
    parser.add_argument('--motion_noise', dest='motion_noise', type=float,
                        default=0.5,
                        help='motion noise for the analytic process model')
    parser.add_argument('--measurement_noise', dest='measurement_noise', type=float,
                        default=0.5,
                        help='measurement noise to create observations in the dataset')
    parser.add_argument('--train_size', dest='train_size', type=int,
                        default=480,
                        help='length of the training dataset')
    parser.add_argument('--initial_cov', dest='initial_cov', type=float,
                        default=0.1,
                        help='noise around the starting state fed as a prior to the model')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=3,
                        help='no of times the model is trained for the complete dataset')
    parser.add_argument('--seed', dest='seed',
                        type=int, default=12345,
                        help='argument for the randomness')
    parser.add_argument('--n_traj', dest='n_traj',
                        type=int, default=5,
                        help='number of trajectories to plot per batch for evaluation')

    args = parser.parse_args()

    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.print(f"Random seed set as {seed}")

    run_example(args.filter, args.loss, args.out_dir, args.batch_size, args.grid_size, args.gpu, args.debug,
                args.trajectory_length, args.motion_noise, args.measurement_noise, args.train_size, args.initial_cov,
                args.seed, args.epochs, args.n_traj)


if __name__ == "__main__":
    main()
