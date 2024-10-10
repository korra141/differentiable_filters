# !/usr/bin/env python3
"""
Example code for training a differentiable filter on a simulated disc tracking
task.
"""
import sys
import tensorflow as tf
import numpy as np
import os
import argparse
import time
import math

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_path)
print("Base path:", base_path)

from differentiable_filters.contexts.s1_simulation_context import S1ToyContext
from differentiable_filters.utils import recordio as tfr
import pdb
import wandb
import random
from differentiable_filters.utils.visualisation import plot_s1_energy
import matplotlib.pyplot as plt
from differentiable_filters.hef_analytical.filter import BayesFilter
from differentiable_filters.hef_analytical.s1_distributions import HarmonicExponentialDistribution,S1Gaussian,S1
from differentiable_filters.hef_analytical.s1_simulator import S1Simulator
from differentiable_filters.hef_analytical.s1_fft import S1FFT
def run_example(filter_type, loss, out_dir, batch_size, grid_size, trajectory_length, motion_noise,
                measurement_noise, train_size, initial_cov, epochs, n_traj,learning_rate):
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

    uuid = wandb.util.generate_id()
    time_ = time.strftime("%Y_%m_%d_%H_%M_%S")

    train_dir = os.path.join(out_dir, uuid + time_ + '/train')
    # data_dir = os.path.join(out_dir + '/data')
    fig_dir = os.path.join(out_dir , uuid + time_+ '/fig')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)


    debug = False

    model = FilterApplication(filter_type, loss, batch_size, grid_size, initial_cov, motion_noise,debug=debug)
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
    measurements_ = tf.expand_dims(tf.convert_to_tensor(measurements),2)
    ground_truth_ = tf.expand_dims(tf.convert_to_tensor(true_trajectories),2)
    train_dataset = tf.data.Dataset.from_tensor_slices((ground_truth_[:train_size], measurements_[:train_size]))
    train_set = train_dataset.shuffle(train_size).batch(batch_size, drop_remainder=True)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (ground_truth_[train_size:train_size + val_size], measurements_[train_size:train_size + val_size]))
    val_set = val_dataset.batch(batch_size, drop_remainder=True)


    test_dataset = tf.data.Dataset.from_tensor_slices(
        (ground_truth_[train_size + val_size:], measurements_[train_size + val_size:]))
    test_set = test_dataset.batch(batch_size, drop_remainder=True)

    # prepare the training
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    custom_step = 0

    run_name = "s1_diff_hef_filter_" + uuid

    wandb.init(
        project = "differential-hef",
        entity ="korra141",
        name = run_name,
        tags = ["version_2"]
    )

    for epoch in range(epochs):
        print("\nStart of epoch %d \n" % (epoch))
        print("Validating ...")
        dict_val = evaluate(model, val_set, "validate", batch_size, trajectory_length, n_traj, fig_dir)
        running_loss = 0
        for (x_batch_train, y_batch_train) in train_set:

            start = time.time()

            with tf.GradientTape() as tape:
                out = model(x_batch_train)

                loss_value, metrics, metric_names = \
                    model.context.get_loss(x_batch_train, y_batch_train, out)

                running_loss += loss_value.numpy().item()

                grads = tape.gradient(loss_value, model.trainable_weights)

                if (custom_step % 50 == 0):
                    dict = {}
                    for i, name in enumerate(metric_names):
                        dict[f'train/{name}'] = tf.reduce_mean(metrics[i])
                    dict['custom_step'] = custom_step
                    for i, grad in enumerate(grads):
                        dict[f'weights_{i}'] = tf.reduce_mean(grad)
                    wandb.log(dict)

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

        train_loss = running_loss / len(train_dataset)
        dict_epoch = {"epoch": epoch,
                      "train_loss": train_loss}
        dict_epoch.update(dict_val)
        wandb.log(dict_epoch)


    # test the trained model on the held out data
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    print("\n Testing")
    out_analytical_hef = lambda x : run_hef_analytical(x, step, grid_size, motion_noise,
                                            measurement_noise, initial_cov)
    test_dict = evaluate(model, test_set, "test", batch_size, trajectory_length, n_traj, fig_dir,out_analytical_hef)
    wandb.log(test_dict)
    wandb.finish()

def run_hef_analytical(test_set, step, grid_size, motion_noise, measurement_noise, initial_cov):
    """
    Run the Analytical Harmonic Exponential Filter on the test set

    Parameters
    ----------
    test_set : tf.data.Dataset
        The test set to run the filter on

    Returns
    -------
    posteriori_distribution, measurement_distribution, belief_prediction : np.ndarray
        The posteriori distribution, measurement distribution, and belief prediction
    """
    np_test_set = np.stack(test_set)
    batch_size = np_test_set.shape[0]
    trajectory_length = np_test_set.shape[1]
    grid = np.linspace(0, 2 * np.pi, grid_size, dtype=np.float64, endpoint=False)[np.newaxis, ...]
    grid_batched = np.tile(grid,[batch_size,1])
    grid_batched_reshape = np.reshape(grid_batched, (batch_size, 1, grid_size))
    fft = S1FFT(bandwidth=grid_size, oversampling_factor=2)

    simulator = S1Simulator(step=np.ones((batch_size, 1, 1)) * step, theta_initial=np_test_set[:, 0][..., np.newaxis], samples=grid_batched_reshape, fft=fft,
                            motion_noise=motion_noise, measurement_noise=measurement_noise)

    prior = S1Gaussian(mu_theta=np_test_set[:, 0][..., np.newaxis], cov=initial_cov, samples=grid_batched_reshape, fft=fft)
    filter = BayesFilter(distribution=S1, prior=prior)
    posterior_list = []
    pred_list = []
    measurement_list = []
    for iter in range(trajectory_length):
        pred_list.append(filter.prediction(motion_model=simulator.motion()).energy)
        posteriori_hat_, measurement = filter.update(measurement_model=simulator.measurement())
        posterior_list.append(posteriori_hat_.energy)
        measurement_list.append(measurement.energy)

    posteriori_distribution = np.stack(posterior_list, axis=1).squeeze(2).astype(np.float64)
    measurement_distribution = np.stack(measurement_list, axis=1).squeeze(2).astype(np.float64)
    belief_prediction = np.stack(pred_list, axis=1).squeeze(2).astype(np.float64)

    return posteriori_distribution, measurement_distribution, belief_prediction


def reset_weights(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
            reset_weights(layer) #apply function recursively
            continue

        #where are the initializers?
        if hasattr(layer, 'cell'):
            init_container = layer.cell
        else:
            init_container = layer

        for key, initializer in init_container.__dict__.items():
            if "initializer" not in key: #is this item an initializer?
                  continue #if no, skip it

            # find the corresponding variable, like the kernel or the bias
            if key == 'recurrent_initializer': #special case check
                var = getattr(init_container, 'recurrent_kernel')
            else:
                var = getattr(init_container, key.replace("_initializer", ""))

            var.assign(initializer(var.shape, var.dtype))
def evaluate(model, dataset, type, batch_size, trajectory_length, n_traj, folder_name=None,out_analytical_hef=None):
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
    outputs_analytical = {}
    metric_names = []
    plotting_dict = {}
    for step, (x_batch, y_batch) in enumerate(dataset):

        out = model(x_batch, training=False)
        plotting_dict['hef_diff'] = out

        loss_value, metrics, metric_names = \
            model.context.get_loss(x_batch, y_batch, out)
        if type == "test":
            out_analytical_hef_ = out_analytical_hef(x_batch)
            plotting_dict["hef_analytical"] = out_analytical_hef_
            loss_value_analytical, metrics_analytical, metric_names_analytical = \
                model.context.get_loss(x_batch, y_batch, out_analytical_hef_)
            if step == 0:
                for ind, k in enumerate(metric_names_analytical):
                    outputs_analytical[k] = [metrics_analytical[ind]]
            else:
                for ind, k in enumerate(metric_names_analytical):
                    outputs_analytical[k].append(metrics_analytical[ind])
        plotting_figure(plotting_dict, y_batch, x_batch, n_traj, batch_size, folder_name, step, trajectory_length,
                        frequency_iter=3)

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

    if type == "test":
        for ind, k in enumerate(metric_names_analytical):
            tf.print("hef analytical ", k, ": ", tf.reduce_mean(outputs_analytical[k]))
            dict["hef analytical" + type + "/" + k] = tf.reduce_mean(outputs_analytical[k])

    return dict


def plotting_figure(dict_predictions, label, input, n_traj, batch_size, folder_name, batch_no, trajectory_length,
                    frequency_iter=3):
    # pdb.set_trace()
    list_idx = random.sample(range(0, batch_size), n_traj)
    measurement = label
    trajectory = input
    plt.style.use('seaborn-dark-palette')
    for i in range(n_traj):
        for traj in range(trajectory_length):
            if traj % frequency_iter == 0:
                fig, ax = plt.subplots()
                for key,outputs in dict_predictions.items():
                    posterior_state, z_pred, pred_state = outputs
                    ax = plot_s1_energy(
                        [posterior_state[list_idx[i], traj], z_pred[list_idx[i], traj], pred_state[list_idx[i], traj]],ax=ax,legend=[rf'{key}_posterior', rf'{key}_measurement', rf'{key}_prediction'])
                ax.plot(tf.math.cos(trajectory[list_idx[i], traj]), tf.math.sin(trajectory[list_idx[i], traj]), 'o',
                        label="pose data")
                ax.plot(tf.math.cos(measurement[list_idx[i], traj]), tf.math.sin(measurement[list_idx[i], traj]), 'o',
                        label="measurement data")
                ax.set_title(f"Trajectory {list_idx[i]} Iteration {traj}",loc='center')
                ax.legend(bbox_to_anchor=(0.85, 1), loc='upper left',fontsize='x-small')
                plt.savefig(f"{folder_name}/s1_hef_{batch_no}_traj_{list_idx[i]}_iter{traj}.png", format='png', dpi=300)
                plt.close()


class FilterApplication(tf.keras.Model):
    def __init__(self, filter_type='ekf', loss='nll', batch_size=32,
                 grid_size=20, initial_cov=0.1, motion_noise=0.5, debug=False,**kwargs):
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
        angle = tf.math.acos(tf.einsum('pmn,pkn->pm', x, mu))
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
        samples_ = tf.expand_dims((tf.linspace(tensor_start, tensor_stop, self.grid_size),0))
        samples_batched = tf.tile(samples_, [self.batch_size, 1])

        init_state = (tf.reshape(tf.convert_to_tensor(self.compute_energy(samples_batched, mu)),
                                 [self.batch_size, -1]), tf.zeros([self.batch_size, 1]))

        outputs = self.rnn_layer(inputs, training=training, initial_state=init_state)

        return outputs


def main():
    parser = argparse.ArgumentParser('run example')
    parser.add_argument('--out_dir', dest='out_dir', type=str, default='output',help='where to store results')
    parser.add_argument('--filter', dest='filter', type=str,
                        default='hef',
                        help='which filter class to use')
    parser.add_argument('--loss', dest='loss', type=str,
                        default='nll',
                        help='which loss function to use')
    parser.add_argument('--batch_size', dest='batch_size',
                        type=int, default=16, help='batch size for training')
    parser.add_argument('--grid_size', dest='grid_size', type=int,
                        default=20,
                        help='bandwidth of the harmonic exponential distribution')
    parser.add_argument('--trajectory_length', dest='trajectory_length', type=int,
                        default=30,
                        help='length of the trajectory used to create dataset in S1')
    parser.add_argument('--motion_noise', dest='motion_noise', type=float,
                        default=0.1,
                        help='motion noise for the analytic process model')
    parser.add_argument('--measurement_noise', dest='measurement_noise', type=float,
                        default=0.1,
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
    parser.add_argument('--learning_rate', dest='learning_rate',
                        type=float, default=1e-3,
                        help='learning rate of the neural network')

    args = parser.parse_args()

    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.print(f"Random seed set as {seed}")

    wandb.init(config=args)

    run_example(args.filter, args.loss, args.out_dir, args.batch_size, args.grid_size,args.trajectory_length, args.motion_noise, args.measurement_noise, args.train_size, args.initial_cov, args.epochs, args.n_traj,args.learning_rate)


if __name__ == "__main__":
    main()
