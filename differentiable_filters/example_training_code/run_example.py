# !/usr/bin/env python3
"""
Example code for training a differentiable filter on a simulated disc tracking
task.
"""
import pdb
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
import wandb
import random
from differentiable_filters.utils.video_generation import generate_video_from_images
from differentiable_filters.utils.visualisation import plot_s1_energy
import matplotlib.pyplot as plt
from differentiable_filters.hef_analytical.filter import BayesFilter
from differentiable_filters.hef_analytical.s1_distributions import HarmonicExponentialDistribution, S1Gaussian, S1
from differentiable_filters.hef_analytical.s1_simulator import S1Simulator
from differentiable_filters.hef_analytical.s1_fft import S1FFT
import glob

import tensorflow as tf

def serialize_example(input_tensor, label_tensor):
    feature = {
        'input': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(input_tensor).numpy()])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(label_tensor).numpy()]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
def write_tfrecord(file_path, dataset):
    with tf.io.TFRecordWriter(file_path) as writer:
        for input_tensor, label_tensor in dataset:
            example = serialize_example(input_tensor, label_tensor)
            writer.write(example)
def parse_example(example_proto):
    feature_description = {
        'input': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    input_tensor = tf.io.parse_tensor(example['input'], out_type=tf.float32)
    label_tensor = tf.io.parse_tensor(example['label'], out_type=tf.float32)
    return input_tensor, label_tensor
def read_tfrecord(file_path):
    raw_dataset = tf.data.TFRecordDataset(file_path)
    parsed_dataset = raw_dataset.map(parse_example)
    return parsed_dataset
def create_and_save_datasets(starting_positions, trajectory_length, control_step, measurement_noise, train_size,
                             val_size, test_size, file_path, name):
    n_samples = train_size + val_size + test_size
    true_trajectories = np.ndarray((n_samples, trajectory_length))
    measurements = np.ndarray((n_samples, trajectory_length))
    # record_writer_train = \
    #     tfr.RecordioWriter(file_path, train_size,
    #                        name + '_train_')
    # record_meta_train = tfr.RecordMeta(name + '_train_')
    #
    # record_writer_val = \
    #     tfr.RecordioWriter(file_path, val_size,
    #                        name + '_val_')
    # record_meta_val = tfr.RecordMeta(name + '_val_')
    #
    # record_writer_test = \
    #     tfr.RecordioWriter(file_path, test_size,
    #                        name + '_test_')
    # record_meta_test = tfr.RecordMeta(name + '_test_')

    for i in range(n_samples):
        theta = starting_positions[i]  # starting position
        for j in range(trajectory_length):
            true_trajectories[i][j] = theta % (2 * np.pi)
            measurements[i][j] = (theta + np.random.normal(0.0, measurement_noise, 1).item()) % (2 * np.pi)
            theta = theta + control_step

    measurements_ = tf.expand_dims(tf.convert_to_tensor(measurements, dtype=tf.float32), 2)
    ground_truth_ = tf.expand_dims(tf.convert_to_tensor(true_trajectories,dtype = tf.float32), 2)
    # measurements_ = measurements[..., np.newaxis]
    # ground_truth_ = true_trajectories[..., np.newaxis]
    train_path = file_path + '/' + name + '_train_' + str(train_size) + '.tfrecord'
    val_path = file_path + '/' + name + '_val_' + str(val_size) + '.tfrecord'
    test_path = file_path + '/' + name + '_test_' + str(test_size) + '.tfrecord'
    # train_data = {"input": measurements_[:train_size], "label": ground_truth_[:train_size]}
    # val_data = {"input": measurements_[train_size:train_size + val_size],
    #             "label": ground_truth_[train_size:train_size + val_size]}
    # test_data = {"input": measurements_[train_size + val_size:], "label": ground_truth_[train_size + val_size:]}
    # tfr.write_tfr(train_data, record_writer_train, record_meta_train, file_path)
    # tfr.write_tfr(val_data, record_writer_val, record_meta_val, file_path)
    # tfr.write_tfr(test_data, record_writer_test, record_meta_test, file_path)
    train_dataset = tf.data.Dataset.from_tensor_slices((measurements_[:train_size],ground_truth_[:train_size]))
    val_dataset = tf.data.Dataset.from_tensor_slices((measurements_[train_size:train_size + val_size],ground_truth_[train_size:train_size + val_size]))
    test_dataset = tf.data.Dataset.from_tensor_slices((measurements_[train_size + val_size:],ground_truth_[train_size + val_size:]))
    write_tfrecord(train_path, train_dataset)
    write_tfrecord(val_path, val_dataset)
    write_tfrecord(test_path, test_dataset)

    return train_path, val_path, test_path


def load_data(data_dir, data_name,train_size, batch_size):
    """
    Collects the tf recod files that make up a dataset and returns the file
    lists for training, validation and testset

    Parameters
    ----------
    data_dir : str
        Path of the folder containing the tf record files
    data_name : str
        Identifyer of the dataset

    Returns
    -------
    train_files : list of str
        List with the filepaths of the tf record files in the training split
    val_files : list of str
        List with the filepaths of the tf record files in the validation split
    test_files : list of str
        List with the filepaths of the tf record files in the test split

    """
    files = os.listdir(data_dir)
    train_path = [os.path.join(data_dir, f) for f in files
                               if f.startswith(data_name + '_') and
                                                  '_train_' in f and '.tfrecord' in f]
    val_path = [os.path.join(data_dir, f) for f in files
                                 if f.startswith(data_name + '_') and
                                                  '_val_' in f and '.tfrecord' in f]
    test_path = [os.path.join(data_dir, f) for f in files
                                      if f.startswith(data_name + '_') and
                                                        '_test_' in f and '.tfrecord' in f]
    train_dataset = read_tfrecord(train_path)
    val_dataset = read_tfrecord(val_path)
    test_dataset = read_tfrecord(test_path)

    train_set = train_dataset.shuffle(train_size).batch(batch_size, drop_remainder=True)
    val_set = val_dataset.batch(batch_size, drop_remainder=True)

    test_set = test_dataset.batch(batch_size, drop_remainder=True)

    return train_set, val_set, test_set


def run_example(filter_type, out_dir, batch_size, grid_size, trajectory_length, motion_noise,
                measurement_noise, train_size, initial_cov, epochs, n_traj, learning_rate, learned_process_model,
                learned_measurement_model, data_dir):
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
    print(train_dir)
    # data_dir = os.path.join(out_dir + '/data')
    fig_dir = os.path.join(out_dir, uuid + time_ + '/fig')
    video_dir = os.path.join(out_dir, uuid + time_ + '/video')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    debug = False
    context = S1ToyContext(batch_size, filter_type, grid_size, motion_noise, measurement_noise,
                           learned_process_model, learned_measurement_model)

    model = FilterApplication(context, filter_type, initial_cov, debug)
    val_size = 72
    test_size = 30
    control_step = 0.3

    n_samples = train_size + val_size + test_size
    starting_positions = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    name = "simple_linear_data"
    file_path = os.path.join(data_dir, 's1_data')
    #pdb.set_trace()
    if os.listdir(file_path) == []:
        train_path,val_path,test_path = create_and_save_datasets(starting_positions, trajectory_length, control_step, measurement_noise, train_size,
                                 val_size, test_size, file_path, name)
        train_set, val_set, test_set = load_data(file_path, name, train_size, batch_size)
    else:
        train_set, val_set, test_set = load_data(file_path, name, train_size, batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    custom_step = 0
    for epoch in range(epochs):
        print("\nStart of epoch %d \n" % (epoch))
        print("Validating ...")
        dict_val = evaluate(model, val_set, epoch, "validate", batch_size, trajectory_length, n_traj, control_step, motion_noise, fig_dir)
        running_loss = 0
        for (x_batch_train, y_batch_train) in train_set:
            print("Training...")
            start = time.time()
            control = tf.ones_like(x_batch_train) * control_step +  tf.random.normal(x_batch_train.shape,0,motion_noise)
            input_ = (x_batch_train, control, y_batch_train[:, 0])

            with tf.GradientTape() as tape:
                out = model(input_)
                loss_value, metrics, metric_names = \
                    model.context.get_loss(x_batch_train, y_batch_train, out)

                running_loss += loss_value.numpy().item()
                if tf.math.reduce_any(tf.math.is_nan(loss_value)):
                    pdb.set_trace()
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

        train_loss = running_loss / train_size
        dict_epoch = {"epoch": epoch,
                      "train_loss": train_loss}
        dict_epoch.update(dict_val)
        wandb.log(dict_epoch)

    # print(model.summary())

    # test the trained model on the held out data
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    print("\n Testing")
    out_analytical_hef = lambda x, y: run_hef_analytical(x, y, control_step, grid_size, motion_noise,
                                                         measurement_noise, initial_cov)
    test_dict = evaluate(model, test_set,0, "test", batch_size, trajectory_length, n_traj, control_step, motion_noise,fig_dir,
                         out_analytical_hef)

    generate_video_from_images(fig_dir,video_dir, 1)

    wandb.log(test_dict)
    wandb.finish()

def generate_video_from_images(image_folder,video_folder_path, fps):
    if not os.path.exists(output_video_folder):
        os.makedirs(output_video_folder)
    # Get list of images in the folder
    image_prefix_list = [f.replace(f.split("_")[-1],"") for f in os.listdir(image_folder) if f.endswith('.png')]
    for image_prefix in image_prefix_list:
        generate_video_from_images(image_folder, video_folder_path, image_prefix, fps)


def run_hef_analytical(measurements, poses, step, grid_size, motion_noise, measurement_noise, initial_cov):
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
    measurements = measurements.numpy()
    poses = poses.numpy()
    batch_size = measurements.shape[0]
    trajectory_length = measurements.shape[1]
    grid = np.linspace(0, 2 * np.pi, grid_size, dtype=np.float64, endpoint=False)[np.newaxis, ...]
    grid_batched = np.tile(grid, [batch_size, 1])
    # grid_batched_reshape = np.reshape(grid_batched, (batch_size, 1, grid_size))
    fft = S1FFT(bandwidth=grid_size, oversampling_factor=2)

    simulator = S1Simulator(step=np.ones((batch_size, 1, 1)) * step, theta_initial=poses[:, 0][..., np.newaxis],
                            samples=grid_batched, fft=fft,
                            motion_noise=motion_noise, measurement_noise=measurement_noise)

    prior = S1Gaussian(mu_theta=poses[:, 0][..., np.newaxis], cov=initial_cov, samples=grid_batched, fft=fft)
    filter = BayesFilter(distribution=S1, prior=prior)
    posterior_list = []
    pred_list = []
    measurement_list = []
    for iter in range(trajectory_length):
        pred_list.append(filter.prediction(motion_model=simulator.motion()).energy)
        posteriori_hat_, measurement = filter.update(
            measurement_model=simulator.measurement(measurement=measurements[:, iter][..., np.newaxis]))
        posterior_list.append(posteriori_hat_.energy)
        measurement_list.append(measurement.energy)
    posteriori_distribution = np.transpose(np.stack(posterior_list), [1, 0, 2]).astype(np.float64)
    measurement_distribution = np.transpose(np.stack(measurement_list), [1, 0, 2]).astype(np.float64)
    belief_prediction = np.transpose(np.stack(pred_list), [1, 0, 2]).astype(np.float64)

    return posteriori_distribution, measurement_distribution, belief_prediction


def reset_weights(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  # if you're using a model as a layer
            reset_weights(layer)  # apply function recursively
            continue

        # where are the initializers?
        if hasattr(layer, 'cell'):
            init_container = layer.cell
        else:
            init_container = layer

        for key, initializer in init_container.__dict__.items():
            if "initializer" not in key:  # is this item an initializer?
                continue  # if no, skip it

            # find the corresponding variable, like the kernel or the bias
            if key == 'recurrent_initializer':  # special case check
                var = getattr(init_container, 'recurrent_kernel')
            else:
                var = getattr(init_container, key.replace("_initializer", ""))

            var.assign(initializer(var.shape, var.dtype))


def evaluate(model, dataset, epoch,  type, batch_size, trajectory_length, control_step,motion_noise, folder_name=None,
             out_analytical_hef=None):
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
        control = tf.ones_like(x_batch) * control_step +  tf.random.normal(x_batch.shape,0,motion_noise)
        input_ = (x_batch, control, y_batch[:, 0])
        out = model(input_, training=False)
        plotting_dict['hef_diff'] = out

        loss_value, metrics, metric_names = \
            model.context.get_loss(x_batch, y_batch, out)
        if type == "test":
            out_analytical_hef_ = out_analytical_hef(x_batch, y_batch)
            plotting_dict["hef_analytical"] = out_analytical_hef_
            loss_value_analytical, metrics_analytical, metric_names_analytical = \
                model.context.get_loss(x_batch, y_batch, out_analytical_hef_)
            if step == 0:
                for ind, k in enumerate(metric_names_analytical):
                    outputs_analytical[k] = [metrics_analytical[ind]]
            else:
                for ind, k in enumerate(metric_names_analytical):
                    outputs_analytical[k].append(metrics_analytical[ind])
        plotting_figure(plotting_dict, y_batch, x_batch, batch_size, folder_name, step, trajectory_length,
                        type,epoch,frequency_iter=3)

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
        print(k, ": ", tf.reduce_mean(outputs[k]))
        dict[type + "/" + k] = tf.reduce_mean(outputs[k])
    print('\n')

    if type == "test":
        for ind, k in enumerate(metric_names_analytical):
            print("hef analytical ", k, ": ", tf.reduce_mean(outputs_analytical[k]))
            dict["hef analytical" + type + "/" + k] = tf.reduce_mean(outputs_analytical[k])

    return dict


def plotting_figure(dict_predictions, label, input, batch_size, folder_name, batch_no, trajectory_length,
                    type,epoch,frequency_iter=3):
    # p db.set_trace()
    plotting_idx = random.sample(range(0, batch_size), 1)
    measurement = label
    trajectory = input
    #plt.style.use('seaborn-dark-palette')

    for traj in range(trajectory_length):
            if traj % frequency_iter == 0:
                fig, ax = plt.subplots()
                for key, outputs in dict_predictions.items():
                    posterior_state, z_pred, pred_state = outputs
                    ax = plot_s1_energy(
                        [posterior_state[plotting_idx, traj], z_pred[plotting_idx, traj], pred_state[plotting_idx, traj]],
                        ax=ax, legend=[rf'{key}_posterior', rf'{key}_measurement', rf'{key}_prediction'])
                ax.plot(tf.math.cos(trajectory[plotting_idx, traj]), tf.math.sin(trajectory[plotting_idx, traj]), 'o',
                        label="pose data")
                ax.plot(tf.math.cos(measurement[plotting_idx, traj]), tf.math.sin(measurement[plotting_idx, traj]), 'o',
                        label="measurement data")
                ax.set_title(f"Trajectory {plotting_idx} Iteration {traj}", loc='center')
                ax.legend(bbox_to_anchor=(0.85, 1), loc='upper left', fontsize='x-small')
                plt.savefig(f"{folder_name}/s1_hef_{type}_{epoch}_{batch_no}_traj_{plotting_idx}_iter{traj}.png", format='png', dpi=300)
                plt.close()


class FilterApplication(tf.keras.Model):
    def __init__(self, context, filter_type='ekf', initial_cov=0.1, debug=False, **kwargs):
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
        self.context = context
        self.grid_size = self.context.grid_size
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
        raw_observations, control, init_pose = inputs
        mu = self.theta_to_2D(init_pose)
        tensor_start = tf.constant(0, dtype=tf.float32)
        tensor_stop = tf.constant(2 * math.pi, dtype=tf.float32)
        samples_ = tf.expand_dims(tf.linspace(tensor_start, tensor_stop, self.grid_size), 0)
        samples_batched = tf.tile(samples_, [self.batch_size, 1])

        init_state = (tf.reshape(tf.convert_to_tensor(self.compute_energy(samples_batched, mu)),
                                 [self.batch_size, -1]), tf.zeros([self.batch_size, 1]))

        inputs = (raw_observations, control)

        outputs = self.rnn_layer(inputs, training=training, initial_state=init_state)

        return outputs


def main():
    parser = argparse.ArgumentParser('run example')
    parser.add_argument('--out_dir', dest='out_dir', type=str, default='output', help='where to store results')
    parser.add_argument('--filter', dest='filter', type=str,
                        default='hef',
                        help='which filter class to use')
    parser.add_argument('--batch_size', dest='batch_size',
                        type=int, default=16, help='batch size for training')
    parser.add_argument('--grid_size', dest='grid_size', type=int,
                        default=30,
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
                        default=100,
                        help='length of the training dataset')
    parser.add_argument('--initial_cov', dest='initial_cov', type=float,
                        default=0.1,
                        help='noise around the starting state fed as a prior to the model')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=20,
                        help='no of times the model is trained for the complete dataset')
    parser.add_argument('--seed', dest='seed',
                        type=int, default=12345,
                        help='argument for the randomness')
    parser.add_argument('--n_traj', dest='n_traj',
                        type=int, default=5,
                        help='number of trajectories to plot per batch for evaluation')
    parser.add_argument('--learning_rate', dest='learning_rate',
                        type=float, default=1e-5,
                        help='learning rate of the neural network')
    parser.add_argument('--learned_process_model', dest='learned_process_model',
                        type=int, choices=[0, 1], default=0,
                        help='flag set to true will learn the process model')
    parser.add_argument('--learned_measurement_model', dest='learned_measurement_model', type=int, choices=[0, 1],
                        default=1)
    parser.add_argument('--data_dir', dest='data_dir',default='data', type=str, help='path to data directory')

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
    print(f"Random seed set as {seed}")

    # wandb.init(config=args)
    wandb.init(
        project="differential-hef",
        entity="korra141",
        tags=["version_4"],
        config=args
    )
    # code = wandb.Artifact('project-source', type='code')
    # for path in glob.glob(base_path + '**/*.py', recursive=True):
    #     code.add_file(path)
    # wandb.run.use_artifact(code)

    run_example(args.filter,args.out_dir, args.batch_size, args.grid_size, args.trajectory_length,
                args.motion_noise, args.measurement_noise, args.train_size, args.initial_cov, args.epochs, args.n_traj,
                args.learning_rate, args.learned_process_model, args.learned_measurement_model, args.data_dir)


if __name__ == "__main__":
    main()
