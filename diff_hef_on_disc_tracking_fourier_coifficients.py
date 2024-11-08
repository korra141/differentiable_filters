import tensorflow as tf
# print(tf.__version__)
import numpy as np
import tensorflow_probability as tfp
import math
import time
import os
import pdb
import sys
sys.path.append(os.getcwd())
from differentiable_filters.data.create_disc_tracking_dataset import DiscTrackingData
from differentiable_filters.example_training_code.run_example_old import load_data
from differentiable_filters.utils import recordio as tfr
import logging

def check_path_exists(directory_path,substring):
	# Replace with the actual directory path where your files are located.
	for filename in os.listdir(directory_path):
		if filename.startswith(substring):
			filepath = os.path.join(directory_path, filename)
			if os.path.exists(filepath):
				print(f"File found: {filepath}")
				return True
	print(f"No file found with a name starting with '{substring}' in '{directory_path}'")
	return False

def main(): 
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    pos_noise = 0.1
    num_distractors = 5
    name = 'disc_tracking_pn=' + str(pos_noise) + '_d=' + str(num_distractors) + '_hetero'
    width = 120
    num_examples = 2000
    file_size = 500
    debug = 0
    hetero_q = 1
    correlated_q = 0

    out_dir = 'data/disc_tracking_dataset/'
    sequence_length = 50
    if not check_path_exists(out_dir,name):
        # print(os.path.join(out_dir,name))
        c = DiscTrackingData(name, out_dir, width, num_examples,
                                         sequence_length, file_size, debug)
        c.create_dataset(num_distractors, hetero_q,correlated_q, pos_noise)
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    batch_size = 20
    motion_noise = 0.1
    measurement_noise = 0.1
    learned_process_model = 1
    bandlimit=(10,10,2,2)
    image_size = 120
    trajectory_length = 20

    context = DiscTrackingContext(batch_size, trajectory_length, bandlimit,motion_noise, measurement_noise, learned_process_model,image_size)
    model = FilterApplication(context, debug=False)

    train_files, val_files, test_files = load_data(out_dir, name)
    train_set = tf.data.TFRecordDataset(train_files)
    train_set = context.preprocess(out_dir, name, train_set, 'train', trajectory_length)
    train_set = train_set.shuffle(500)
    train_set = train_set.batch(batch_size, drop_remainder=True)

    val_set = tf.data.TFRecordDataset(val_files)
    val_set = context.preprocess(out_dir, name, val_set, 'val',trajectory_length )
    val_set = val_set.batch(batch_size, drop_remainder=True)

    test_set = tf.data.TFRecordDataset(test_files)
    test_set = context.preprocess(out_dir, name, test_set, 'test', trajectory_length)
    test_set = test_set.batch(batch_size, drop_remainder=True)
    learning_rate=1e-3
    epochs = 5
    step = 0

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for epoch in range(epochs):
        print("\nStart of epoch %d \n" % (epoch))
        # evaluate(model, val_set, batch_size)
        for i, (x_batch_train, y_batch_train) in enumerate(train_set):
            start = time.time()
            with tf.GradientTape() as tape:
                # sample a random disturbance of the initial state from the
                # initial covariance
                n_val = np.random.normal(loc=np.zeros((model.dim_x)),
                                          scale=model.initial_covariance,
                                          size=(batch_size, model.dim_x))
                x_batch_train = (*x_batch_train, n_val)
                out = model(x_batch_train, training=True)
                print(f"posterior {out[0]}, measurement likelihood {out[1]}, prediction {out[2]}\n\n")

                # Compute the loss value for this minibatch.
                loss_value, metrics, metric_names = \
                    model.context.get_loss(x_batch_train, y_batch_train, out)
                
                # if i % 20 == 0:
                print(f"Loss at batch {i}, epoch {epoch} is {loss_value}")


            # Use the gradient tape to automatically retrieve the
            # gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            end = time.time()

            # Log every 50 batches.
            if step % 50 == 0:
                print("Training loss at step %d: %.4f (took %.3f seconds) " %
                      (step, float(loss_value), float(end-start)))
            step += 1

    # test the trained model on the held out data
    print("\n Testing with sequence length 30")
    #evaluate(model, test_set, batch_size)


# Predict the observation in latent space from the predicted belief

class ObservationModel(tf.keras.Model):
    def __init__(self,dim_z):
        super(ObservationModel, self).__init__()
        self.dim_z = dim_z
        # Dense layers
        self.dense1 = tf.keras.layers.Dense(128, activation='relu',kernel_initializer=tf.initializers.HeUniform(),
                # kernel_initializer='random_normal',
                bias_initializer='zeros')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu',kernel_initializer=tf.initializers.HeUniform(),
                # kernel_initializer='random_normal',
                bias_initializer='zeros')
        self.flatten = tf.keras.layers.Flatten()
        # Output layer: coefficients of the distribution (example: mean and variance)
        self.output_layer = tf.keras.layers.Dense(self.dim_z)  # Output 2 values (mean and variance) - adjust as needed


    def call(self, inputs,training):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

# prompt: Write a class ProcessModel that takes in input as distribution with size grid_size and outputs the distribution in fourier coifficients


class ProcessModel(tf.keras.Model):
    def __init__(self, bandlimit,dimx):
        super(ProcessModel, self).__init__()
        self.output_units = math.prod(bandlimit)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu',kernel_initializer=tf.initializers.HeUniform(),
                # kernel_initializer='random_normal',
                bias_initializer='zeros')
        self.dense2 = tf.keras.layers.Dense(512, activation='relu',kernel_initializer=tf.initializers.HeUniform(),
                # kernel_initializer='random_normal',
                bias_initializer='zeros')
        self.dense3 = tf.keras.layers.Dense(self.output_units)
        self.reshape = tf.keras.layers.Reshape(bandlimit)


    def call(self, inputs,training):
        # pdb.set_trace()
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        out = self.reshape(x)
        return out

class SensorModel(tf.keras.Model):
    def __init__(self, dim_z):
        super(SensorModel, self).__init__()
        self.dim_z = dim_z

    def build(self, input_shape):
        self.sensor_conv1 = tf.keras.layers.Conv2D(
            filters=4,
            kernel_size=5,
            strides=[2, 2],
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l2=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l2=1e-3),
            name='sensor_conv1')
        self.sensor_conv2 = tf.keras.layers.Conv2D(
            filters=8, kernel_size=3,
            strides=[2, 2],
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l2=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l2=1e-3),
            name='sensor_conv2')

        self.flatten = tf.keras.layers.Flatten()

        self.sensor_fc1 = tf.keras.layers.Dense(
            units=16,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l2=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l2=1e-3),
            name='sensor_fc1')
        self.sensor_fc2 = tf.keras.layers.Dense(
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l2=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l2=1e-3),
            name='sensor_fc2')
        self.sensor_fc3 = tf.keras.layers.Dense(
            units=2,
            kernel_initializer=tf.initializers.glorot_normal(),
            kernel_regularizer=tf.keras.regularizers.l2(l2=1e-3),
            bias_regularizer=tf.keras.regularizers.l2(l2=1e-3),
            name='sensor_fc3',
            activation=None)

    def call(self, images, training):
        conv1 = self.sensor_conv1(images)
        conv1 = tf.nn.max_pool2d(conv1, 2, 2, padding='SAME')
        conv2 = self.sensor_conv2(conv1)
        #conv2 = tf.nn.max_pool2d(conv2, 2, 2, padding='SAME')

        input_data = self.flatten(conv2)
        fc1 = self.sensor_fc1(input_data)
        fc2 = self.sensor_fc2(fc1)
        pos = self.sensor_fc3(fc2)

        return pos, fc2

class ObservationLikelihoodNetwork(tf.keras.Model):
    def __init__(self,bandlimit,dimx):
        self.output_units = math.prod(bandlimit)
        super(ObservationLikelihoodNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu',kernel_initializer=tf.initializers.HeUniform(),
                # kernel_initializer='random_normal',
                bias_initializer='zeros')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu',kernel_initializer=tf.initializers.HeUniform(),
                # kernel_initializer='random_normal',
                bias_initializer='zeros')
        self.dense3 = tf.keras.layers.Dense(self.output_units)
        self.reshape = tf.keras.layers.Reshape(bandlimit)
    def call(self, predicted_obs, encoded_obs,training):
        # Concatenate predicted and encoded observations
        combined_input = tf.concat([predicted_obs, encoded_obs], axis=-1)
        x = self.dense1(combined_input)
        x = self.dense2(x)
        x = self.dense3(x)
        out = self.reshape(x)
        # Optionally, apply a sigmoid to constrain output to [0,1] if you want probability
        # likelihood = tf.keras.activations.sigmoid(log_likelihood)
        return out

class DiscTrackingContext(tf.keras.Model):
    def __init__(self, batch_size, trajectory_length, bandlimit,motion_noise, measurement_noise, learned_process_model,image_size):
        super().__init__()
        self.batch_size = batch_size
        self.trajectory_length = trajectory_length
        self.learned_process_model = learned_process_model
        # # parameters of the process model
        self.bandlimit = bandlimit
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        self.dim_x = 4
        self.dim_z = 32
        self.dim_u = None
        self.observation_model = ObservationModel(self.dim_z)
        if learned_process_model:
            self.process_model = ProcessModel(self.bandlimit,self.dim_x)
        self.sensor_model = SensorModel(self.dim_z)
        self.observation_likelihood = ObservationLikelihoodNetwork(self.bandlimit,self.dim_x)
        self.image_size = image_size

    def run_observation_model(self, observations, training):
        """
        Predicts the observations for a given state

        Parameters
        ----------
        state : tensor [batch_size (x number of particles/sigma points), dim_x]
            the predicted state
        training : boolean tensor
            flag that indicates if model is in training or test mode

        Returns
        -------
        tf.keras.layer
            A layer that computes the expected observations for the input
            state and the Jacobian  of the observation model

        """
        # pdb.set_trace()
        # if self.learned_measurement_model:
            # joint_input = tf.keras.layers.Concatenate(axis=1)([state, observations])
        out = self.observation_model(observations, training=training)


        # else:
        #     mean = observations
        #     cov = tf.ones_like(observations) * (self.measurement_noise ** 2)

        return out

    def run_sensor_model(self, raw_observations, training):
        """
        Process raw observations and return the predicted observations z
        for the filter and an encoding for predicting the observation noise

        Parameters
        ----------
        raw_observations : list of tensors
            Raw sensory observations
        training : boolean tensor
            flag that indicates if model is in training or test mode

        Returns
        -------
        z : tensor [batch_size, dim_z]
            Low-dimensional observations
        enc : tensor [batch_size, 32]
            An encoding of the raw observations that can be used for predicting
            heteroscedastic observation noise
        """
        return self.sensor_model(raw_observations, training=training)

    ###########################################################################
    # process model
    ###########################################################################
    def run_process_model(self, old_state, training):
        """
        Predicts the next state given the old state and actions performed

        """
        out = self.process_model(old_state, training=training)

        return out
    ###########################################################################
    # observation likelihood model
    ###########################################################################

    def likelihood_model(self,z_pred_energy_predicted,encoding,training):

        return self.observation_likelihood(z_pred_energy_predicted,encoding,training=training)


    ###########################################################################
    # loss functions
    ###########################################################################
    def get_loss(self, data, label, prediction):

        """
        Compute the loss for the filtering application - defined in the context

        Args:
            prediction: list of predicted tensors
            label: list of label tensors

        Returns:
            loss: the total loss for training the filtering application
            metrics: additional metrics we might want to log for evaluation
            metric-names: the names for those metrics
        """
        posterior_dist_parameters, measurement_dist_parameters , predicted_dist_parameters = prediction
        # print("values in label data ",len(label))
        pose, prrocess_noise, visible_pixels_td = label

        # print("values in input data", len(data))
        observations, initial_state, noise_initial_data = data

        nll_posterior = self.neg_log_likelihood(posterior_dist_parameters, pose, self.bandlimit)
        # nll_likelihood = self.neg_log_likelihood(measurement_dist_parameters, observations, self.bandlimit)
        nll_pred = self.neg_log_likelihood(predicted_dist_parameters, pose, self.bandlimit)

        # compute the mode of the distribution
        # mode_pose_posterior = self.compute_mode_(posterior_dist_parameters)
        # mode_pose_pred = self.compute_mode_(predicted_dist_parameters)
        # mode_obs = self.compute_mode_(measurement_likelihood_dist_parameters)

        # # diff_mode_pose_posterior = tf.math.minimum(
        # #     2 * math.pi * tf.ones_like(pose) - tf.math.abs(mode_pose_posterior - pose),
        # #     tf.math.abs(mode_pose_posterior - pose))
        # diff_mode_pose_pred = tf.math.minimum(2 * math.pi * tf.ones_like(pose) - tf.math.abs(mode_pose_pred - pose),
        #                                       tf.math.abs(mode_pose_pred - pose))
        # # diff_mode_obs = tf.math.minimum(2 * math.pi * tf.ones_like(pose) - tf.math.abs(mode_obs - pose),
        # #                                 tf.math.abs(mode_obs - pose))

        # ate_mode_post = self.average_traj_error(diff_mode_pose_posterior)
        # ate_mode_pred = self.average_traj_error(diff_mode_pose_pred)
        # ate_mode_meas = self.average_traj_error(diff_mode_obs)

        # mae_mode_post = tf.reduce_mean(diff_mode_pose_posterior)
        # mae_mode_pred = tf.reduce_mean(diff_mode_pose_pred)
        # mae_mode_meas = tf.reduce_mean(diff_mode_obs)

        # total_loss = nl_loss_posterior

        # TODO: get the weight decay

        # wd = []
        # for la in self.layers:
        #     wd += la.losses
        # wd = tf.add_n(wd)


        total = nll_posterior

        # metrics = [total, nll_pred, nll_posterior, nll_likelihood, ate_mode_post, ate_mode_pred, ate_mode_meas,
        #            mae_mode_post, mae_mode_pred, mae_mode_meas]
        # metric_names = ["total", "nll_pred", "nl_loss_posterior", "nl_loss_measurement", "ate_mode_post", "ate_mode_pred",
        #                 "ate_mode_meas",
        #                 "mae_mode_post", "mae_mode_pred", "mae_mode_meas"]
        metrics = [total, nll_pred, nll_posterior]
        metric_names = ["total", "nll_pred", "nl_loss_posterior"]
        return total, metrics, metric_names

    def neg_log_likelihood(self, density_coiff, true_value, grid_size):
        """
              Compute the negative log likelihood loss over Harmonic Exponential Distribution for trajectory. This function would need to be called over a batch.

              Args:
                  energy_samples: list of predicted tensors for one pose
                  true_value: ground truth value for the variable being estimated, need to be complex

              Returns:
                  loss: returns a scalar value for negative log likelihood of the Harmonic Exponential Distribution for the ground truth over the trajectory.

        """
        
        eta = tf.cast(density_coiff, tf.complex64)  # [bandlimit,trajectory_length,dimx,bandlimit]
        eta_flattend = tf.reshape(eta,[eta.shape[0],eta.shape[1],-1])
        # calculating the normalisation constat 
        energy_samples = tf.signal.ifft(eta_flattend)
        maximum = tf.expand_dims(tf.math.reduce_max(tf.math.real(energy_samples), axis=-1), -1)
        maximum = tf.cast(maximum, dtype=tf.complex64)
        moments = tf.signal.fft(tf.exp(energy_samples - maximum))
        ln_z_ = tf.cast(tf.math.real(tf.math.log(moments[..., 0] / (math.pi * math.prod(self.bandlimit) * math.pi / 62)) + maximum),
            dtype=tf.float32) 
        # pdb.set_trace()
        # ln_z_ = tf.reshape(ln_z_, [ln_z_.shape[0],ln_z_.shape[1]] + list(self.bandlimit))
        
        # print("shape of the normalisation constant: ",ln_z_.shape )
        print("mean normalisation constant: ", tf.reduce_mean(ln_z_))


        k_values = [tf.expand_dims(tf.range(0, limit, dtype=tf.float32) - limit // 2,0) for limit in self.bandlimit] #list([1,bandlimit])
        value = true_value
        temp_value = tf.zeros([self.batch_size, self.trajectory_length,1,1,1,1],dtype=tf.complex64)
        size_bandlimit = len(self.bandlimit)
        identity_matrix = tf.ones((size_bandlimit,size_bandlimit))
        size_ = tf.linalg.set_diag(identity_matrix, self.bandlimit)
        # pdb.set_trace()
        for i in range(size_bandlimit):
          temp_value = temp_value * tf.cast(tf.reshape(tf.reshape(value[:,:,i],[-1,1])* k_values[i],[self.batch_size, self.trajectory_length] + list(size_[i])),dtype=tf.complex64)/self.bandlimit[i]
        exponential_term = tf.math.exp(tf.constant(2*math.pi*1j, dtype=tf.complex64) * temp_value) # [batch_size, trajectory_length,bandlimit]
        inverse_transform = tf.math.real(tf.math.reduce_sum(eta * exponential_term ,axis=[2,3,4,5]))/math.prod(self.bandlimit)  # [batch_size, trajectory_length,  1]
        # pdb.set_trace()
        nll = tf.reduce_mean(tf.cast(-inverse_transform + ln_z_, dtype=tf.float32))
        return nll
    ###########################################################################
    # data loading
    ###########################################################################
    def preprocess(self, path, name, dataset, data_mode, sl, num_threads=1):
        """
        Converts from tf.Records to tensors and applys preprocessing

        Parameters
        ----------
        path : str
            Path to the directory that contains the tf.Record files
        name : str
            The name of the dataset.
        dataset : tf.data.Dataset
            Atf.data.TFRecordDataset object that contains the filenames of all
            tf.Records to read in.
        data_mode : str
            Defines for which data split the data is read. Can be "train",
            "val" or "test"
        sl : int
            Desired sequence length. This enables extraction of shorter
            subsequences from the original sequences.
        num_threads : int, optional
            The number of threads used to preeprocess the data. The default is
            1.

        Returns
        -------
        dataset : tf.data.Dataset
            A dataset with input and label tensors.

        """
        keys = ['start_state', 'image', 'state', 'q', 'visible']
        record_meta = tfr.RecordMeta.load(path, name + '_' + data_mode + '_')

        dataset = \
            dataset.map(lambda x: self._parse_function(x, keys, record_meta,
                                                       sl, data_mode),
                        num_parallel_calls=num_threads)
        dataset = dataset.flat_map(lambda x, y:
                                   tf.data.Dataset.from_tensor_slices((x, y)))
        return dataset

    def _decode_im(self, im):
        """
        Decode an image from a byte-string.

        Parameters
        ----------
        im : tf.String tensor
            Byte string that encodes one image

        Returns
        -------
        im : tf.float32 tensor
            The decoded image

        """
        im = tf.io.decode_image(im, channels=3, dtype=tf.dtypes.uint8)
        im = tf.cast(im, tf.float32) / 255.
        im.set_shape([self.image_size, self.image_size, 3])
        return im

    def _parse_function(self, example_proto, keys, record_meta, sl, data_mode):
        """
        This function defines how a single example from the tf.Recod data is
        parsed into tensors. Also applies preprocessing.

        Parameters
        ----------
        example_proto : tf.train.Example
            A single example from the dataset
        keys : list of str
            The names of the data that is contained in each example
        record_meta : tfr.RecordMeta
            An object that holds meta information about the the datase, such
            as tensor shapes and data types
        sl : int
            Desired sequence length. This enables extraction of shorter
            subsequences from the original sequences.
        data_mode : str
            Defines for which data split the data is read. Can be "train",
            "val" or "test"

        Raises
        ------
        ValueError
            If the desired sequence length (sl) is longer than the lenght of
            the sequences in the dataset

        Returns
        -------
        inputs : tuple of tensors
            The tensors that serve as input for the differentiable filter. In
            this case the sequence of image observations and the inital state
            of the system
        labels : tuple of tensors
            The labels. In this case the true state sequence, the true
            process nosie covariance and the number of visible target disc
            pixels in each image.

        """
        features = {}
        for key in keys:
            record_meta.add_tf_feature(key, features)

        parsed_features = tf.io.parse_single_example(example_proto,
                                                     features)
        for key in keys:
            features[key] = record_meta.reshape_and_cast(key,
                                                         parsed_features)
        state = features['state']
        start = features['start_state']
        q = features['q']
        vis = features['visible']

        length = q.get_shape()[0]

        # reading in images from bytes
        im = features['image']
        im = tf.map_fn(self._decode_im, im, fn_output_signature=tf.float32)

        if sl > length:
            raise ValueError('Desired training sequence length is ' +
                             'longer than dataset sequence length: ' +
                             'Desired: ' + str(sl) + ', data: ' +
                             str(length))

        if sl == length or data_mode == 'test':
            start_inds = [-1]
        else:
            # we use several sub-sequences of the full sequence
            num = length // sl
            start_inds = \
                np.arange(-1, length-sl-1, (sl+1)//2)
            start_inds = start_inds[:num]

        # prepare the lists of output tensors
        ims = []
        starts = []
        states = []
        qs = []
        viss = []
        for si in start_inds:
            if si >= 0:
                starts += [state[si]]
            else:
                starts += [start]
            end = si+sl+1
            ims += [im[si+1:end]]
            states += [state[si+1:end]]
            qs += [q[si+1:end]]
            viss += [vis[si+1:end]]

        # observations, initial state,
        inputs =  tuple([tf.stack(ims), tf.stack(starts)])
        labels =  tuple([tf.stack(states), tf.stack(qs), tf.stack(viss)])
        return inputs, labels

"""
RNN cell implementing a Differentiable Extended Kalman Filter
"""

from differentiable_filters.filters import filter_cell_base as base

class HEFCell(base.FilterCellBase):
    def __init__(self, context, problem, bandlimit, update_rate=1, debug=False):
        """
        RNN cell implementing a Differentiable Harmonic Exponential Filter

        Parameters
        ----------
        context : tf.keras.Model
            A context class that implements all functions that are specific to
            the filtering problem (e.g. process and observation model)
        problem : str
            A string identifyer for the problem defined by the context
        update_rate : int, optional
            The rate at which observations come in (allows simulating lower
            observation rates). Default is 1
        debug : bool, optional
            If true, the filters will print out information at each step.
            Default is False.
        """
        base.FilterCellBase.__init__(self, context, problem, update_rate,
                                     debug)
        self.bandlimit = bandlimit
        self.batch_size = context.batch_size


    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        It can be represented by an Integer, a TensorShape or a tuple of
        Integers or TensorShapes.
        """
        # return [[1],[self.grid_size],[1]]
        return [self.bandlimit,[1]]

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        # [[1],[1],[self.grid_size],[self.grid_size],[self.grid_size],[self.grid_size]]
        return [self.bandlimit,self.bandlimit,self.bandlimit,self.bandlimit]
    def call(self, inputs, states, training):
        """
        The function that contains the logic for one RNN step calculation.

        Parameters
        ----------
        inputs : list of tensors
            the input tensors, which is a slice from the overall RNN input
            by the time dimension (usually the second dimension).
        states : list of tensors
            the state tensor from previous step as specified by state_size. In
            the case of timestep 0, it will be the
            initial state user specified, or zero filled tensor otherwise.
        training : bool
            if the cell is run in training or test mode

        Returns
        -------
        output : list of tensors
            output tensors as defined in output_size
        new_state : list of tensors
            the new predicted state as defined in state_size
        """
        # turn off the '/rnn' name scope to improve summary logging
        with (tf.name_scope("")):
            # get the inputs
            pdb.set_trace()
            observations = inputs
            prior, c_step = states
            
            process_parameters= self.context.run_process_model(prior,training)  
            pred_state_parameters = self._prediction_step(prior,process_parameters)


            measurement_noise, encoding = self.context.run_sensor_model(observations,
                                                        training=training)

            z_pred = self.context.run_observation_model(pred_state_parameters,
                                                           training=training)
            z_likelihood = self.context.likelihood_model(z_pred,encoding,training)

            ###################################################################
            # update the predictions with the observations

            state_up = self._update(pred_state_parameters, z_likelihood)
            state = tf.cast(tf.reshape(state_up, [self.batch_size] + list(self.bandlimit)),dtype=tf.float64)
            z_likelihood = tf.cast(tf.reshape(z_likelihood, [self.batch_size] + list(self.bandlimit)),dtype=tf.float64)
            pred_state_parameters = tf.cast(tf.reshape(pred_state_parameters, [self.batch_size] + list(self.bandlimit)),dtype=tf.float64)
            process_parameters = tf.cast(tf.reshape(process_parameters, [self.batch_size] + list(self.bandlimit)),dtype=tf.float64)


            new_state = (state, c_step + 1)

            output = (state,z_likelihood,pred_state_parameters)


            return output, new_state


    def _prediction_step(self,prior,process_parameters):
        #pdb.set_trace()

        prior_flatten = tf.reshape(prior,[self.batch_size,-1])
        energy_prior= tf.signal.irfft(tf.cast(prior_flatten,tf.complex64))
        ln_z_1 = self.calculate_normalisation_const(energy_prior)

        process_parameters_flatten = tf.reshape(process_parameters,[self.batch_size,-1])
        energy_process = tf.signal.irfft(tf.cast(process_parameters_flatten,tf.complex64))

        ln_z_2 = self.calculate_normalisation_const(energy_process)

        prob_1 = tf.math.exp(energy_prior - ln_z_1)
        prob_2 = tf.math.exp(energy_process - ln_z_2)

        m1 = tf.cast(tf.signal.rfft(prob_1),dtype=tf.complex64)
        m2 = tf.cast(tf.signal.rfft(prob_2),dtype=tf.complex64)

        m_conv = m1*m2

        eta = tf.math.real(self.convert_moments_eta(m_conv))


        return tf.reshape(eta,[self.batch_size,-1])


    def calculate_normalisation_const(self,energy):
        # import pdb;pdb.set_trace()
        maximum = tf.expand_dims(tf.math.reduce_max(energy, axis=-1), -1)
        moments = tf.signal.rfft(tf.exp(energy - maximum))
        ln_z_ = tf.expand_dims(tf.math.real(tf.math.log(moments[:, 0])), -1) + maximum

        return ln_z_


    def _update(self, prior_eta, measurement_model):
        measurement_model = tf.reshape(measurement_model,[self.batch_size,-1])
        eta_update =  prior_eta + measurement_model
        return self.eta_normalise(eta_update)

    def convert_moments_eta(self,moments):
        #pdb.set_trace()
        unorm_prob = tf.signal.irfft(tf.cast(moments,tf.complex64))
        ln_z_ = tf.expand_dims(tf.math.real(tf.math.log(moments[:, 0])), 1)
        unorm_prob_real = tf.math.real(unorm_prob)
        unorm_prob_real = tf.where(unorm_prob_real > 0, unorm_prob_real, 1e-8)
        norm_energy = tf.math.log(unorm_prob_real) - ln_z_
        eta = tf.signal.rfft(norm_energy)
        return eta

    def eta_normalise(self,eta):
        # Normalise this
        unnorm_energy = tf.signal.irfft(tf.cast(eta,tf.complex64))
        ln_z_ = self.calculate_normalisation_const(unnorm_energy)
        norm_energy = unnorm_energy - ln_z_
        eta_normalise = tf.signal.rfft(norm_energy)
        return eta_normalise



    def convert_from_energy_eta(self,energy_samples):
        eta = tf.signal.rfft(energy_samples,axes=-1)
        return eta


def multivariate_normal_density_batch(x, mean, covariance):
    """
    Compute the probability density of a multivariate normal distribution for multiple points.

    Args:
    - x: Tensor of shape [batch_size, 4], the points at which to evaluate the PDF.
    - mean: Tensor of shape [4], the mean vector of the distribution.
    - covariance: Tensor of shape [4, 4], the covariance matrix of the distribution.

    Returns:
    - A tensor of shape [batch_size], representing the probability densities for each point in x.
    """
    # Number of dimensions (4 for R4)
    d = tf.shape(mean)[-1]

    # Compute the inverse and determinant of the covariance matrix
    covariance_inv = tf.linalg.inv(covariance)
    covariance_det = tf.linalg.det(covariance)

    # Compute the difference vector (x - mean)
    diff = (x - mean)[:,:,None,:]  # Shape [batch_size, bandlimit_prod, 4]
    # Compute the quadratic form: (x - mean)T * inv(covariance) * (x - mean)
    # To handle batch processing, we use matrix multiplication
    quadratic_term = tf.matmul(tf.matmul(diff,covariance_inv),diff,transpose_b=True)  # Shape [batch_size]

    term = tf.cast(tf.squeeze(tf.squeeze(quadratic_term,-1),-1),dtype=tf.float64)

    # Compute the normalization factor: (2pi)^(d/2) * |covariance|^(1/2)
    normalization_factor = ((2 * tf.constant(math.pi, dtype=tf.float64)) ** (d / 2)) * tf.cast(tf.sqrt(covariance_det),tf.float64)

    max = tf.expand_dims(tf.reduce_max(term,axis=-1),-1)

    # Compute the PDF for each point in the batch
    pdf = tf.exp(-0.5 * (term-max)) / normalization_factor

    return tf.math.log(pdf) - max/2

class FilterApplication(tf.keras.Model):
    def __init__(self, context, debug=False, **kwargs):


        super(FilterApplication, self).__init__(**kwargs)

        # -------------------------- (1) --------------------------------------
        # Construct the context class that describes the problem on which
        # we want to run a differentiable filter
        #----------------------------------------------------------------------
        # -------------------------- (2) --------------------------------------
        # Instantiate the desired filter cell
        #----------------------------------------------------------------------
        self.context = context
        problem="disc"
        self.cell = HEFCell(self.context,problem, context.bandlimit,debug=debug)



        # -------------------------- (3) --------------------------------------
        # wrap the Filter cell in a keras RNN Layer
        # ---------------------------------------------------------------------
        self.rnn_layer = tf.keras.layers.RNN(self.cell, return_sequences=True,
                                             unroll=False)

        # store some shape related information
        self.batch_size = self.context.batch_size
        self.dim_x = self.context.dim_x
        self.dim_z = self.context.dim_z
        self.dim_u = self.context.dim_u
        self.image_size = self.context.image_size
        self.bandlimit = self.context.bandlimit

        # -------------------------- (4) --------------------------------------
        # Define the covariance matrix for the initial belief of the filter
        # ---------------------------------------------------------------------
        self.initial_covariance = np.array([10.0, 10.0, 5.0, 5.0])/ 60.
        self.initial_covariance = self.initial_covariance.astype(np.float32)
        covar_start = tf.square(self.initial_covariance)
        covar_start = tf.linalg.tensor_diag(covar_start)
        self.covar_start = tf.tile(covar_start[None,None, :, :],
                                   [self.batch_size, 1, 1, 1])



    def call(self, inputs, training=True):
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
        raw_observations, initial_state, noise = inputs
        # -------------------------- (1) --------------------------------------
        # Construct the initial state of the differentiable filter RNN
        # ---------------------------------------------------------------------
        initial_state += noise

        init_state = \
              (self.prior_eta(initial_state[:,None,:],
                self.covar_start),
                tf.zeros([self.batch_size, 1]))
        # -------------------------- (2) --------------------------------------
        # Run the filtering algorithm for the full sequence.
        # Inputs for all filters are the raw observations and the control
        # actions. Since we do not use a control input in this example, we
        # pass a vector of zeros instead.
        # ---------------------------------------------------------------------

        inputs = raw_observations

        outputs = self.rnn_layer(inputs, training=training,
                                 initial_state=init_state)

        return outputs
    def prior_eta(self,mean, covariance):
    #   dist = tfp.distributions.MultivariateNormalFullCovariance(mean ,covariance)
      x_tf = tf.cast(tf.linspace(- self.image_size//2 ,self.image_size//2,self.bandlimit[0]),tf.float32)
      y_tf = tf.cast(tf.linspace(- self.image_size//2 ,self.image_size//2 ,self.bandlimit[1]),tf.float32)
      v_x_tf = tf.cast(tf.linspace(-6,6,self.bandlimit[2]),tf.float32)
      v_y_tf = tf.cast(tf.linspace(-6,6,self.bandlimit[3]),tf.float32)
      x_mesh,y_mesh,v_x_mesh, v_y_mesh = tf.meshgrid(x_tf,y_tf,v_x_tf,v_y_tf)
      x = tf.tile(tf.reshape(tf.stack((x_mesh,y_mesh,v_x_mesh, v_y_mesh),axis=-1),[1,-1,4]),[self.batch_size,1,1])
      energy_samples = multivariate_normal_density_batch(x, mean, covariance)
      # print("energy_samples type",type(energy_samples))
      pdb.set_trace()
      eta_flatten = tf.signal.fft(tf.cast(energy_samples,tf.complex64))
      eta = tf.math.real(tf.reshape(eta_flatten,[self.batch_size] + list(self.bandlimit) ))
      print("prior eta",tf.reduce_max(eta))
      return eta



def evaluate(model, dataset, batch_size):
    """
    Evaluates the model on the given dataset (without training)

    Parameters
    ----------
    model : tf.keras.Model
        The model to evaluate
    dataset : tf.data.Dataset
        The dataset on which to evaluate the model
    batch_size : int
        The batch size used.

    Returns
    -------
    None.

    """
    outputs = {}
    # Iterate over the batches of the testset.
    # pdb.set_trace()
    for step, (x_batch, y_batch) in enumerate(dataset):
        # sample a random disturbance from the initial covariance
        # and add it to the intial state
        n_val = np.random.normal(loc=np.zeros((model.dim_x)),
                                 scale=model.initial_covariance,
                                 size=(batch_size, model.dim_x))
        x_batch = (*x_batch, n_val)

        # Run the forward pass of the layer.
        out = model(x_batch, training=False)

        # Compute the loss and metrics for this minibatch.
        loss_value, metrics, metric_names = \
            model.context.get_loss(x_batch, y_batch, out)

        if step == 0:
            for ind, k in enumerate(metric_names):
                outputs[k] = [metrics[ind]]
        else:
            for ind, k in enumerate(metric_names):
                outputs[k].append(metrics[ind])

    print('Result: ')
    for ind, k in enumerate(metric_names):
        tf.print(k, ": ", tf.reduce_mean(outputs[k]))
    print('\n')

if  __name__ == "__main__":
    main()
