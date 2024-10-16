# Creating context for toy example on S1.
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math
import pdb

from differentiable_filters.contexts import base_context as base


class S1ToyContext(base.BaseContext):
    def __init__(self, batch_size, filter_type, grid_size, motion_noise, loss,learned_process_model):
        """
        Minimal example context for the simulated disc tracking task on S1. A context
        contains problem-specific information such as the state size or process
        and sensor models.

        Parameters
        ----------
        batch_size : int
            The batch size used.
        filter_type : str
            Which filter is used with the context.
        loss : str
            Which loss function to use.
        hetero_q : bool
            Learn heteroscedastic process noise?
        hetero_r : bool
            Learn heteroscedastic observation noise?
        learned_process : bool
            Learn the process model or use an analytical one?
        rescale : float, optional
            A factor by which the state-space has been downscaled. Default is 1.
        mixture_likelihood : bool, optional
            Only used with the particle filter. If true, the particle
            distribution is approximated with a GMM for calculating the
            nll loss. Else, a single gaussian is used. Default is True
        mixture_std : float, optional
            Only used with the particle filter if mixture_likelihood is true.
            The fixed covariance used for the individual gaussians in the GMM.
            Default is 0.1
        """
        super(base.BaseContext, self).__init__()

        # mark which filter and loss function are used
        self.filter_type = filter_type

        # define the state size and name the components
        self.batch_size = batch_size
        self.learned_process_model = learned_process_model

        # # parameters of the process model
        self.grid_size = grid_size
        self.zeroth_freq_index = math.floor(self.grid_size / 2)
        self.motion_noise = motion_noise
        self.loss = loss

        self.observation_model = ObservationModel(self.batch_size, self.grid_size)
        if learned_process_model:
            self.process_model = ProcessModel(self.grid_size,self.batch_size)
        self.dim_x = None
        self.dim_z = None
        self.dim_u = None

    def run_observation_model(self, state, training):
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
        return self.observation_model(state, training)

    ###########################################################################
    # process model
    ###########################################################################
    def run_process_model(self, old_state,control, training):
        """
        Predicts the next state given the old state and actions performed

        """
        if self.learned_process_model:
            joint_input = tf.keras.layers.Concatenate(axis=1)([old_state, control])
            out = self.process_model(joint_input,training)
        else:
            out = self.analytical_model(control)
        if tf.math.reduce_any(tf.math.is_nan(out)):
            pdb.set_trace()
        return out

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


        posterior_state, z_pred, pred_state = prediction
        pose = label

        observation = data

        nll_posterior = self.neg_log_likelihood((posterior_state, pose),self.grid_size)
        nll_likelihood = self.neg_log_likelihood((z_pred, pose),self.grid_size)

        nl_loss_posterior = tf.reduce_mean(nll_posterior)
        nl_loss_measurement = tf.reduce_mean(nll_likelihood)

        # compute the mode of the distribution
        mode_pose_posterior, mean_pose_posterior = self.compute_mode_(posterior_state)
        mode_pose_pred, mean_pose_pred = self.compute_mode_(pred_state)
        mode_obs, mean_obs = self.compute_mode_(z_pred)

        mean_pose_posterior_s1 = self.compute_mean_S1(posterior_state)
        mean_pose_pred_s1 = self.compute_mean_S1(pred_state)
        mean_obs_s1 = self.compute_mean_S1(z_pred)

        abs_mean_pose_posterior = tf.math.abs(mean_pose_posterior - pose)
        abs_mean_pose_posterior_s1 = tf.math.abs(mean_pose_posterior_s1 - pose)
        abs_mean_pose_pred = tf.math.abs(mean_pose_pred - pose)
        abs_mean_pose_pred_s1 = tf.math.abs(mean_pose_pred_s1 - pose)
        abs_mean_obs = tf.math.abs(mean_obs - observation)
        abs_mean_obs_s1 = tf.math.abs(mean_obs_s1 - observation)

        mae_mean1_post = tf.reduce_mean(abs_mean_pose_posterior)
        mae_mean2_post = tf.reduce_mean(abs_mean_pose_posterior_s1)
        mae_mean1_pred = tf.reduce_mean(abs_mean_pose_pred)
        mae_mean2_pred = tf.reduce_mean(abs_mean_pose_pred_s1)
        mae_mean1_meas = tf.reduce_mean(abs_mean_obs)
        mae_mean2_meas = tf.reduce_mean(abs_mean_obs_s1)

        diff_mode_pose_posterior = tf.math.abs(mode_pose_posterior - pose)
        diff_mode_pose_pred = tf.math.abs(mode_pose_pred - pose)
        diff_mode_obs = tf.math.abs(mode_obs - observation)

        ate_mode_post = self.average_traj_error(diff_mode_pose_posterior)
        ate_mode_pred = self.average_traj_error(diff_mode_pose_pred)
        ate_mode_meas = self.average_traj_error(diff_mode_obs)

        ate_mean1_post = self.average_traj_error(abs_mean_pose_posterior)
        ate_mean2_post = self.average_traj_error(abs_mean_pose_posterior_s1)
        ate_mean1_pred = self.average_traj_error(abs_mean_pose_pred)
        ate_mean2_pred = self.average_traj_error(abs_mean_pose_pred_s1)
        ate_mean1_meas = self.average_traj_error(abs_mean_obs)
        ate_mean2_meas = self.average_traj_error(abs_mean_obs_s1)

        mae_mode_post = tf.reduce_mean(tf.math.abs(diff_mode_pose_posterior))
        mae_mode_pred = tf.reduce_mean(tf.math.abs(diff_mode_pose_pred))
        mae_mode_meas = tf.reduce_mean(tf.math.abs(diff_mode_obs))

        # total_loss = nl_loss_posterior

        # TODO: get the weight decay

        # wd = []
        # for la in self.layers:
        #     wd += la.losses
        # wd = tf.add_n(wd)

        if self.loss == "nl_measurement":
            total = nl_loss_measurement
        else:
            total = nl_loss_posterior
        # total = tf.reduce_mean(mse_obs) + wd
        if tf.math.reduce_any(tf.math.is_nan(total)):
            pdb.set_trace()
        metrics = [total, nl_loss_posterior, nl_loss_measurement, ate_mode_post, ate_mode_pred, ate_mode_meas,
                   ate_mean1_post, ate_mean2_post, ate_mean1_pred, ate_mean2_pred,
                   ate_mean1_meas, ate_mean2_meas,
                   mae_mode_post, mae_mode_pred, mae_mode_meas, mae_mean1_post, mae_mean2_post, mae_mean1_pred,
                   mae_mean2_pred, mae_mean1_meas, mae_mean2_meas]
        metric_names = ["total", "nl_loss_posterior", "nl_loss_measurement", "ate_mode_post", "ate_mode_pred",
                        "ate_mode_meas",
                        "ate_mean1_post", "ate_mean2_post", "ate_mean1_pred", "ate_mean2_pred",
                        "ate_mean1_meas", "ate_mean2_meas",
                        "mae_mode_post", "mae_mode_pred", "mae_mode_meas", "mae_mean1_post", "mae_mean2_post",
                        "mae_mean1_pred", "mae_mean2_pred", "mae_mean1_meas", "mae_mean2_meas"]
        return total, metrics, metric_names

    def compute_mean_S1(self, energy_samples):
        dim = energy_samples.shape[1]
        tensor_start = tf.constant(0, dtype=tf.float64)
        tensor_stop = tf.constant(2 * math.pi, dtype=tf.float64)
        poses = tf.broadcast_to(tf.linspace(tensor_start, tensor_stop, self.grid_size + 1)[:-1],[1,1,self.grid_size])
        poses_ = tf.tile(poses, [self.batch_size, dim, 1])
        maximum = tf.expand_dims(tf.math.reduce_max(energy_samples, axis=2), 2)
        moments = tf.signal.rfft(tf.exp(energy_samples - maximum))
        ln_z_ = tf.expand_dims(tf.math.real(tf.math.log(moments[:, :, 0] / math.pi)), 2) + maximum
        prob = tf.cast(tf.math.exp(energy_samples - ln_z_), dtype=tf.float64)
        mean = tfp.math.trapz(prob * poses_, poses_, axis=-1)
        return tf.expand_dims(mean, 2)

    def average_traj_error(self, trajector_error):
        return tf.math.sqrt(tf.math.reduce_mean(trajector_error * trajector_error))

    def compute_mode_(self, energy_samples):

        maximum = tf.expand_dims(tf.math.reduce_max(energy_samples, axis=2), 2)
        moments = tf.signal.rfft(tf.exp(energy_samples - maximum))
        ln_z_ = tf.expand_dims(tf.math.real(tf.math.log(moments[:, :, 0] / math.pi)), 2) + maximum
        dim = energy_samples.shape[1]
        tensor_start = tf.constant(0, dtype=tf.float64)
        tensor_stop = tf.constant(2 * math.pi, dtype=tf.float64)
        poses = tf.broadcast_to(tf.linspace(tensor_start, tensor_stop, self.grid_size + 1)[:-1], [1, 1, self.grid_size])
        poses_ = tf.tile(poses, [self.batch_size, dim, 1])
        prob = tf.math.exp(energy_samples - ln_z_)
        mode_idx = tf.expand_dims(tf.argmax(prob, axis=2), 2)
        poses_mode = tf.expand_dims(tf.gather_nd(poses_, mode_idx, batch_dims=2), 2)
        mean = tf.expand_dims(tf.cast(tf.math.real(moments[:, :, 1] / moments[:, :, 0]), dtype=tf.float64), 2)
        return poses_mode, mean

    # def eta_symmetric_non_symmetric(self, coefficients):
    #     coefficients_complete = []
    #     for i in range(self.grid_size):
    #         if i == self.zeroth_freq_index:
    #             coefficients_complete.append(coefficients[0])
    #         elif i < self.zeroth_freq_index:
    #             coefficients_complete.append(tf.math.conj(coefficients[self.zeroth_freq_index - i]))
    #         elif i > self.zeroth_freq_index:
    #             coefficients_complete.append(coefficients[i - self.zeroth_freq_index])
    #
    #     return tf.stack(coefficients_complete)

    def neg_log_likelihood(self, input_output_tuple,grid_size):
        """
              Compute the negative log likelihood loss over Harmonic Exponential Distribution for trajectory. This function would need to be called over a batch.

              Args:
                  energy_samples: list of predicted tensors for one pose
                  true_value: ground truth value for the variable being estimated, need to be complex

              Returns:
                  loss: returns a scalar value for negative log likelihood of the Harmonic Exponential Distribution for the ground truth over the trajectory.

        """

        energy_samples, true_value = input_output_tuple
        energy_samples = tf.cast(energy_samples, tf.complex64)
        eta = tf.cast(tf.signal.fftshift(tf.signal.fft(energy_samples),axes = -1), dtype=tf.complex64)
        maximum = tf.expand_dims(tf.math.reduce_max(tf.math.real(energy_samples), axis=-1), -1)
        maximum = tf.cast(maximum, dtype=tf.complex64)
        moments = tf.signal.fft(tf.exp(energy_samples - maximum))
        ln_z_ = tf.cast(tf.math.real(tf.expand_dims(tf.math.log(moments[..., 0] / (math.pi*grid_size*math.pi/62)), -1) + maximum),
                        dtype=tf.float64) # [batch_size, trajectory_length, 1]
        k_values = tf.range(0, self.grid_size, 1,dtype=tf.dtypes.float64)  - self.zeroth_freq_index
        k_values = tf.expand_dims(tf.expand_dims(tf.expand_dims(k_values, 0), 0),-1)# [1, 1, grid_size,1]
        value = tf.expand_dims(true_value,-1) # [batch_size, trajectory_length, 1,1]
        # pdb.set_trace()
        exponential_term = tf.math.exp(tf.constant(1j,dtype=tf.complex64) * tf.cast(k_values * value,dtype=tf.complex64)) #[batch_size, trajectory_length, grid_size,1]
        inverse_transform = tf.math.real(tf.math.reduce_sum(tf.expand_dims(eta, -1) * exponential_term,axis=2)) # [batch_size, trajectory_length,  1]
        nll = tf.reduce_mean(tf.cast(-inverse_transform / grid_size,dtype=tf.float64) + ln_z_)
        return nll




    def energy(self, step, samples):
        cov = self.motion_noise ** 2 if self.motion_noise != 0.0 else 0.1
        mu = step
        energy = (tf.cos(samples - mu)) / cov
        return energy

    # def theta_to_2D(self, theta):
    #     r = 1.0
    #     ct = tf.cos(theta)
    #     st = tf.sin(theta)
    #     out = tf.stack([r * ct, r * st], axis=-1)
    #     return out

    def analytical_model(self,step):
        samples = tf.linspace(0.0, 2 * math.pi, self.grid_size + 1)[:-1][tf.newaxis, :]
        samples_ = tf.tile(samples, [self.batch_size, 1])
        return tf.reshape(self.energy(step,samples_), [self.batch_size, self.grid_size])



class ObservationModel(tf.keras.Model):
    def __init__(self, batch_size, grid_size):
        super().__init__()
        self.batch_size = batch_size
        self.grid_size = grid_size

    def build(self, input_shape=None):
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(16, activation="relu", name="layer1"),
                tf.keras.layers.Dense(64, activation="relu", name="layer2"),
                tf.keras.layers.Dense(self.grid_size, name="layer3"),
            ]
        )

    def call(self, input, training):
        return self.model(input, training=training)


# def add_noise_model(self, input_shape):
#     """
#     Creates a neural network-based noise model.
#
#     Parameters
#     ----------
#     input_shape : tuple
#         The shape of the input to the noise model.
#
#     Returns
#     -------
#     tf.keras.Model
#         A Keras model representing the noise model.
#     """
#     inputs = tf.keras.Input(shape=input_shape)
#     x = tf.keras.layers.Dense(32, activation="relu")(inputs)
#     x = tf.keras.layers.Dense(64, activation="relu")(x)
#     x = tf.keras.layers.Dense(32, activation="relu")(x)
#     outputs = tf.keras.layers.Dense(self.grid_size)(x)
#     noise_model = tf.keras.Model(inputs, outputs, name="noise_model")
#     return noise_model

class ProcessModel(tf.keras.Model):
    """
        Arguments:
            grid_size: bandwidth of the filter
            step: Assumes constant step between two poses, x_t+1 = x_t + step
            motion_noise: Helps add uncertainity the step making it a stochastic process
            batch_size: training samples used to train the process model

        Output:
            Outputs the energy of the state transition function.

        Currently this class is not learning the process model through data but assumes a wrapped normal gaussian distribution to represent p_u(x_t - x_t-1)
    """

    def __init__(self, grid_size, batch_size):
        super().__init__()
        self.grid_size = grid_size
        self.batch_size = batch_size
    def build(self, input_shape=None):
        self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    units=32,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.initializers.glorot_normal(),
                    kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
                    bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
                    name='process_fc1'),
                tf.keras.layers.Dense(
                    units=64,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.initializers.glorot_normal(),
                    kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
                    bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
                    name='process_fc2'),
                tf.keras.layers.Dense(
                    units=self.grid_size,
                    activation=None,
                    kernel_initializer=tf.initializers.glorot_normal(),
                    kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
                    bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
                    name='process_fc3'),
            ])


    def call(self,input=None,training=None):
        return self.model(input,training=training)

