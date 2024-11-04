
import tensorflow as tf
import math

from differentiable_filters.contexts import base_context as base

class DiscTrackingContext(base.BaseContext):
    def __init__(self, batch_size, filter_type, grid_size, motion_noise, measurement_noise, learned_process_model,
                 learned_measurement_model):
        super(base.BaseContext, self).__init__()

        # mark which filter and loss function are used
        self.filter_type = filter_type

        # define the state size and name the components
        self.batch_size = batch_size
        self.learned_process_model = learned_process_model
        self.learned_measurement_model = learned_measurement_model
        # # parameters of the process model
        self.grid_size = grid_size
        self.zeroth_freq_index = math.floor(self.grid_size / 2)
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        if learned_measurement_model:
            self.observation_model = ObservationModel(self.batch_size, self.grid_size)
        if learned_process_model:
            self.process_model = ProcessModel(self.grid_size, self.batch_size)
        self.dim_x = None
        self.dim_z = None
        self.dim_u = None


def run_observation_model(self, observations, epoch, training):
    pass


def run_process_model(self, control, epoch, training):
    pass


def get_loss(self, data, label, control, prediction):
    pass


class ObservationModel(tf.keras.Model):
    def __init__(self, batch_size, grid_size):
        super().__init__()
        self.batch_size = batch_size
        self.grid_size = grid_size

    def build(self, input_shape=None):
        self.logcov_model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units=8,
                activation=tf.nn.relu,
                kernel_initializer=tf.initializers.HeUniform(),
                # kernel_initializer='random_normal',
                bias_initializer='zeros',
                # kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
                # bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
                name='observation_fc1'),
            tf.keras.layers.Dense(
                units=1,
                activation=None,
                kernel_initializer=tf.initializers.HeUniform(),
                # kernel_initializer='random_normal',
                bias_initializer='zeros',
                # kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
                # bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
                name='observation_fc2'),
        ], name='observation_sequential')

        self.mean_model = tf.keras.layers.Dense(units=1, activation=None,
                                                kernel_initializer=tf.initializers.ones(),
                                                bias_initializer='zeros', name='observation_mean_fc')
        # self.fc = tf.keras.layers.Dense(units=16, activation=tf.nn.relu, kernel_initializer=tf.initializers.HeUniform(),
        #                                 bias_initializer='zeros', name='observation_back_fc')

    def call(self, input, training):
        # x = self.fc(input)
        log_cov = self.logcov_model(input, training=training)
        mean = self.mean_model(input, training=training)
        return mean % (2 * math.pi), log_cov

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
        self.logcov_model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units=8,
                activation=tf.nn.relu,
                kernel_initializer=tf.initializers.HeUniform(),
                # kernel_initializer='random_normal',
                bias_initializer='zeros',
                # kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
                # bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
                name='process_fc1'),
            tf.keras.layers.Dense(
                units=1,
                activation=None,
                kernel_initializer=tf.initializers.HeUniform(),
                # kernel_initializer='random_normal',
                bias_initializer='zeros',
                # kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
                # bias_regularizer=tf.keras.regularizers.l2(l=1e-3),
                name='process_fc2'),
        ], name='process_sequential')

        self.mean_model = tf.keras.layers.Dense(units=1, activation=None,
                                                kernel_initializer=tf.initializers.ones(),
                                                bias_initializer='zeros', name='process_mean_fc')

        # self.fc = tf.keras.layers.Dense(units=10, activation=tf.nn.relu, kernel_initializer='random_normal',
        #                                 bias_initializer='zeros', name='process_back_fc')

    def call(self, input=None, training=None):
        # x = self.fc(input)
        log_cov = self.logcov_model(input, training=training)
        mean = self.mean_model(input, training=training)
        return mean %(2 * math.pi), log_cov
