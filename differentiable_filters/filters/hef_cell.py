"""
RNN cell implementing a Differentiable Extended Kalman Filter
"""

import tensorflow as tf
from differentiable_filters.filters import filter_cell_base as base
import math
import pdb

class HEFCell(base.FilterCellBase):
    def __init__(self, context, problem, grid_size, update_rate=1, debug=False):
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
        self.grid_size = grid_size

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        It can be represented by an Integer, a TensorShape or a tuple of
        Integers or TensorShapes.
        """
        # return [[1],[self.grid_size],[1]]
        return [[self.grid_size],[1]]

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        # [[1],[1],[self.grid_size],[self.grid_size],[self.grid_size],[self.grid_size]]
        return [[1],[1],[self.grid_size],[self.grid_size]]
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
            observations, control,epoch = inputs
            # import pdb;pdb.set_trace()
            prior, c_step = states

            # old_state = tf.reshape(old_state, [self.batch_size, -1])
            # predict the next state
            process_energy_mean, process_energy_cov = self.context.run_process_model(control,epoch[0,0],training)


            process_energy = self.context.analytical_model(process_energy_mean, process_energy_cov)
            pred_state_eta,pred_state_energy = self._prediction_step(prior,process_energy)


            # z_pred_energy_mean,z_pred_energy_cov = self.context.run_observation_model(observations,epoch[0,0],
            #                                                training)
            # z_pred_energy = self.context.analytical_model(z_pred_energy_mean, z_pred_energy_cov)


            ###################################################################
            # update the predictions with the observations
            # state_up = self._update(pred_state_eta, z_pred_energy)
            # state = tf.cast(tf.reshape(state_up, [self.batch_size, -1]),dtype=tf.float64)
            # z_pred_energy = tf.cast(tf.reshape(z_pred_energy, [self.batch_size, -1]),dtype=tf.float64)
            pred_state_energy = tf.cast(tf.reshape(pred_state_energy, [self.batch_size, -1]),dtype=tf.float64)
            process_energy = tf.cast(tf.reshape(process_energy, [self.batch_size, -1]),dtype=tf.float64)

            # mean_state = tf.expand_dims(tf.math.reduce_mean(state, axis=1),-1)
            # the recurrent state contains the updated state estimate
            new_state = (pred_state_energy, c_step + 1)
            # new_state = (c_step[0] + 1)

            # print(f" for the epoch {epoch[0,0]} the process energy mean is {tf.math.reduce_mean(process_energy_mean)} and the process energy covariance is {tf.math.reduce_mean(process_energy_cov)}")
            print(f" for the epoch {epoch[0,0]} the observation energy mean is {tf.math.reduce_mean(process_energy_mean)} and the observation energy covariance is {tf.math.reduce_mean(process_energy_cov)}")


            # output = (process_energy_mean, z_pred_energy_mean, state, z_pred_energy, pred_state_energy,process_energy)
            output = (process_energy_mean, process_energy_cov,process_energy,pred_state_energy)


            return output, new_state

    def setting_layer_trainable(self, model,value):
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):  # if you're using a model as a layer
                self.setting_layer_trainable(layer,value)  # apply function recursively
                continue
            layer.get_layer("sequential").trainable = value

    def _prediction_step(self,energy_samples_old,process_energy_samples):
        ln_z_1 = self.calculate_normalisation_const(energy_samples_old)
        ln_z_2 = self.calculate_normalisation_const(process_energy_samples)

        prob_1 = tf.math.exp(energy_samples_old - ln_z_1)
        prob_2 = tf.math.exp(process_energy_samples - ln_z_2)

        m1 = tf.cast(tf.signal.rfft(prob_1),dtype=tf.complex64)
        m2 = tf.cast(tf.signal.rfft(prob_2),dtype=tf.complex64)

        m_conv = m1*m2
        # pdb.set_trace()
        #Currently the moments are convolution are unnormlized and so are the outputs eta and energy
        eta, energy = self.convert_moments_eta_energy(m_conv)

        return eta, energy


    def calculate_normalisation_const(self,energy):
        # import pdb;pdb.set_trace()
        maximum = tf.expand_dims(tf.math.reduce_max(energy, axis=1), 1)
        moments = tf.signal.rfft(tf.exp(energy - maximum))
        ln_z_ = tf.expand_dims(tf.math.real(tf.math.log(moments[:, 0] / (math.pi * self.grid_size * math.pi / 62))), 1) + maximum

        return ln_z_


    def _update(self, prior_eta, measurement_model):
        eta_update =  prior_eta + self.convert_from_energy_eta(measurement_model)
        return self.convert_from_eta_energy(eta_update)

    def convert_moments_eta_energy(self,moments):
        prob = tf.signal.irfft(moments)
        # ln_z_ = tf.expand_dims(tf.math.real(tf.math.log(moments[:, 0] / (math.pi * self.grid_size * math.pi / 62))),1)
        prob_real = tf.math.real(prob)
        prob_process = tf.where(prob_real>0,prob_real,1e-8)
        energy = tf.math.log(prob_process)
        eta = tf.signal.rfft(energy)
        return eta,energy

    def convert_from_eta_energy(self,eta):
        return tf.math.real(tf.signal.irfft(eta))

    def convert_from_energy_eta(self,energy_samples):
        eta = tf.signal.rfft(energy_samples)
        return eta


