import tensorflow as tf

class SE2Group:
    """
    Class representing the SE(2) group of rigid body transformations in R^2
    """

    def __init__(self, R: tf.Tensor = tf.eye(2), t: tf.Tensor = tf.zeros(2)):
        """
        Construct an SE2 object from a Rotation matrix R and translation t.

        :param R: rotation matrix (2,2)
        :param t: translation vector (2)
        """
        self.R = R
        self.t = t

    @classmethod
    def from_parameters(cls, x: float = 0., y: float = 0., theta: float = 0.0):
        """
        Construct an SE2 object from x, y, theta parameters.

        :param x: x component of translation
        :param y: y component of translation
        :param theta: rotation angle in radians
        :return: an SE2 object of the given parameters
        """
        R = tf.zeros((2, 2))
        R = tf.tensor_scatter_nd_update(R, [[0, 0], [0, 1], [1, 0], [1, 1]],
                                        [tf.cos(theta), -tf.sin(theta), tf.sin(theta), tf.cos(theta)])
        t = tf.constant([x, y], dtype=tf.float32)
        return cls(R, t)

    def parameters(self) -> tf.Tensor:
        """
        Get the x, y, theta parameters of the current transformation.

        :return: x, y, theta parameters as a 3 dimensional vector.
        """
        theta = tf.atan2(self.R[1, 0], self.R[0, 0])
        return tf.concat([self.t, [theta]], axis=0)

    def __matmul__(self, other) -> 'SE2Group':
        """
        Group composition of SE(2)
        """
        R = tf.matmul(self.R, other.R)
        t = tf.matmul(self.R, tf.expand_dims(other.t, axis=-1)) + tf.expand_dims(self.t, axis=-1)
        return SE2Group(R, tf.squeeze(t))

    def inv(self) -> 'SE2Group':
        """
        Inversion of the current transformation.

        :return: SE2 object of the inverted transformation.
        """
        R_inv = tf.transpose(self.R)
        t_inv = -tf.matmul(R_inv, tf.expand_dims(self.t, axis=-1))
        return SE2Group(R_inv, tf.squeeze(t_inv))