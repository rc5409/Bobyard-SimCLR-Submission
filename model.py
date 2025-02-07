
import tensorflow as tf
from custom_resnet import build_resnet50

class ProjectionHead(tf.keras.layers.Layer):
    def __init__(self, projection_dim=128, **kwargs):
        super(ProjectionHead, self).__init__(**kwargs)
        self.projection_head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(projection_dim)
        ])

    def call(self, inputs, training=False):
        return self.projection_head(inputs, training=training)

class SimCLR(tf.keras.Model):
    def __init__(self, projection_dim=128, input_shape=(32, 32, 3)):
        super(SimCLR, self).__init__()
        self.encoder = build_resnet50()

        self.projection_head = ProjectionHead(projection_dim=projection_dim)

    def call(self, inputs, training=False):

        encoded_features = self.encoder(inputs, training=training)
        projected_features = self.projection_head(encoded_features, training=training)
        return projected_features
