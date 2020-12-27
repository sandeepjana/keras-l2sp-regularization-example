'''
Inspired from
https://github.com/holyseven/TransferLearningClassification
https://keras.io/guides/customizing_what_happens_in_fit/
https://towardsdatascience.com/tensorflow-2-2-and-a-custom-training-logic-16fa72934ac3
# https://stackoverflow.com/a/62440411

class Model:
github.com/tensorflow/tensorflow/blob/582c8d236cb079023657287c318ff26adb239002/tensorflow/python/keras/engine/training.py#L216
'''

from numpy.lib.function_base import extract
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model, load_model
import numpy as np

assert float(tf.__version__[:3]) >= 2.2, "Requires TensorFlow 2.2 or later."

# It seems there are two ways to subclass keras.Model
# 1. Funcitonal style API where Model(inputs, outputs, name=...) can be replaced
# by exactly CustomModel(inputs, outputs, name=...). No additional arguments possible
# 2. Sub-classing which accepts any kind of arguments
# I don't know if one is preferred over the other


class WrapModelL2SP(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_alpha = None
        self.reg_beta = None
        self.backbone_init_weights = None
        self.iteration_counter = 0

    # note: get_config <--> from_config pair is too complicated
    # for functional API path, which is different for straightforward (?) subclassing

    def compile(self, *arg, **kwargs):
        """Configures the model for training.
        reg_alpha: L2-SP weight decay rate for feature extractor layers
        reg_beta: regular weight decay rate for final layers
        backbone_init_weights: weights of the pre-trained feature extractor 
        """
        self.reg_alpha = kwargs.pop('reg_alpha')
        self.reg_beta = kwargs.pop('reg_beta')
        self.backbone_init_weights = kwargs.pop('backbone_init_weights')
        super().compile(*arg, **kwargs)

    def l2sp_regularization_loss(self):
        feature_var_diffs = [(v - w) for v, w in zip(self.weights, self.backbone_init_weights)
                             if v.trainable]
        loss = self.reg_alpha * sum(tf.math.reduce_sum(tf.math.square(diff))
                                    for diff in feature_var_diffs)
        rest_of_vars = self.weights[len(self.backbone_init_weights):]
        loss += self.reg_beta * sum(tf.math.reduce_sum(tf.math.square(v))
                                    for v in rest_of_vars if v.trainable)
        return loss

    def print_reg_loss(self, normal_loss, reg_loss):
        print(f'losses {normal_loss, reg_loss}')
        print('-' * 100)
        print(f'{self.iteration_counter:3d}: {self.get_weights()}')

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # Not passing regularization_losses=self.losses
            normal_loss = self.compiled_loss(y, y_pred)
            reg_loss = self.l2sp_regularization_loss()
            loss = normal_loss + reg_loss

        # if self.iteration_counter % 200 == 0:
        #     self.print_reg_loss(normal_loss, reg_loss)
        self.iteration_counter += 1

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


input_shape = (4,)
inputs = layers.Input(shape=input_shape)
features = layers.Dense(2, activation='relu')(inputs)
extractor = Model(inputs, features)
# extractor.trainable = False
# set some known value for init weights

def set_unity_weights(m):
    w = [np.ones_like(a) for a in m.get_weights()]
    m.set_weights(w)
    return w

backbone_init_weights = set_unity_weights(extractor)

print('=' * 100)
print(f'Initial weights: {backbone_init_weights}')


inputs = layers.Input(shape=input_shape)
# extractor.trainble = False 
features = extractor(inputs)
outputs = layers.Dense(1)(features)

model = WrapModelL2SP(inputs, outputs)
set_unity_weights(model)


def get_few_weights(m):
    print('=' * 100)
    return m.get_weights()


model.compile(optimizer="adam", loss="mse", 
              reg_alpha=100, reg_beta=100, backbone_init_weights=backbone_init_weights)
model.run_eagerly = True

x = np.random.random((1000, ) + input_shape)
y = np.dot(x, np.random.random(input_shape))

# printing compiled loss and metrics only, L2-SP loss value not yet logged above
model.fit(x, y, epochs=100, validation_split=0.25)

print(f'Final weights: {get_few_weights(model)}')

filename = 'model.h5'
model.save(filename)


model = load_model(filename, custom_objects={'WrapModelL2SP': WrapModelL2SP},
                   compile=False)
print(f'Reloaded weights: {get_few_weights(model)}')
