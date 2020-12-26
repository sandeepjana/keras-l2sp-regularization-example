'''
Inspired from
https://github.com/holyseven/TransferLearningClassification
https://keras.io/guides/customizing_what_happens_in_fit/
https://towardsdatascience.com/tensorflow-2-2-and-a-custom-training-logic-16fa72934ac3
# https://stackoverflow.com/a/62440411

class Model:
github.com/tensorflow/tensorflow/blob/582c8d236cb079023657287c318ff26adb239002/tensorflow/python/keras/engine/training.py#L216
'''

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
        # tf.identity is bad naming for deep copy operation
        # https://github.com/tensorflow/tensorflow/issues/11186
        self.train_vars_init = [tf.identity(a)
                                for a in self.trainable_variables]
        self.iteration_counter = 0
        self.reg_alpha = None

    # note: get_config <--> from_config pair is too complicated 
    # for functional API path
    
    def compile(self, *arg, **kwargs):
        """Configures the model for training.
        """
        self.reg_alpha = kwargs.pop('reg_alpha')
        super().compile(*arg, **kwargs)

    def l2sp_regularization_loss(self):
        loss = 0
        for current, init in zip(self.trainable_variables,
                                 self.train_vars_init):
            loss += self.reg_alpha * \
                tf.math.reduce_sum(tf.math.square(current - init))
        return loss

    def set_weights(self, *args, **kwargs):
        super().set_weights(*args, **kwargs)
        self.train_vars_init = [tf.identity(a)
                                for a in self.trainable_variables]

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

        if self.iteration_counter % 200 == 0:
            print(f'losses {normal_loss, reg_loss}')
            print(f'{self.iteration_counter:3d}: train_vars_init \
                {self.train_vars_init[0][:10]}')
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


inputs = layers.Input(shape=(32,))
outputs = layers.Dense(1)(inputs)
model = Model(inputs, outputs)

# Wrap an existing model -- you may or may not have model definition code
model = WrapModelL2SP(model.inputs, model.outputs)

weights = model.get_weights()

np.random.seed(1947)
new_weights = [np.random.random(a.shape) for a in weights]
model.set_weights(new_weights)


def get_few_weights(m):
    return m.get_weights()[0][:5]


print(f'Initial weights: {get_few_weights(model)}')

model.compile(optimizer="adam", loss="mse", metrics=["mae"],
    reg_alpha=1)
model.run_eagerly = True

# Just use `fit` as usual
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))

# printing compiled loss and metrics only, L2-SP loss value not yet logged above
model.fit(x, y, epochs=1)

print(f'Final weights: {get_few_weights(model)}')

filename = 'model.h5'
model.save(filename)

del model

model = load_model(filename, custom_objects={'WrapModelL2SP': WrapModelL2SP},
    compile=False)
print(f'Reloaded weights: {get_few_weights(model)}')
