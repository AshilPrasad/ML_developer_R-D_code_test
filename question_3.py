import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Concatenate

class AddLayer(Layer):
    def __init__(self, **kwargs):
        super(AddLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AddLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.add(inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class MultiplyLayer(Layer):
    def __init__(self, **kwargs):
        super(MultiplyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MultiplyLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.multiply(inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class CombineLayer(Layer):
    def __init__(self, **kwargs):
        super(CombineLayer, self).__init__(**kwargs)
        self.add_layer = AddLayer()
        self.multiply_layer = MultiplyLayer()

    def build(self, input_shape):
        self.add_layer.build(input_shape)
        self.multiply_layer.build(input_shape)
        super(CombineLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        addition_output = self.add_layer(inputs)
        multiplication_output = self.multiply_layer(inputs)
        combined_output = tf.keras.layers.Concatenate(axis=-1)([addition_output, multiplication_output])
        return combined_output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0][0], input_shape[0][1] * 2))

# Example data for batch inference
batch_size = 4
input_data_a = np.random.random((batch_size, 10))
input_data_b = np.random.random((batch_size, 10))

# Custom layers
add_layer = AddLayer(name='custom_addition')
multiply_layer = MultiplyLayer(name='custom_multiplication')
combine_layer = CombineLayer(name='custom_combine')

# Inputs
input_a = tf.keras.layers.Input(shape=(10,), name='input_a')
input_b = tf.keras.layers.Input(shape=(10,), name='input_b')

# Custom layer outputs
added_output = add_layer([input_a, input_b])
multiplied_output = multiply_layer([input_a, input_b])
combined_output = combine_layer([input_a, input_b])

# Create models
addition_model = tf.keras.Model(inputs=[input_a, input_b], outputs=added_output)
multiplication_model = tf.keras.Model(inputs=[input_a, input_b], outputs=multiplied_output)
combined_model = tf.keras.Model(inputs=[input_a, input_b], outputs=combined_output)

# Batch inference
addition_result = addition_model.predict([input_data_a, input_data_b])
multiplication_result = multiplication_model.predict([input_data_a, input_data_b])
combined_result = combined_model.predict([input_data_a, input_data_b])

# Display results
print("Batch Inference Results:")
print("Addition Result:")
print(addition_result)

print("\nMultiplication Result:")
print(multiplication_result)

print("\nCombined Result:")
print(combined_result)
