import tensorflow as tf
from tensorflow.keras.layers import Layer

class AddLayer(Layer):
    def __init__(self, **kwargs):
        super(AddLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # No trainable parameters to be added
        super(AddLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # Assuming two inputs of the same shape
        return tf.add(inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        # Output shape is the same as the input shape
        return input_shape[0]

class MultiplyLayer(Layer):
    def __init__(self, **kwargs):
        super(MultiplyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # No trainable parameters to be added
        super(MultiplyLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # Assuming two inputs of the same shape
        return tf.multiply(inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        # Output shape is the same as the input shape
        return input_shape[0]

# Example usage
input_a = tf.keras.layers.Input(shape=(10,), name='input_a')
input_b = tf.keras.layers.Input(shape=(10,), name='input_b')

# Custom addition layer
add_layer = AddLayer(name='custom_addition')
added_output = add_layer([input_a, input_b])

# Custom multiplication layer
multiply_layer = MultiplyLayer(name='custom_multiplication')
multiplied_output = multiply_layer([input_a, input_b])

# Create models
addition_model = tf.keras.Model(inputs=[input_a, input_b], outputs=added_output)
multiplication_model = tf.keras.Model(inputs=[input_a, input_b], outputs=multiplied_output)

# Display model summaries
print("Addition Model Summary:")
addition_model.summary()

print("\nMultiplication Model Summary:")
multiplication_model.summary()
