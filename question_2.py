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


class CombineLayer(Layer):
    def __init__(self, **kwargs):
        super(CombineLayer, self).__init__(**kwargs)
        # Custom layers for addition and multiplication
        self.add_layer = AddLayer()
        self.multiply_layer = MultiplyLayer()

    def build(self, input_shape):
        # Build custom layers
        self.add_layer.build(input_shape)
        self.multiply_layer.build(input_shape)
        super(CombineLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # Call custom layers and concatenate their outputs
        addition_output = self.add_layer(inputs)
        multiplication_output = self.multiply_layer(inputs)
        combined_output = tf.keras.layers.Concatenate(axis=-1)([addition_output, multiplication_output])
        return combined_output

    def compute_output_shape(self, input_shape):
        # Output shape is twice the input shape due to concatenation
        return tf.TensorShape((input_shape[0][0], input_shape[0][1] * 2))
    
# Example usage
input_a = tf.keras.layers.Input(shape=(10,), name='input_a')
input_b = tf.keras.layers.Input(shape=(10,), name='input_b')

# Custom combined layer
combine_layer = CombineLayer(name='custom_combine')
combined_output = combine_layer([input_a, input_b])

# Create model
combined_model = tf.keras.Model(inputs=[input_a, input_b], outputs=combined_output)

# Display model summary
print("Combined Model Summary:")
combined_model.summary()
