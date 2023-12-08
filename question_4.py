import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Reshape

# Function to split an image into tiles
def split_image_into_tiles(image):
    # Extract patches with size (4, 4, channels)
    patches = tf.image.extract_patches(
        images=tf.expand_dims(image, axis=0),  # Add batch dimension
        sizes=[1, 4, 4, 1],
        strides=[1, 4, 4, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )

    # Reshape patches to (16, channels)
    num_patches = tf.reduce_prod(tf.shape(patches)[1:3])
    reshaped_patches = tf.reshape(patches, (num_patches, -1))

    return reshaped_patches

# Example data
image_size = (16, 16, 3)  # Assuming an RGB image
input_image = tf.random.normal(image_size)

# Input layer
input_layer = Input(shape=image_size, name='input_image')

# Lambda layer to split the image into tiles
split_tiles_layer = Lambda(split_image_into_tiles, name='split_tiles')(input_layer)

# Display input image shape
print("Input Image Shape:", input_image.shape)

# Perform inference
result = split_tiles_layer(input_image)

# Display result
print("Resulting Tiles Shape:", result.shape)
