import os
import tensorflow as tf

# Define the directory containing your .keras files
directory = "C:/Users/puter/Downloads/final/data/keraspt1"  # Update this path if needed

# Loop through all .keras files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".keras"):
        keras_path = os.path.join(directory, filename)

        # Load the .keras model file
        model = tf.keras.models.load_model(keras_path)

        # Define the path for saving the .h5 model
        h5_filename = filename.replace(".keras", ".h5")
        h5_path = os.path.join(directory, h5_filename)

        # Save the model as .h5
        model.save(h5_path)

        print(f"Converted {filename} to {h5_filename}")
