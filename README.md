# Fashion-MNIST DCGAN

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate realistic images of fashion items from the Fashion-MNIST dataset. It leverages TensorFlow 2 to train the model effectively.

## Key Features

- **Multi-Layer Perceptron (MLP) Generator**: Generates fashion item images from a 100-dimensional noise vector.
- **Convolutional Discriminator**: Distinguishes between real and generated fashion images.
- **Adam Optimizer**: Efficiently optimizes both the generator and discriminator.
- **Checkpoint Saving**: Periodically saves model checkpoints for potential resumption or evaluation.
- **Image Generation**: Visualizes generated images at different epochs for monitoring training progress.
- **Model Saving (Optional)**: Saves the trained generator and discriminator models in HDF5 format (.h5) for potential deployment or further experimentation. (Consider using the native Keras format, e.g., `model.save('my_model.keras')` for better compatibility.)

## Getting Started

### Prerequisites

- Python 3.6 or later
- TensorFlow 2.x (`pip install tensorflow`)
- Other required libraries are installed within the notebook (see code for details)

### Clone the Repository

```bash
git clone https://github.com/atharvv8/Fashion_GAN
```
### Run the Notebook

1. Open `Generating_Fashion_Designs.ipynb` in a Jupyter Notebook environment.
2. Execute the code cells sequentially.

## Project Structure

```plaintext
fashion_mnist_gan/
├── Generating_Fashion_Designs.ipynb  # The main notebook containing the GAN implementation
├── README.md                # This file (you're here!)

## Model Architecture

### Generator

- **Dense layer**: Projects noise vector into a high-dimensional feature space.
- **Batch normalization and LeakyReLU activation**: Improves stability and learning.
- **Reshape**: Converts to a 7x7 feature map.
- **Transposed convolutional layers**: Progressively upscales the image and introduces spatial features.
- **Tanh activation**: Produces images in the range [-1, 1].

### Discriminator

- **Convolutional layers with LeakyReLU activation**: Extracts features from the input image.
- **Dropout layers**: Prevents overfitting.
- **Flatten layer**: Converts the feature map into a vector.
- **Dense layer with a single output neuron and sigmoid activation**: Classifies the input as real or fake.

## Loss Functions

- **Binary Crossentropy**: Measures the difference between the discriminator's output and the expected labels (real: 1, fake: 0).

## Training Process

### Load the Fashion-MNIST dataset

- Uses TensorFlow's built-in function to load the dataset.
- Preprocesses the images by normalization (-1, 1).

### Define the Generator and Discriminator models

- Refer to the code for detailed architecture configurations.

### Define optimizers

- Uses the Adam optimizer with a learning rate of 1e-4 for both the generator and discriminator.

### Training Loop

- Iterates over epochs and batches of images.
- For each batch:
  - Generates noise vectors.
  - Trains the discriminator on real and generated images.
  - Trains the generator to fool the discriminator.
- Periodically saves model checkpoints.
- Visualizes generated images at different epochs.

### Optional: Model Saving

- Saves the trained generator and discriminator models in HDF5 format (.h5) using `model.save()`.

## Visualization

- The notebook generates images at different epochs and displays them using Matplotlib.

## Disclaimer

The use of the provided GAN implementation is for educational and research purposes only. Adapt it responsibly and be mindful of potential biases in the dataset.

## Future Improvements

- Experiment with different network architectures (e.g., deeper models, residual connections).
- Explore alternative loss functions (e.g., WGAN, LSGAN) and optimizers.
- Incorporate techniques like spectral normalization for improved training stability.
- Evaluate the model's performance on a more complex dataset.
- Consider using techniques like progressive growing to generate higher-resolution images.


