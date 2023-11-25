import idx2numpy
import matplotlib.pyplot as plt

# Function to read ubyte file
def read_ubyte_file(file_path):
    with open(file_path, 'rb') as f:
        return idx2numpy.convert_from_file(f)

# Function to display images
def display_images(images, labels, num_images=10):
    fig, axs = plt.subplots(10, 10, figsize=(20, 20))
    for i in range(num_images):
        axs[i // 10, i % 10].imshow(images[i], cmap='gray')
        axs[i // 10, i % 10].set_title(f"Label: {labels[i]}", fontsize=8)
        axs[i // 10, i % 10].axis('off')
    plt.tight_layout()
    plt.show()

# Define file paths for MNIST dataset (update with your file paths)
images_file = "filtered-train-images.idx3-ubyte"
labels_file = "filtered-train-labels.idx1-ubyte"

# Extract images and labels
images, labels = read_ubyte_file(images_file), read_ubyte_file(labels_file)

# Display the first 100 images
display_images(images, labels, num_images=99)
