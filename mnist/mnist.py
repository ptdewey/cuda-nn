import idx2numpy
import numpy as np

# Function to read ubyte file
def read_ubyte_file(file_path):
    with open(file_path, 'rb') as f:
        return idx2numpy.convert_from_file(f)



# Function to extract images and labels from ubyte files
def extract_images_labels(images_file, labels_file):
    images = read_ubyte_file(images_file)
    labels = read_ubyte_file(labels_file)
    return images, labels


# Function to filter images and labels based on specified digits
def filter_digits(images, labels, digits):
    mask = np.isin(labels, digits)
    filtered_images = images[mask]
    filtered_labels = labels[mask]
    return filtered_images, filtered_labels


# Function to write ubyte file
def write_ubyte_file(file_path, data):
    with open(file_path, 'wb') as f:
        idx2numpy.convert_to_file(f, data)


# Define file paths for MNIST dataset (update with your file paths)
train_images_file = "train-images.idx3-ubyte"
train_labels_file = "train-labels.idx1-ubyte"
test_images_file = "t10k-images.idx3-ubyte"
test_labels_file = "t10k-labels.idx1-ubyte"

# Specify the new file paths for the filtered data
filtered_train_images_file = "filtered-train-images.idx3-ubyte"
filtered_train_labels_file = "filtered-train-labels.idx1-ubyte"
filtered_test_images_file = "filtered-test-images.idx3-ubyte"
filtered_test_labels_file = "filtered-test-labels.idx1-ubyte"

# Extract images and labels
train_images, train_labels = extract_images_labels(train_images_file, train_labels_file)
test_images, test_labels = extract_images_labels(test_images_file, test_labels_file)

# Specify the digits to keep (0 and 1 for binary classification)
digits_to_keep = [0, 1]

# Filter images and labels for the specified digits
filtered_train_images, filtered_train_labels = filter_digits(train_images, train_labels, digits_to_keep)
filtered_test_images, filtered_test_labels = filter_digits(test_images, test_labels, digits_to_keep)

# Write the filtered images and labels to new ubyte files
write_ubyte_file(filtered_train_images_file, filtered_train_images)
write_ubyte_file(filtered_train_labels_file, filtered_train_labels)
write_ubyte_file(filtered_test_images_file, filtered_test_images)
write_ubyte_file(filtered_test_labels_file, filtered_test_labels)

# Check the dimensions of the filtered datasets
print("Dimensions of filtered training data:", filtered_train_images.shape)
print("Dimensions of filtered test data:", filtered_test_images.shape)

# fi = read_ubyte_file(filtered_train_images_file)
# fi = read_ubyte_file(train_images_file)
# head = [fi[_] for _ in range(1)]
# print(head)
