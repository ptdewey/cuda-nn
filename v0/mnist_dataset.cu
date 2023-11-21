#include "mnist_dataset.hh"
#include <iostream>
#include <fstream>

MNISTDataset::MNISTDataset(size_t batch_size, size_t number_of_batches, const std::string& imagesFilePath, const std::string& labelsFilePath) :
    batch_size(batch_size), number_of_batches(number_of_batches) {
    // Load MNIST image data
    std::ifstream imagesFile(imagesFilePath, std::ios::binary);
    if (!imagesFile.is_open()) {
        std::cerr << "Error opening MNIST image file." << std::endl;
        return;
    }

    // Load MNIST label data
    std::ifstream labelsFile(labelsFilePath, std::ios::binary);
    if (!labelsFile.is_open()) {
        std::cerr << "Error opening MNIST label file." << std::endl;
        imagesFile.close();
        return;
    }

    // MNIST header information (you might need to adjust this based on the actual format of your MNIST data)
    int magic_number_images, num_images, num_rows, num_cols;
    imagesFile.read(reinterpret_cast<char*>(&magic_number_images), sizeof(magic_number_images));
    imagesFile.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    imagesFile.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    imagesFile.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
    labelsFile.read(reinterpret_cast<char*>(&magic_number_images), sizeof(magic_number_images));
    labelsFile.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));

    // Assuming MNIST images are 28x28, you might need to adjust this based on your actual MNIST data
    const int image_size = 28 * 28;
    const int label_size = 1;

    // Read MNIST data and create batches
    for (int i = 0; i < number_of_batches; i++) {
        batches.push_back(Matrix(Shape(batch_size, image_size)));
        targets.push_back(Matrix(Shape(batch_size, label_size)));

        batches[i].allocateMemory();
        targets[i].allocateMemory();

        for (int k = 0; k < batch_size; k++) {
            // Read MNIST image and normalize
            for (int pixel = 0; pixel < image_size; pixel++) {
                unsigned char pixel_value;
                imagesFile.read(reinterpret_cast<char*>(&pixel_value), sizeof(pixel_value));
                batches[i][k * image_size + pixel] = static_cast<float>(pixel_value) / 255.0;
                // int pv = batches[i][k * image_size + pixel] > 0.5 ? 1 : 0;
                // if (pixel % 28 == 0 && pixel != 0) {
                //     std::cout << "\n";
                // }
                // std::cout << pv << " ";
            }

            // Read MNIST label
            unsigned char label;
            labelsFile.read(reinterpret_cast<char*>(&label), sizeof(label));
            
            int l = static_cast<int>(label);
        
            // std::cout << "\n" << l << "\n";

            // Assign label based on MNIST class
            targets[i][k] = l;
        }

        batches[i].copyHostToDevice();
        targets[i].copyHostToDevice();
    }

    imagesFile.close();
    labelsFile.close();
}

int MNISTDataset::getNumOfBatches() {
	return number_of_batches;
}

std::vector<Matrix>& MNISTDataset::getBatches() {
	return batches;
}

std::vector<Matrix>& MNISTDataset::getTargets() {
	return targets;
}
