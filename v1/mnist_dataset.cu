#include "mnist_dataset.hh"
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>  // for time function
#include <cstdlib>  // for rand function

MNISTDataset::MNISTDataset(size_t batch_size, size_t number_of_batches, const std::string& imagesFilePath, const std::string& labelsFilePath) :
    batch_size(batch_size), number_of_batches(number_of_batches) {

    // uncomment for printing
    // int TEST = 0;

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

    const int image_size = 28 * 28;
    const int label_size = 1;

    // read mnist headers
    int magic_number_images, num_images, num_rows, num_cols;
    imagesFile.read(reinterpret_cast<char*>(&magic_number_images), sizeof(magic_number_images));
    imagesFile.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    imagesFile.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    imagesFile.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
    labelsFile.read(reinterpret_cast<char*>(&magic_number_images), sizeof(magic_number_images));
    labelsFile.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));

    // Read all MNIST data into pairs
    std::vector<std::pair<std::vector<float>, int>> imageLabelPairs;

    // Read until the file is empty
    while (!imagesFile.eof() && !labelsFile.eof()) {
        std::vector<float> image(image_size);

        for (int pixel = 0; pixel < image_size; ++pixel) {
            unsigned char pixel_value;
            imagesFile.read(reinterpret_cast<char*>(&pixel_value), sizeof(pixel_value));
            if (!imagesFile.eof()) {
                image[pixel] = static_cast<float>(pixel_value) / 255.0;
            }
        }

        if (!labelsFile.eof()) {
            unsigned char label;
            labelsFile.read(reinterpret_cast<char*>(&label), sizeof(label));
            imageLabelPairs.push_back(std::make_pair(image, static_cast<int>(label)));
        }
    }

    // Shuffle the pairs using a custom shuffling approach
    srand(static_cast<unsigned int>(time(nullptr)));  // seed for rand()

    for (size_t i = 0; i < imageLabelPairs.size() - 1; ++i) {
        size_t j = i + rand() % (imageLabelPairs.size() - i);

        // Swap pairs
        std::swap(imageLabelPairs[i], imageLabelPairs[j]);
    }

    // Create batches
    for (size_t i = 0; i < number_of_batches; ++i) {
        batches.push_back(Matrix(Shape(batch_size, image_size)));
        targets.push_back(Matrix(Shape(batch_size, label_size)));

        batches[i].allocateMemory();
        targets[i].allocateMemory();

        for (size_t k = 0; k < batch_size; ++k) {
            // Copy shuffled data to batches
            const auto& pair = imageLabelPairs[i * batch_size + k];
            const std::vector<float>& image = pair.first;
            int label = pair.second;

            for (int pixel = 0; pixel < image_size; ++pixel) {
                batches[i][k * image_size + pixel] = image[pixel];
                #ifdef TEST
                if (pixel % 28 == 0 && pixel != 0) {
                    std::cout << "\n";
                }
                int pv = batches[i][k * image_size + pixel] > 0.5 ? 1 : 0;
                std::cout << pv << " ";
                #endif
            }
            targets[i][k] = static_cast<float>(label);
            #ifdef TEST
            std::cout << "\n" << targets[i][k] << "\n";
            #endif
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
