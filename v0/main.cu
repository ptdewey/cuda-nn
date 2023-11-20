#include <iostream>
#include <string>
#include <time.h>

#include "neural_network.hh"
#include "layers/linear_layer.hh"
#include "layers/relu_activation.hh"
#include "layers/sigmoid_activation.hh"
#include "nn_utils/nn_exception.hh"
#include "nn_utils/bce_cost.hh"

// #include "coordinates_dataset.hh"
// #include "csv_dataset.hh"
#include "mnist_dataset.hh"

// float computeAccuracy(const Matrix& predictions, const Matrix& targets);
float computeAccuracy(const Matrix& predictions, const Matrix& targets) {
    int m = predictions.shape.x;
    int correct_predictions = 0;

    for (int i = 0; i < m; i++) {
        float prediction = predictions[i] > 0.5 ? 1 : 0;
        if (prediction == targets[i]) {
            correct_predictions++;
        }
    }

    return static_cast<float>(correct_predictions) / m;
}

int main(int argc, char** argv) {
    // adjust these
    size_t batch_size = 32;
    size_t num_batches = 256;
    int epochs = 125;

    size_t l1 = 1800;
    size_t l2 = 28;

    if (argc >= 4) {
        epochs = atoi(argv[1]);
        l1 = atoi(argv[2]);
        l2 = atoi(argv[3]);
    }

    int print_epoch = 25;
    if (argc >= 5) {
        print_epoch = atoi(argv[4]);
    }
    if (argc >= 6) {
        cudaSetDevice(atoi(argv[5]));
    }

    srand( time(NULL) );
    BCECost bce_cost;
    NeuralNetwork nn;

    // Coordinates Dataset
    // CoordinatesDataset dataset(batch_size, num_batches);
    // nn.addLayer(new LinearLayer("linear_1", Shape(2, 30)));
    // nn.addLayer(new ReLUActivation("relu_1"));
    // nn.addLayer(new LinearLayer("linear_2", Shape(30, 1)));
    // nn.addLayer(new SigmoidActivation("sigmoid_output"));

    // mnist: 
    std::string image_file = "../mnist/filtered-train-images.idx3-ubyte";
    std::string labels_file = "../mnist/filtered-train-labels.idx1-ubyte";
    MNISTDataset dataset(batch_size, num_batches, image_file, labels_file);
    nn.addLayer(new LinearLayer("linear_1", Shape(784, l1)));
    nn.addLayer(new ReLUActivation("relu_1"));
    // nn.addLayer(new LinearLayer("linear_2", Shape(l1, l2)));
    // nn.addLayer(new ReLUActivation("relu_2"));
    nn.addLayer(new LinearLayer("linear_3", Shape(l1, 1)));
    // nn.addLayer(new LinearLayer("linear_3", Shape(l2, 1)));
    nn.addLayer(new SigmoidActivation("sigmoid_output"));

    // network training
    Matrix Y;

#ifdef PROFILE
    for (int epoch = 0; epoch < 2; epoch++) {
    int print_epoch = 1;
#else
    for (int epoch = 0; epoch < epochs + 1; epoch++) {
    // int print_epoch = 25;
#endif
        float cost = 0.0;

        for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) {
            Y = nn.forward(dataset.getBatches().at(batch));
            nn.backprop(Y, dataset.getTargets().at(batch));
            cost += bce_cost.cost(Y, dataset.getTargets().at(batch));
        }

        if (print_epoch > -1 && epoch % print_epoch == 0) {
            std::cout 	<< "Epoch: " << epoch
                << ", Cost: " << cost / dataset.getNumOfBatches()
                << std::endl;
        }
    }

    // compute accuracy
    Y = nn.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));
    Y.copyDeviceToHost();

    float accuracy = computeAccuracy(
        Y, dataset.getTargets().at(dataset.getNumOfBatches() - 1));
    std::cout << "Accuracy: " << accuracy << std::endl;

    return 0;
}

