#include <iostream>
#include <string>
#include <time.h>

#include "neural_network.hh"
#include "layers/linear_layer.hh"
#include "layers/relu_activation.hh"
#include "layers/sigmoid_activation.hh"
// #include "layers/softmax_activation.hh"
#include "nn_utils/nn_exception.hh"
#include "nn_utils/bce_cost.hh"
// #include "nn_utils/ce_cost.hh"

// #include "coordinates_dataset.hh"
#include "mnist_dataset.hh"

// TODO: clean up commented code at some point

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
    int print_epoch = 25;
    size_t batch_size = 32;
    size_t num_batches = 256;
    int epochs = 125;

    size_t l1 = 1700;
    size_t l2 = 28;

    if (argc >= 2) {
        epochs = atoi(argv[1]);
    }
    if (argc >= 4) {
        l1 = atoi(argv[2]);
        l2 = atoi(argv[3]);
    }
    if (argc >= 5) {
        print_epoch = atoi(argv[4]);
    }
    if (argc >= 6) {
        cudaSetDevice(atoi(argv[5]));
    }

    srand( time(NULL) );
    BCECost bce_cost;
    // CrossEntropyLoss ce_cost;
    NeuralNetwork nn;

    // Coordinates Dataset
    // CoordinatesDataset dataset(batch_size, num_batches);
    // nn.addLayer(new LinearLayer("linear_1", Shape(2, 30)));
    // nn.addLayer(new ReLUActivation("relu_1"));
    // nn.addLayer(new LinearLayer("linear_2", Shape(30, 1)));
    // nn.addLayer(new SigmoidActivation("sigmoid_output"));

    // mnist: 
    // 0/1 dataset
    std::string image_file = "../mnist/filtered-train-images.idx3-ubyte";
    std::string labels_file = "../mnist/filtered-train-labels.idx1-ubyte";
    // full dataset:
    // std::string labels_file = "../mnist/train-labels.idx1-ubyte";
    // std::string image_file = "../mnist/train-images.idx3-ubyte";

    MNISTDataset dataset(batch_size, num_batches, image_file, labels_file);

    nn.addLayer(new LinearLayer("linear_1", Shape(784, l1)));
    nn.addLayer(new ReLUActivation("relu_1"));
    nn.addLayer(new LinearLayer("linear_2", Shape(l1, l2)));
    nn.addLayer(new ReLUActivation("relu_2"));
    // nn.addLayer(new LinearLayer("linear_3", Shape(l1, 1)));
    nn.addLayer(new LinearLayer("linear_3", Shape(l2, 1)));
    // nn.addLayer(new LinearLayer("linear_3", Shape(l2, 10)));
    nn.addLayer(new SigmoidActivation("sigmoid_output"));
    // nn.addLayer(new SoftmaxActivation("softmax_output"));

    // network training
    Matrix Y;

    for (int epoch = 0; epoch < epochs + 1; epoch++) {
        float cost = 0.0;

        for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) {
            Y = nn.forward(dataset.getBatches().at(batch));
            nn.backprop(Y, dataset.getTargets().at(batch));
            cost += bce_cost.cost(Y, dataset.getTargets().at(batch));
            // cost += ce_cost.cost(Y, dataset.getTargets().at(batch));
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
