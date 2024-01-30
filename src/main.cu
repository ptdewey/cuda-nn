#include <iostream>
#include <string>
#include <time.h>

#include "neural_network.hh"
#include "layers/linear_layer.hh"
#include "layers/relu_activation.hh"
#include "layers/sigmoid_activation.hh"
#include "layers/softmax_activation.hh"
#include "layers/tanh_activation.hh"
#include "nn_utils/nn_exception.hh"
#include "nn_utils/bce_cost.hh"
#include "nn_utils/ce_cost.hh"
#include "nn_utils/mse_cost.hh"

#include "coordinates_dataset.hh"
#include "mnist_dataset.hh"

float computeAccuracy(const Matrix& predictions, const Matrix& targets);


int main(int argc, char** argv) {
    // adjust these
    int print_epoch = 1;
    size_t batch_size = 2;
    size_t num_batches = 30000;
    int epochs = 3;
    size_t l1 = 512;
    size_t l2 = 128;

    // Update hyperparameters with command line args
    if (argc >= 2) {
        epochs = atoi(argv[1]); 
    }
    if (argc >= 4) {
        l1 = atoi(argv[2]);
        l2 = atoi(argv[3]);
    }
    if (argc >= 6) {
        batch_size = atoi(argv[4]);
        num_batches = atoi(argv[5]);
    }
    if (argc >= 7) {
        print_epoch = atoi(argv[6]);
    }

    srand(time(NULL));

    NeuralNetwork nn;

    // Coordinates Dataset
#ifdef COORD
    CoordinatesDataset dataset(100, 21);
    BCECost cf;
    nn.addLayer(new LinearLayer("linear_1", Shape(2, 30)));
    nn.addLayer(new ReLUActivation("relu_1"));
    nn.addLayer(new LinearLayer("linear_2", Shape(30, 1)));
    nn.addLayer(new SigmoidActivation("sigmoid_output"));

#elif BIN
    // mnist binary classifier: 
    // 0/1 dataset
    std::string labels_file = "../mnist/filtered-train-labels.idx1-ubyte";
    std::string image_file = "../mnist/filtered-train-images.idx3-ubyte";
    std::string test_image_file = "../mnist/filtered-test-images.idx3-ubyte";
    std::string test_labels_file = "../mnist/filtered-test-labels.idx1-ubyte";
    MNISTDataset dataset(batch_size, num_batches, image_file, labels_file);
    int ts = 2100;
    BCECost cf;
    nn.addLayer(new LinearLayer("linear_1", Shape(784, l1)));
    nn.addLayer(new ReLUActivation("relu_1"));
    nn.addLayer(new LinearLayer("linear_2", Shape(l1, l2)));
    nn.addLayer(new ReLUActivation("relu_2"));
    nn.addLayer(new LinearLayer("linear_3", Shape(l2, 1)));
    nn.addLayer(new SigmoidActivation("sigmoid_output"));
    // nn.addLayer(new LinearLayer("linear_3", Shape(l2, 2)));
    // nn.addLayer(new SoftmaxActivation("softmax_output"));
#else
    // full 0-9 MNIST dataset:
    std::string labels_file = "../mnist/train-labels.idx1-ubyte";
    std::string image_file = "../mnist/train-images.idx3-ubyte";
    std::string test_image_file = "../mnist/t10k-images.idx3-ubyte";
    std::string test_labels_file = "../mnist/t10k-labels.idx1-ubyte";
    MNISTDataset dataset(batch_size, num_batches, image_file, labels_file);
    int ts = 10000;

    // NOTE: choose cost function here
    MSECost cf;
    // CECost cf;

    nn.addLayer(new LinearLayer("linear_1", Shape(784, l1)));
    nn.addLayer(new ReLUActivation("relu_1"));
    // nn.addLayer(new TanhActivation("tanh_1"));
    nn.addLayer(new LinearLayer("linear_2", Shape(l1, l2)));
    nn.addLayer(new ReLUActivation("relu_2"));
    // nn.addLayer(new TanhActivation("tanh_2"));
    nn.addLayer(new LinearLayer("linear_3", Shape(l2, 10)));
    nn.addLayer(new SoftmaxActivation("softmax_output"));
#endif

    // network training
    Matrix Y;

    // PERF: parallelize training loop
    for (int epoch = 0; epoch < epochs + 1; epoch++) {
        float cost = 0.0;

        for (int batch = 0; batch < dataset.getNumOfBatches(); batch++) {
            Y = nn.forward(dataset.getBatches().at(batch));
            nn.backprop(Y, dataset.getTargets().at(batch), &cf);
            cost += cf.cost(Y, dataset.getTargets().at(batch));
        }

        if (print_epoch > -1 && epoch % print_epoch == 0) {
            std::cout 	<< "Epoch: " << epoch << ", Cost: " << cost / dataset.getNumOfBatches() << std::endl;
        }
    }

    // compute accuracy
    Y = nn.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));
    Y.copyDeviceToHost();

    float accuracy = computeAccuracy(
        Y, dataset.getTargets().at(dataset.getNumOfBatches() - 1));
    std::cout << "Last training batch accuracy: " << accuracy << std::endl;

    /**
     * TESTING
     */
#ifndef TEST
    MNISTDataset test_set(batch_size, ts / batch_size, test_image_file, test_labels_file);
    Matrix T;
    float test_acc = 0.0;
    for (int i = 1; i <= ts / batch_size; i++) {
        T = nn.forward(test_set.getBatches().at(test_set.getNumOfBatches() - i));
        T.copyDeviceToHost();
        test_acc += computeAccuracy(T, test_set.getTargets().at(test_set.getNumOfBatches() - i));
    }
    test_acc /= (ts / batch_size);
    std::cout << "Test Accuracy: " << test_acc << std::endl;
#endif

    return 0;
}

float computeAccuracy(const Matrix& predictions, const Matrix& targets) {
    int m = predictions.shape.x;
    int n = predictions.shape.y;
    int correct_predictions = 0;

    // multiclass case
    if (n > 1) {
        for (int i = 0; i < m; i++) {
            float prediction;
            float psum = 0;
            float best_pred = 0.0;
            for (int c = 0; c < n; c++) {
                psum += predictions[i *n + c];
                if (predictions[i * n + c] > best_pred) {
                    best_pred = predictions[i * n + c];
                    prediction = c;
                }
            }
            // std::cout << "Prediction: " << prediction <<  " Label: " << targets[i] << std::endl;
            if (prediction == targets[i]) {
                correct_predictions++;
            }
        }
    }
    else {
        // single class case
        for (int i = 0; i < m; i++) {
            float prediction = predictions[i] > 0.5 ? 1 : 0;
            if (prediction == targets[i]) {
                correct_predictions++;
            }
        }
    }

    return static_cast<float>(correct_predictions) / m;
}

