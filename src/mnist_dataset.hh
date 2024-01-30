#pragma once

#include "nn_utils/matrix.hh"
#include <vector>
#include <string>

class MNISTDataset {
private:
    size_t batch_size;
    size_t number_of_batches;

    std::vector<Matrix> batches;
    std::vector<Matrix> targets;

public:
    MNISTDataset(size_t batch_size, size_t number_of_batches, const std::string& mnistImagePath, const std::string& mnistLabelPath);

    int getNumOfBatches();
    std::vector<Matrix>& getBatches();
    std::vector<Matrix>& getTargets();
};
