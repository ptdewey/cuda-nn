#pragma once

#include <vector>
#include "layers/nn_layer.hh"
#include "nn_utils/bce_cost.hh"
#include "nn_utils/ce_cost.hh"
#include "nn_utils/mse_cost.hh"

class NeuralNetwork {
private:
	std::vector<NNLayer*> layers;

	Matrix Y;
	Matrix dY;
	float learning_rate;

public:
	// NeuralNetwork(float learning_rate = 0.01);
	NeuralNetwork(float learning_rate = 0.1);
	~NeuralNetwork();

	Matrix forward(Matrix X);
	void backprop(Matrix predictions, Matrix target, Cost* cost);

	void addLayer(NNLayer *layer);
	std::vector<NNLayer*> getLayers() const;
};
