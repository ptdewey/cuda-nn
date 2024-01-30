#pragma once
#include "matrix.hh"
#include "cost.hh"

class BCECost : public Cost {
public:
    BCECost();
    ~BCECost();

	float cost(Matrix predictions, Matrix target);
	Matrix dCost(Matrix predictions, Matrix target, Matrix dY);
};
