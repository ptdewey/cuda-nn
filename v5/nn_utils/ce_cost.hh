#pragma once
#include "matrix.hh"
#include "cost.hh"

class CECost : public Cost {
public:
    CECost();
    ~CECost();

	float cost(Matrix predictions, Matrix target);
	Matrix dCost(Matrix predictions, Matrix target, Matrix dY);
};
