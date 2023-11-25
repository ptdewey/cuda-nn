#pragma once
#include "matrix.hh"

class CECost {
public:
	float cost(Matrix predictions, Matrix target);
	Matrix dCost(Matrix predictions, Matrix target, Matrix dY);
};
