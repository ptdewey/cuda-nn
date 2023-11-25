#pragma once
#include "matrix.hh"

class MSECost {
public:
    float cost(Matrix predictions, Matrix target);
    Matrix dCost(Matrix predictions, Matrix target, Matrix dY);
};
