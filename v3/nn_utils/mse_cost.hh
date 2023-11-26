#pragma once
#include "matrix.hh"
#include "cost.hh"

class MSECost : public Cost {
public:
    MSECost();
    ~MSECost();

    float cost(Matrix predictions, Matrix target);
    Matrix dCost(Matrix predictions, Matrix target, Matrix dY);
};
