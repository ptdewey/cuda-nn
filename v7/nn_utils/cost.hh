#pragma once

#include <iostream>

#include "matrix.hh"

class Cost {
public:
    virtual ~Cost() = 0;

    virtual float cost(Matrix predictions, Matrix target) = 0;
    virtual Matrix dCost(Matrix predictions, Matrix target, Matrix dy) = 0;
};

inline Cost::~Cost() {}

