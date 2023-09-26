#pragma once
#include <iostream>
#include "matrixOperations.h"
#include "layer.h"

class Tanh: public Layer{
    public:
    VECTOR2D forward(VECTOR2D &input) override;
    VECTOR2D backward(VECTOR2D grad, double rate) override;

};

class Softmax: public Layer{
    public:
    VECTOR2D forward(VECTOR2D &input) override;
    VECTOR2D backward(VECTOR2D grad, double rate) override;

};