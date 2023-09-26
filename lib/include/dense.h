#pragma once
#include <iostream>
#include "matrixOperations.h"
#include "layer.h"
class Dense: public Layer
{
    public:
    //VECTOR2D input;
    Dense(int inputSize, int outputSize);
    VECTOR2D forward(VECTOR2D &input) override;
    VECTOR2D backward(VECTOR2D grad, double rate) override;
};