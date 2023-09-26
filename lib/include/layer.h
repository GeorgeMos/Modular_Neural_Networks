#pragma once
#include <iostream>
#include "matrixOperations.h"

class Layer
{
    public:
    VECTOR2D input;
    VECTOR2D output;
    VECTOR2D weights;
    VECTOR2D bias;
    virtual VECTOR2D forward(VECTOR2D &input);
    virtual VECTOR2D backward(VECTOR2D grad, double rate);
    virtual ~Layer();
};