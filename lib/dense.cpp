#include "dense.h"

Dense::Dense(int inputSize, int outputSize)
{
    //this->weights = randMatrix(outputSize, inputSize);
    //this->bias = randMatrix(outputSize, 1);
    this->weights = randGausianMatrix(outputSize, inputSize);
    this->bias = randGausianMatrix(outputSize, 1);
}

VECTOR2D Dense::forward(VECTOR2D &input)
{
    this->input = input;
    VECTOR2D mDot = dot(this->weights, this->input);
    VECTOR2D output = add(mDot, this->bias);
    //std::cout << input.size() << " " << input[0].size() << "\n";
    //printMatrix(this->weights);
    return output;
}

VECTOR2D Dense::backward(VECTOR2D grad, double rate)
{
    VECTOR2D inputT = transpose(this->input);
    VECTOR2D weightGrad = dot(grad, inputT);
    VECTOR2D weightsT = transpose(this->weights);
    VECTOR2D inputGrad = dot(weightsT, grad);

    VECTOR2D mulWeights = multiply(weightGrad, rate);
    VECTOR2D w = subtract(this->weights, mulWeights);
    this->weights = w;

    VECTOR2D mulBias = multiply(grad, rate);
    VECTOR2D b = subtract(this->bias, mulBias);
    this->bias = b;

    return inputGrad;
}

