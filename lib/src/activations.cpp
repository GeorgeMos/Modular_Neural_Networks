#include "../include/activations.h"

VECTOR2D Tanh::forward(VECTOR2D &input)
{
    this->input = input;
    VECTOR2D mTan = tanh_m(this->input);
    //printMatrix(mTan);
    return mTan;
}

VECTOR2D Tanh::backward(VECTOR2D grad, double rate)
{
    VECTOR2D mPrime = tanhPrime_m(this->input);
    VECTOR2D mMul = multiply(grad, mPrime);
    return mMul;
}

VECTOR2D Softmax::forward(VECTOR2D &input)
{
    this->input = input;
    VECTOR2D tmp = exp_m(this->input);
    double msum = sum(tmp);
    this->output = divide(tmp, msum);
    return this->output;
}

VECTOR2D Softmax::backward(VECTOR2D grad, double rate)
{
    int n = this->output.size();
    VECTOR2D ident = identity(n);
    VECTOR2D outputT = transpose(this->output);
    VECTOR2D outTbroad = broadcast(outputT);
    VECTOR2D mSub = subtract(ident, outTbroad);
    VECTOR2D outBroad = transpose(outTbroad);
    VECTOR2D mMul = multiply(mSub, outBroad);
    VECTOR2D mDot = dot(mMul, grad);
    return mDot;
}
