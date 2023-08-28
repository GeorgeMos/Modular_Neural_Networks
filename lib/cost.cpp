#include "cost.h"

double squareDiff(VECTOR2D &prediction, VECTOR2D &expectation)
{
    VECTOR2D diff = subtract(prediction, expectation);
    VECTOR2D mPow = power(diff, 2);
    double mMean = mean(mPow);
    return mMean;
}

VECTOR2D prime(VECTOR2D &prediction, VECTOR2D &expectation)
{
    VECTOR2D diff = subtract(prediction, expectation);
    VECTOR2D mull = multiply(diff, 2);
    VECTOR2D mDiv = divide(mull, prediction.size());
    return mDiv;
}
