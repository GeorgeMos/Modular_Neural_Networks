#pragma once
#include <iostream>
#include <vector>
#include "matrixOperations.h"

double squareDiff(VECTOR2D &prediction, VECTOR2D &expectation);

VECTOR2D prime(VECTOR2D &prediction, VECTOR2D &expectation);