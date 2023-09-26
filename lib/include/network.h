#pragma once
#include <iostream>
#include <vector>
#include <time.h>
#include "layer.h"
#include "matrixOperations.h"
#include "cost.h"
#include "dense.h"

typedef std::vector<Layer*> NETWORK;

VECTOR2D predict(NETWORK &network, VECTOR2D &input);

void train(NETWORK &network, VECTOR2D &inputData, VECTOR2D &inputResults, double eps, double rate, int cycles, bool verbose);

void run(NETWORK &network, VECTOR2D &inputData, VECTOR2D &inputResults);