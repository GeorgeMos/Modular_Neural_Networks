#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <cmath>
#include <string>
#include <thread>

//#define USE_OPENMP
//#define OMP_DOT_MATRIX

#ifdef USE_OPENMP
#include <omp.h>
#endif
#include <algorithm>

typedef std::vector<std::vector<double>> VECTOR2D;
typedef std::vector<double> VECTOR;

/*
VECTOR2D dot(VECTOR2D &a, VECTOR &b);

VECTOR transpose(VECTOR &matrix);

void printMatrix(VECTOR matrix);

VECTOR multiply(VECTOR matrix, double num);

VECTOR subtract(VECTOR &a, VECTOR &b);

VECTOR power(VECTOR &matrix, int pow);

double mean(VECTOR &matrix);

VECTOR divide(VECTOR &matrix, double num);
*/


VECTOR2D dot(VECTOR2D &a, VECTOR2D &b);

VECTOR2D transpose(VECTOR2D &matrix);

void printMatrix(VECTOR2D matrix);

VECTOR2D randMatrix(int sizeX, int sizeY);

VECTOR randIntMatrix(int size, int range);

VECTOR2D randGausianMatrix(int sizeX, int sizeY);

VECTOR2D multiply(VECTOR2D &matrix, double num);

VECTOR2D multiply(VECTOR2D &a, VECTOR2D &b);

VECTOR2D subtract(VECTOR2D &a, VECTOR2D &b);

VECTOR2D subtract(float a, VECTOR2D &b);

VECTOR2D add(VECTOR2D &a, VECTOR2D &b);

VECTOR2D power(VECTOR2D &matrix, double pow);

double mean(VECTOR2D &matrix);

VECTOR2D divide(VECTOR2D &matrix, double num);

VECTOR2D tanh_m(VECTOR2D &matrix);

VECTOR2D tanhPrime_m(VECTOR2D &matrix);

int argMax(VECTOR2D &matrix);

VECTOR2D exp_m(VECTOR2D &matrix);

double sum(VECTOR2D &matrix);

VECTOR2D identity(int dimention);

VECTOR2D broadcast(VECTOR2D &matrix);