#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "matrixOperations.h"

int ReverseInt (int i);
void ReadMNISTimage(std::string filename, int NumberOfImages, int DataOfAnImage,VECTOR2D &arr);
void ReadMNISTlabel(std::string filename, int NumberOfImages, int DataOfAnImage,VECTOR2D &arr);