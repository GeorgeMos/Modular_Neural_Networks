#include "matrixOperations.h"

/**
 * @brief Returns the dot product of the 2 input matrices
 * 
 * @param a 
 * @param b 
 * @return VECTOR2D 
 */
VECTOR2D dot(VECTOR2D &a, VECTOR2D &b)
{
    int sizeaX = a.size();
    int sizeaY = a[0].size();

    int sizebX = b.size();
    int sizebY = b[0].size();

    VECTOR2D outMatrix;

    //Output vector resizing
    outMatrix.resize(sizeaX);
    for(int i = 0; i < sizeaX; i++){
        outMatrix[i].resize(sizebY);
    }
    if(sizeaY == sizebX){
        VECTOR2D bT = transpose(b);
        //sizebY becomes sizebX and vice-versa

        double rez = 0;
        //Itterate x for a
        #pragma omp parallel for collapse(2) private(rez)
        for(int ax = 0; ax < sizeaX; ax++){
            //Itterate x for b
            for(int xb = 0; xb < sizebY; xb++){
                //Itterate y for a and b
                for(int y = 0; y < sizebX; y++){
                    //std::cout << omp_get_thread_num() << "\n";
                    rez += a[ax][y] * bT[xb][y];
                }
                outMatrix[ax][xb] = rez;
                rez = 0; 
            }
        }
        return outMatrix;
    }
    else{
        throw std::invalid_argument("dot Invalid Invalid matrix sizes a.x: " + std::to_string(a.size()) + 
                                    " a.y: " + std::to_string(a[0].size()) + 
                                    " b.x: " + std::to_string(b.size()) + 
                                    " b.y: " + std::to_string(b[0].size()));
        return VECTOR2D();
    }
}


/**
 * @brief Transposes the input matrix
 * 
 * @param matrix 
 * @return VECTOR2D 
 */
VECTOR2D transpose(VECTOR2D &matrix)
{
    VECTOR2D outMatrix;
    int sizeX = matrix.size();
    int sizeY = matrix[0].size();

    //Resizing the output matrix
    outMatrix.resize(sizeY);
    for(int i = 0; i < sizeY; i++){
        outMatrix[i].resize(sizeX);
    }

    //Transposing
    #pragma omp parallel for
    for(int x = 0; x<sizeX; x++){
        for(int y = 0; y < sizeY; y++){
            outMatrix[y][x] = matrix[x][y];
        }
    }

    return outMatrix;
}

/**
 * @brief Prints the input matrix in a readable way
 * 
 * @param matrix 
 */
void printMatrix(VECTOR2D matrix)
{
    std::cout << "[";
    for(int i = 0; i<matrix.size(); i++){
        std::cout << "[";
        for(int j = 0; j < matrix[0].size(); j++){
            std::cout << matrix[i][j] << " ";
        }
        std::cout << "]\n";
    }
    std::cout << "]" << '\n';
}

/**
 * @brief Generates a matrix with randomised values (0 - 1)
 * 
 * @param sizeX 
 * @param sizeY 
 * @return VECTOR2D 
 */
VECTOR2D randMatrix(int sizeX, int sizeY)
{
    VECTOR2D outputMatrix;

    //Resizing
    outputMatrix.resize(sizeX);
    for(int i = 0; i < sizeX; i++){
        outputMatrix[i].resize(sizeY);
    }

    srand(time(NULL));
    for(int x = 0; x < sizeX; x++){
        for(int y = 0; y < sizeY; y++){
            outputMatrix[x][y] = (double)(rand() / static_cast<float>(RAND_MAX));
        }
    }
    return outputMatrix;
}

/**
 * @brief Element-Wise multiplication
 * 
 * @param matrix 
 * @param num 
 * @return VECTOR2D 
 */
VECTOR2D multiply(VECTOR2D &matrix, double num)
{
    int sizeX = matrix.size();
    int sizeY = matrix[0].size();

    VECTOR2D outputMatrix;

    //Resizing
    outputMatrix.resize(sizeX);
    for(int i = 0; i < sizeX; i++){
        outputMatrix[i].resize(sizeY);
    }

    #pragma omp parallel for collapse(2)
    for(int x = 0; x < sizeX; x++){
        for(int y = 0; y < sizeY; y++){
            outputMatrix[x][y] = matrix[x][y]*num;
        }
    }
    return outputMatrix;
}

VECTOR2D multiply(VECTOR2D &a, VECTOR2D &b)
{
    if(a.size() == b.size() && a[0].size() == b[0].size()){
        int sizeX = a.size();
        int sizeY = a[0].size();

        VECTOR2D outputMatrix;

        //Resizing
        outputMatrix.resize(sizeX);
        for(int i = 0; i < sizeX; i++){
            outputMatrix[i].resize(sizeY);
        }

        #pragma omp parallel for collapse(2)
        for(int x = 0; x < sizeX; x++){
            for(int y = 0; y < sizeY; y++){
                outputMatrix[x][y] = a[x][y] * b[x][y];
            }
        }
        return outputMatrix;
    }
    else{
        throw std::invalid_argument("multiply Invalid matrix sizes a.x: " + std::to_string(a.size()) + 
                                    " a.y: " + std::to_string(a[0].size()) + 
                                    " b.x: " + std::to_string(b.size()) + 
                                    " b.y: " + std::to_string(b[0].size()));
        return VECTOR2D();
    }
}

/**
 * @brief Element-Wise subtraction (a - b)
 * 
 * @param a 
 * @param b 
 * @return VECTOR2D 
 */
VECTOR2D subtract(VECTOR2D &a, VECTOR2D &b)
{

    if(a.size() == b.size() && a[0].size() == b[0].size()){
        int sizeX = a.size();
        int sizeY = a[0].size();

        VECTOR2D outputMatrix;

        //Resizing
        outputMatrix.resize(sizeX);
        for(int i = 0; i < sizeX; i++){
            outputMatrix[i].resize(sizeY);
        }

        #pragma omp parallel for collapse(2)
        for(int x = 0; x < sizeX; x++){
            for(int y = 0; y < sizeY; y++){
                outputMatrix[x][y] = a[x][y] - b[x][y];
            }
        }
        return outputMatrix;
    }
    else{
        throw std::invalid_argument("subtract Invalid matrix sizes a.x: " + std::to_string(a.size()) + 
                                    " a.y: " + std::to_string(a[0].size()) + 
                                    " b.x: " + std::to_string(b.size()) + 
                                    " b.y: " + std::to_string(b[0].size()));
        return VECTOR2D();
    }
}

VECTOR2D subtract(float a, VECTOR2D &b)
{
    int sizeX = b.size();
    int sizeY = b[0].size();

    VECTOR2D outputMatrix;

    //Resizing
    outputMatrix.resize(sizeX);
    for(int i = 0; i < sizeX; i++){
        outputMatrix[i].resize(sizeY);
    }

    #pragma omp parallel for collapse(2)
    for(int x = 0; x < sizeX; x++){
        for(int y = 0; y < sizeY; y++){
            outputMatrix[x][y] = a - b[x][y];
        }
    }
    return outputMatrix;
}

/**
 * @brief Element-Wise addition (a + b)
 * 
 * @param a 
 * @param b 
 * @return VECTOR2D 
 */
VECTOR2D add(VECTOR2D &a, VECTOR2D &b)
{
    if(a.size() == b.size() && a[0].size() == b[0].size()){
        int sizeX = a.size();
        int sizeY = a[0].size();

        VECTOR2D outputMatrix;

        //Resizing
        outputMatrix.resize(sizeX);
        for(int i = 0; i < sizeX; i++){
            outputMatrix[i].resize(sizeY);
        }

        #pragma omp parallel for collapse(2)
        for(int x = 0; x < sizeX; x++){
            for(int y = 0; y < sizeY; y++){
                outputMatrix[x][y] = a[x][y] + b[x][y];
            }
        }
        return outputMatrix;
    }
    else{
        throw std::invalid_argument("subtract Invalid matrix sizes a.x: " + std::to_string(a.size()) + 
                                    " a.y: " + std::to_string(a[0].size()) + 
                                    " b.x: " + std::to_string(b.size()) + 
                                    " b.y: " + std::to_string(b[0].size()));
        return VECTOR2D();
    }
}

/**
 * @brief Raises every element of the input matrix to the input power
 * 
 * @param matrix 
 * @param pow 
 * @return VECTOR2D 
 */
VECTOR2D power(VECTOR2D &matrix, double pow)
{
    int sizeX = matrix.size();
    int sizeY = matrix[0].size();

    VECTOR2D outputMatrix;

    //Resizing
    outputMatrix.resize(sizeX);
    for(int i = 0; i < sizeX; i++){
        outputMatrix[i].resize(sizeY);
    }
    #pragma omp parallel for collapse(2)
    for(int x = 0; x < sizeX; x++){
        for(int y = 0; y < sizeY; y++){
            //std::cout << omp_get_thread_num() << "\n";
            outputMatrix[x][y] = std::pow(matrix[x][y], pow);
        }
    }
    return outputMatrix;
}

/**
 * @brief Returns the mean value of the input matrix
 * 
 * @param matrix 
 * @return double 
 */
double mean(VECTOR2D &matrix)
{
    int sizeX = matrix.size();
    int sizeY = matrix[0].size();

    double sum = 0;
    double mean = 0;

    for(int x = 0; x < sizeX; x++){
        for(int y = 0; y < sizeY; y++){
            sum += matrix[x][y];
        }
    }

    mean = sum/(sizeX*sizeY);

    return mean;
}

/**
 * @brief Element-Wise Division (matrix/num)
 * 
 * @param matrix 
 * @param num 
 * @return VECTOR2D 
 */
VECTOR2D divide(VECTOR2D &matrix, double num)
{
    int sizeX = matrix.size();
    int sizeY = matrix[0].size();

    VECTOR2D outputMatrix;

    //Resizing
    outputMatrix.resize(sizeX);
    for(int i = 0; i < sizeX; i++){
        outputMatrix[i].resize(sizeY);
    }

    #pragma omp parallel for collapse(2)
    for(int x = 0; x < sizeX; x++){
        for(int y = 0; y < sizeY; y++){
            outputMatrix[x][y] = matrix[x][y]/num;
        }
    }
    return outputMatrix;
}

VECTOR2D tanh_m(VECTOR2D &matrix)
{
    int sizeX = matrix.size();
    int sizeY = matrix[0].size();

    VECTOR2D outputMatrix;

    //Resizing
    outputMatrix.resize(sizeX);
    for(int i = 0; i < sizeX; i++){
        outputMatrix[i].resize(sizeY);
    }

    for(int x = 0; x < sizeX; x++){
        for(int y = 0; y < sizeY; y++){
            outputMatrix[x][y] = std::tanh(matrix[x][y]);
        }
    }
    return outputMatrix;
}

VECTOR2D tanhPrime_m(VECTOR2D &matrix)
{
    VECTOR2D mTanh = tanh_m(matrix);
    VECTOR2D mPow = power(mTanh, 2);
    VECTOR2D mSub = subtract(1, mPow);

    return mSub;
}

int argMax(VECTOR2D &matrix)
{
    int size = matrix.size();

    double max = 0.0;
    int maxIndex = 0;
    for(int i = 0; i < size; i++){
        if(matrix[i][0] > max){
            max = matrix[i][0];
            maxIndex = i;
        }
    }
    return maxIndex;
}

VECTOR2D exp_m(VECTOR2D &matrix)
{
    int sizeX = matrix.size();
    int sizeY = matrix[0].size();

    VECTOR2D outputMatrix;

    //Resizing
    outputMatrix.resize(sizeX);
    for(int i = 0; i < sizeX; i++){
        outputMatrix[i].resize(sizeY);
    }

    #pragma omp parallel for collapse(2)
    for(int x = 0; x < sizeX; x++){
        for(int y = 0; y < sizeY; y++){
            outputMatrix[x][y] = std::exp(matrix[x][y]);
        }
    }
    return outputMatrix;
}

double sum(VECTOR2D &matrix)
{
    int sizeX = matrix.size();
    int sizeY = matrix[0].size();

    double sum = 0;

    for(int x = 0; x < sizeX; x++){
        for(int y = 0; y < sizeY; y++){
            sum += matrix[x][y];
        }
    }


    return sum;
}

/**
 * @brief Returns an identity matrix with the input dimentions
 * 
 * @param dimention 
 * @return VECTOR2D 
 */
VECTOR2D identity(int dimention)
{
    VECTOR2D outputMatrix;

    //Resizing
    outputMatrix.resize(dimention);
    for(int i = 0; i < dimention; i++){
        outputMatrix[i].resize(dimention);
    }
    for(int x = 0; x < dimention; x++){
        outputMatrix[x][x] = 1.0;
    }
    return outputMatrix;
}

/**
 * @brief Similar to numpy's broadcasting function. Extends a row vector to a full matrix
 * 
 * @param matrix 
 * @return VECTOR2D 
 */
VECTOR2D broadcast(VECTOR2D &matrix)
{
    int dimention = matrix[0].size();
    VECTOR2D outputMatrix;

    //Resizing
    outputMatrix.resize(dimention);
    for(int i = 0; i < dimention; i++){
        outputMatrix[i].resize(dimention);
    }
    for(int x = 0; x < dimention; x++){
        for(int y = 0; y < dimention; y++){
            outputMatrix[x][y] = matrix[0][y];
        }
    }
    return outputMatrix;
}
