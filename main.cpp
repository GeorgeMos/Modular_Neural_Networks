#include<iostream>
#include<string>
#include <vector>

#include "lib/matrixOperations.h"
#include "lib/network.h"
#include "lib/dense.h"
#include "lib/activations.h"
#include "lib/mnist.h"



void runXor(){
    VECTOR2D trainData = 
{
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
};

VECTOR2D trainResults = 
{
    {0},
    {1},
    {1},
    {0}
};
    NETWORK network = {
        new Dense(2, 3),
        new Softmax(),
        new Dense(3, 1),
        new Softmax()
    };

    std::cout << "Starting training\n";
    train(network, trainData, trainResults, 0.001, 0.01, 1000, true);
    std::cout << "Training Complete\n";
    run(network, trainData, trainResults);
    
}


void trainMnist(NETWORK &network, int epochs, int imageNum, int batchNum, int batchSize, double rate, bool verbose){
    VECTOR2D trainData;
    VECTOR2D trainResults;
    VECTOR2D runData;
    VECTOR2D runResults;
    try{
        
        VECTOR2D mnistTrainData;
        VECTOR2D mnistTrainResults;
        VECTOR2D mnistRunData; 
        VECTOR2D mnistRunResults;

        
        ReadMNISTimage("datasets/mnist/training/train-images.idx3-ubyte",imageNum,784,mnistTrainData);
        ReadMNISTlabel("datasets/mnist/training/train-labels.idx1-ubyte",imageNum,1,mnistTrainResults);
        ReadMNISTimage("datasets/mnist/testing/t10k-images.idx3-ubyte",10,784,mnistRunData);
        ReadMNISTlabel("datasets/mnist/testing/t10k-labels.idx1-ubyte",10,1,mnistRunResults);

        std::cout << "Data Loading Complete!\n";
        
        trainData.resize(batchSize);
        for(int j = 0; j < batchSize; j++){
            trainData[j] = mnistTrainData[j + batchNum*batchSize];
        }

        trainResults.resize(batchSize);
        for(int i = 0; i < batchSize; i++){
            trainResults[i].resize(10, 0.0);
            trainResults[i][mnistTrainResults[i + batchNum*batchSize][0]] = 1.0;
        }


        
        

        runData = mnistRunData;
        runResults.resize(mnistRunResults.size());
        for(int i = 0; i < runResults.size(); i++){
            runResults[i].resize(10, 0.0);
            runResults[i][mnistRunResults[i][0]] = 1.0;
        }
        
        std::cout << "Starting training\n";
        train(network, trainData, trainResults, 0.001, rate, epochs, verbose);
        std::cout << "Training Complete\n";
        //run(network, trainData, trainResults);
        
    }
    catch(std::invalid_argument const& ex){
         std::cout << ex.what() << '\n';
    }
}

void runMnist(NETWORK network, int imageNum){
    VECTOR2D trainData;
    VECTOR2D trainResults;
    VECTOR2D runData;
    VECTOR2D runResults;
    try{
        
        VECTOR2D mnistTrainData;
        VECTOR2D mnistTrainResults;
        VECTOR2D mnistRunData; 
        VECTOR2D mnistRunResults;

        
        ReadMNISTimage("datasets/mnist/training/train-images.idx3-ubyte",imageNum,784,mnistTrainData);
        ReadMNISTlabel("datasets/mnist/training/train-labels.idx1-ubyte",imageNum,1,mnistTrainResults);
        ReadMNISTimage("datasets/mnist/testing/t10k-images.idx3-ubyte",imageNum,784,mnistRunData);
        ReadMNISTlabel("datasets/mnist/testing/t10k-labels.idx1-ubyte",imageNum,1,mnistRunResults);

        trainData = mnistTrainData;
        trainResults.resize(mnistTrainResults.size());
        for(int i = 0; i < trainResults.size(); i++){
            trainResults[i].resize(10, 0.0);
            trainResults[i][mnistTrainResults[i][0]] = 1.0;
        }

        

        runData = mnistRunData;
        runResults.resize(mnistRunResults.size());
        for(int i = 0; i < runResults.size(); i++){
            runResults[i].resize(10, 0.0);
            runResults[i][mnistRunResults[i][0]] = 1.0;
        }

        //run(network, runData, runResults);
        run(network, trainData, trainResults);

    }
    catch(std::invalid_argument const& ex){
         std::cout << ex.what() << '\n';
    }

}

void storeNetwork(std::string filename, NETWORK &network){
    std::ofstream file;
    file.open(filename);
    int sx;
    int sy;
    for(int i = 0; i < network.size(); i++){
        sx = network.at(i)->weights.size();
        if(sx > 0){
            sy = network.at(i)->weights.at(0).size();
            for(int x = 0; x < sx; x++){
                for(int y = 0; y < sy; y++){
                    file << network[i]->weights.at(x).at(y) << " ";
                }
                file << "\n";
            }
        }
        file << "#\n";
    }
    file.close();
}

std::vector<std::string> split(std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}

void readNetwork(std::string filename, NETWORK &network){
    std::ifstream file;
    file.open(filename);
    int size = network.size();
    network.resize(size);
    std::string delimiter = " ";
    std::vector<std::string> lineV;
    std::string line;
    int n = 0;
    int x = 0;

    while(getline(file, line)){
        if(!line.compare("#")){
            n++;
            x = 0;
        }
        lineV = split(line, delimiter);
        for(int i = 0; i < lineV.size(); i++){
            if(!lineV[i].compare("")){
                x++;
            }
            else{
                if(line.compare("#") && lineV.size() > 1){
                    double num = std::stod(lineV.at(i));
                    network.at(n)->weights.at(x).at(i) = num;
                }
            }
        }

    }

    file.close();
}


int main(int argc, char** argv){
    
    int numEpochs, numImages;
    std::string fileName;


    NETWORK network = {
        new Dense(28*28, 40),
        new Softmax(),
        new Dense(40, 10),
        new Softmax()
    };

    int batchSize;
    
    if(argc == 5){
        numImages = atoi(argv[1]);
        numEpochs = atoi(argv[2]);
        
        if(numImages == 0 || numEpochs == 0){
            std::cout << "Wrong input parameters\n";
            return 0;
        }else{
            batchSize = atoi(argv[3]);
            if(batchSize != 0){
                for(int i = 0; i < numImages/batchSize; i++){
                    std::cout << "Training batch number: " << i << "\n";
                    trainMnist(network, numEpochs, numImages, i, batchSize, 0.02, true);
                }
            }else{
                trainMnist(network, numEpochs, numImages, 0, numImages, 0.02, true);
            }
            storeNetwork(argv[4], network);
        }
    }
    else if(argc == 3){
        numImages = atoi(argv[1]);
        if(numImages == 0){
            std::cout << "Wrong input parameters\n";
            return 0;
        }
        else{
            readNetwork(argv[2], network);
            runMnist(network, numImages);
        }
    }
    else{
        std::cout << "Usage: ./ml {Number of Images}, {Number of Epochs}, {batch size (0=no batch)}, {Model Output File} For training \nor\n ./ml {Number of Images} {Model Input File} for running\n";
        return 0;
    }
    
    return 0;
}