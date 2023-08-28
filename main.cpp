#include<iostream>
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


void trainMnist(NETWORK &network, int epochs){
    VECTOR2D trainData;
    VECTOR2D trainResults;
    VECTOR2D runData;
    VECTOR2D runResults;
    try{
        
        VECTOR2D mnistTrainData;
        VECTOR2D mnistTrainResults;
        VECTOR2D mnistRunData; 
        VECTOR2D mnistRunResults;

        
        ReadMNISTimage("datasets/mnist/training/train-images.idx3-ubyte",1000,784,mnistTrainData);
        ReadMNISTlabel("datasets/mnist/training/train-labels.idx1-ubyte",1000,1,mnistTrainResults);
        ReadMNISTimage("datasets/mnist/testing/t10k-images.idx3-ubyte",10,784,mnistRunData);
        ReadMNISTlabel("datasets/mnist/testing/t10k-labels.idx1-ubyte",10,1,mnistRunResults);

        std::cout << "Data Loading Complete!\n";
        
        
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
        
        std::cout << "Starting training\n";
        train(network, trainData, trainResults, 0.001, 0.01, epochs, true);
        std::cout << "Training Complete\n";
        //run(network, trainData, trainResults);
        
    }
    catch(std::invalid_argument const& ex){
         std::cout << ex.what() << '\n';
    }
}

void runMnist(NETWORK network){
    VECTOR2D trainData;
    VECTOR2D trainResults;
    VECTOR2D runData;
    VECTOR2D runResults;
    try{
        
        VECTOR2D mnistTrainData;
        VECTOR2D mnistTrainResults;
        VECTOR2D mnistRunData; 
        VECTOR2D mnistRunResults;

        
        ReadMNISTimage("datasets/mnist/training/train-images.idx3-ubyte",100,784,mnistTrainData);
        ReadMNISTlabel("datasets/mnist/training/train-labels.idx1-ubyte",100,1,mnistTrainResults);
        ReadMNISTimage("datasets/mnist/testing/t10k-images.idx3-ubyte",10,784,mnistRunData);
        ReadMNISTlabel("datasets/mnist/testing/t10k-labels.idx1-ubyte",10,1,mnistRunResults);

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
        //TODO:Write the weights to a csv format
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
    //std::getline(file, line);
    int n = 0;
    int x = 0;

    while(getline(file, line)){
        if(!line.compare("#")){
            //std::cout << "Layer: " << n << "\n";
            n++;
            x = 0;
        }
        lineV = split(line, delimiter);
        //std::cout << "Size: " << lineV.size() << "\n";
        for(int i = 0; i < lineV.size(); i++){
            if(!lineV[i].compare("")){
                //std::cout << "x: " << x << "\n";
                x++;
            }
            else{
                if(line.compare("#") && lineV.size() > 1){
                    //std::cout << lineV[i] << "-";
                    double num = std::stod(lineV.at(i));
                    network.at(n)->weights.at(x).at(i) = num;
                    //std::cout << network[0]->weights[x][i] << "\n";
                }
            }
        }

    }

    file.close();
}


int main(){
    
    NETWORK network = {
        new Dense(28*28, 40),
        new Softmax(),
        new Dense(40, 10),
        new Softmax()
    };
    

    
    //runXor();
    trainMnist(network, 3000);


    //storeNetwork("netFile.txt", network);
    //readNetwork("netFile.txt", network);
    runMnist(network);
    
    return 0;
}