#include "network.h"


VECTOR2D predict(NETWORK &network, VECTOR2D &input)
{
    VECTOR2D output = input;
    int netSize = network.size();

    for(int i = 0; i < netSize; i++){
        //output = ((Dense*)&(network[i]))->forward(output);
        //std::cout << output.size() << '\n';
        //std::cout << i << '\n';
        //VECTOR2D outputT = transpose(output);
        VECTOR2D tempOut;
        tempOut = network[i]->forward(output);
        output = tempOut;
        //output = static_cast<Dense*>(&network[i])->forward(output);
    }
    return output;
}

void train(NETWORK &network, VECTOR2D &inputData, VECTOR2D &inputResults, double eps, double rate, int cycles, bool verbose)
{
    int dataSize = inputData.size();
    int netSize = network.size();
    VECTOR shuffleMap;
    time_t timeS;
    for(int i = 0; i < cycles; i++){
        timeS = time(&timeS);
        double loss = 0.0;
        VECTOR2D output;

        for(int j = 0; j < dataSize; j++){
            shuffleMap = randIntMatrix(dataSize, dataSize);

            VECTOR2D d, r, dT, rT;
            VECTOR2D grad;

            d.resize(1);
            dT.resize(1);
            d[0].resize(dataSize);

            r.resize(1);
            rT.resize(1);
            r[0].resize(dataSize);
            //if(verbose){std::cout << "Data Index: " << j << "\n";}
            dT[0] = inputData[(int)shuffleMap[j]];
            rT[0] = inputResults[(int)shuffleMap[j]];

            d = transpose(dT);
            r = transpose(rT);

            //std::cout << d.size() << " " << d[0].size() << "\n";
            //printMatrix(d);
            output = predict(network, d);
            //std::cout << "Pred\n";
            
            loss += squareDiff(output, r);
            //std::cout << "diff\n";
            grad = prime(output, r);
            VECTOR2D tempGrad;
            for(int k = netSize-1; k >= 0; k--){
                tempGrad = network[k]->backward(grad, rate);
                grad = tempGrad;
                
            }
            
            
        }
        loss = loss/(double)dataSize;
        time_t timeE = time(&timeE);
        if(verbose){
            std::cout << "Epoch: " << i << "/" << cycles << " Error: " << loss 
            << " Data Size: " << dataSize << " Threads: " << omp_get_num_procs() 
            << " Time(s): " << (timeE - timeS) << '\n';
        }
    }
}

void run(NETWORK &network, VECTOR2D &inputData, VECTOR2D &inputResults)
{

    int dataSize = inputData.size();

        VECTOR2D d, r, dT, rT;
        VECTOR2D output;

        d.resize(1);
        dT.resize(1);
        //d[0].resize(dataSize);

        r.resize(1);
        rT.resize(1);
        //r[0].resize(dataSize);

        double loss = 0;
    for(int j = 0; j < dataSize; j++){
        dT[0] = inputData[j];
        rT[0] = inputResults[j];

        d = transpose(dT);
        r = transpose(rT);

        //printMatrix(dT);
       output = predict(network, d);
       loss += squareDiff(output, r);
       //VECTOR out = transpose(output).at(0);
       //VECTOR
       //std::cout << "Prediction: "  << argMax(transpose(output).at(0)) << " Expectation: " << argMax(transpose(r).at(0)) << "\n"; 
       
       std::cout << "Prediction: "  << "\n"; 
        //printMatrix(output); 
        std::cout<< argMax(output) << "\n";
        std::cout << " Expectation:\n";
        //printMatrix(r); 
        std::cout<< argMax(r) << "\n";
        std::cout << '\n';
    }
    std::cout << "Loss: " << loss << "\n";
}
