cc = g++
fileName = main.cpp
target = ml

sources = lib/matrixOperations.cpp lib/layer.cpp lib/dense.cpp lib/network.cpp lib/cost.cpp lib/mnist.cpp lib/activations.cpp

all: 
	$(cc) -march=native -ffast-math -O3 -fopenmp -lpthread $(fileName) $(sources) -o $(target)

win:
	$(cc) $(fileName) $(sources) -o $(target).exe -fopenmp -lpthread -march=native -ffast-math


clean:
	rm $(target)