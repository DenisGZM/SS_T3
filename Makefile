deb: main.cpp
	mpicxx -O0 -g -std=c++14 main -o a.out
main: main.cpp
	mpicxx -O2 main.cpp -std=c++14 -o a.out
omp:
	mpicxx -02 main.cpp -fopenmp -std=c++14 -o a.out