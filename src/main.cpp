#include <iostream>
#include "kernels.hpp"
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

class Matrix {
private:
    std::size_t n;
    std::vector<int> data;

public:
    // Initialize an N x N matrix filled with a default value (e.g. 0)
    Matrix(std::size_t size, int init_val = 0)
        : n(size), data(size * size, init_val) {}

    // Overload operator() for non-const access: mat(r, c) = 5;
    int& operator()(std::size_t row, std::size_t col) {
        return data[row * n + col];
    }

    // Overload operator() for const access: int x = mat(r, c);
    const int& operator()(std::size_t row, std::size_t col) const {
        return data[row * n + col];
    }

    std::size_t size() const { return n; }

    // Direct access to the raw underlying contiguous buffer
    int* raw_data() { return data.data(); }
    const int* raw_data() const { return data.data(); }
};
double getRandomValue(double min, double max){
        static std::random_device rd;
        static std::mt19937 gen(rd());

        std::uniform_real_distribution<double> dist(min,max);
        return dist(gen);

}
void printMatrix(Matrix A){
    // Print matrix
    std::size_t N=A.size();
    for (std::size_t r = 0; r < N; ++r) {
        for (std::size_t c = 0; c < N; ++c) {
            std::cout << A(r, c) << "\t ";
        }
        std::cout << "\n";
    }
}
void naiveGemm(Matrix A,
                Matrix B                               
){  
    std::cout<< "Matrix A"<< "\n";
    printMatrix(A);
    std::cout << "X" << "\n";

    std::cout<< "Matrix B"<< "\n";
    printMatrix(B);
    std::size_t size= A.size();
    Matrix C(size,0);
    auto start =std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < size; ++i) {
        for (std::size_t j = 0; j < size; ++j) {
            for (std::size_t k = 0; k < size; ++k) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
    auto end= std::chrono::steady_clock::now();
    double N = static_cast<double>(size);
    double totalFlops= 2*(N*N*N);

    std::chrono::duration<double> elapsed_seconds = end - start;
    double seconds = elapsed_seconds.count();
    double base= 10.0;
    double exponent=9.0;
    double result=std::pow(base, exponent);

    double GFLOPS= totalFlops/ (seconds*result);
    std::cout <<"EQUALS C"<< "\n";
    printMatrix(C);
    std::cout <<"Perfomance (GFLOPS):" <<GFLOPS << "\n" ;
}


int main() {
    std::size_t N = 10;
    Matrix matB(N);
    Matrix matA(N);

    // Populate matrix with values 1 to N*N
    
    for (std::size_t r = 0; r < N; ++r) {
        for (std::size_t c = 0; c < N; ++c) {
            matB(r, c) = getRandomValue(0.0,1000.0);
        }
    }
    for (std::size_t r = 0; r < N; ++r) {
        for (std::size_t c = 0; c < N; ++c) {
            matA(r, c) = getRandomValue(0.0,10000.0);
        }
    }
    
    naiveGemm( matA,matB);
   
    return 0;
}



