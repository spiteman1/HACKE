#include <iostream>
#include "kernels.hpp"
#include <vector>

void sanity_check() {
    std::cout << "[HACKE] Toolchain online. Ready for Objective 1.\n";
}

class Matrix{
    private:
    std::size_t n;
    std::vector<int> data;

    public:
    // Initialize an N x N matrix filled with a default value (e.g. 0)
    Matrix(std::size_t size, int init_val =0 )
    : n(size), data(size * size, init_val) {} 
    
    // Overload operator() for non-const access: mat(r, c) = 5;
    int& operator()(std::size_t row, std::size_t col){
        return data[row*n +col];
    }
    // Overload operator() for const access: int x = mat(r, c);
    const int& operator()(std::size_t row, std::size_t col) const{
        return data[row*n +col];
    }

    std::size_t size() const {return n; }

    //Direct access to the raw underlying contigous buffer
    int* raw_data() {return data.data();}
    const int* raw_data() const {return data.data();}
};

int main() {
    std:: size_t N=3;
    Matrix mat(N);

    // Populate matrix with values 1 to N*N
    int counter = 1;
    for (std::size_t r = 0; r < N; ++r) {
        for (std::size_t c = 0; c < N; ++c) {
            mat(r, c) = counter++;
        }
    }

    // Print matrix
    for (std::size_t r = 0; r < N; ++r) {
        for (std::size_t c = 0; c < N; ++c) {
            std::cout << mat(r, c) << " ";
        }
        std::cout << "\n";
    }
    return 0;
}