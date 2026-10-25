cmake_minimum_required(VERSION 3.16)
project(HACKE LANGUAGES CXX)

# Lock in C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tell the compiler where .hpp header files live
include_directories(include)

# Strict compiler warnings  + Maximum Hardware Optimization
# -O3: Maximum speed optimization
# -march=native: Unlock AVX2/FMA instructions specific to your exact CPU
# -Wall -Wextra: Warn about sketchy C++ code immediately
add_compile_options(-O3 -march=native -Wall -Wextra)

# Create the final executable from source files
add_executable(hacke_bench src/main.cpp)