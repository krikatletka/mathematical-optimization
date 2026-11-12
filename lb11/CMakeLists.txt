cmake_minimum_required(VERSION 3.16)
project(lr11_penalty_methods LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(lr11_part1 part1.cpp optimization.cpp)
add_executable(lr11_part2 part2.cpp optimization.cpp)
add_executable(lr11_tests tests.cpp optimization.cpp)

enable_testing()
add_test(NAME lr11_control_tests COMMAND lr11_tests)
