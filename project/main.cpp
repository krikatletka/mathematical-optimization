/*
    main.cpp — запуск підготовки та лабораторних робіт.
*/
#include <iostream>
#include <exception>

#include "src/preparation.h"
#include "src/lab1.h"

int main()
{
    try
    {
        ProjectData data = prepareProject();

        runLab1(data.clean, data.noise.noisy);

        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}