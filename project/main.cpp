/*
    main.cpp — запуск етапів проєкту
    та обробка помилок.
*/
#include <iostream>
#include <exception>

#include "src/preparation.h"

int main()
{
    try
    {
        ProjectData data = prepareProject();

        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}