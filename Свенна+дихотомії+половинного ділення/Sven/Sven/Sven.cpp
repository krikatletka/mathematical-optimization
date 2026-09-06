#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

using namespace std;

double f(double x)
{
    return x * log(x) - 2.0 * sin(x / 2.0) - x;
}

struct Iteration
{
    int k;
    double delta;
    double x;
    double fx;
    string comparison;
    string interval;
};

string makeInterval(double a, double b)
{
    ostringstream result;

    result << fixed << setprecision(4)
        << "[" << a << "; " << b << "]";

    return result.str();
}

void printLine()
{
    std::cout << "+----------+----------+----------+-------------+-------------------------+---------------------+\n";
}

void printTable(const vector<Iteration>& table)
{
    std::cout << fixed << setprecision(4);

    std::cout << "\nResults of Sven method\n";

    printLine();

    std::cout << "| " << left
        << setw(8) << "k"
        << "| " << setw(8) << "Delta"
        << "| " << setw(8) << "x_k"
        << "| " << setw(11) << "f(x_k)"
        << "| " << setw(23) << "f(x_k) < f(previous)?"
        << "| " << setw(19) << "[a_0; b_0]"
        << "|\n";

    printLine();

    for (const Iteration& row : table)
    {
        std::cout << "| " << left
            << setw(8) << row.k
            << "| " << setw(8) << row.delta
            << "| " << setw(8) << row.x
            << "| " << setw(11) << row.fx
            << "| " << setw(23) << row.comparison
            << "| " << setw(19) << row.interval
            << "|\n";
    }

    printLine();
}

int main()
{
    const double x0 = 1.75;
    const double delta0 = 1.0;

    vector<Iteration> table;

    double a0 = 0.0;
    double b0 = 0.0;
    double fx0 = f(x0);

    table.push_back({0, delta0,  x0, fx0, "", ""});

    double xRight = x0 + delta0;
    double fRight = f(xRight);

    table.push_back({1,  delta0, xRight, fRight, fRight < fx0 ? "+" : "-", ""});

    if (fRight < fx0)
    {
        double delta = delta0;

        double previousPreviousX = x0;
        double previousX = xRight;
        double previousF = fRight;

        int k = 2;
        int stepPower = 1;

        while (true)
        {
            double currentX = previousX + pow(2.0, stepPower) * delta;

            double currentF = f(currentX);

            if (currentF < previousF)
            {
                table.push_back({k, delta, currentX, currentF, "+", ""});

                previousPreviousX = previousX;
                previousX = currentX;
                previousF = currentF;
                k++;
                stepPower++;
            }
            else
            {
                a0 = min(previousPreviousX, currentX);
                b0 = max(previousPreviousX, currentX);
                table.push_back({k,  delta, currentX,  currentF, "-",  makeInterval(a0, b0)});

                break;
            }
        }
    }
    else
    {
        double delta = -delta0;
        double xLeft = x0 + delta;

        if (xLeft <= 0)
        {
            std::cout << "Error: x must be greater than zero.\n";
            return 1;
        }

        double fLeft = f(xLeft);

        if (fLeft >= fx0)
        {
            a0 = xLeft;
            b0 = xRight;
            table.push_back({2, delta, xLeft, fLeft, "-", makeInterval(a0, b0)});
        }
        else
        {
            table.push_back({2,  delta, xLeft, fLeft, "+", ""});

            double previousPreviousX = x0;
            double previousX = xLeft;
            double previousF = fLeft;

            int k = 3;
            int stepPower = 1;

            while (true)
            {
                double currentX = previousX + pow(2.0, stepPower) * delta;

                if (currentX <= 0)
                {
                    std::cout << "Error: x must be greater than zero.\n";
                    return 1;
                }

                double currentF = f(currentX);

                if (currentF < previousF)
                {
                    table.push_back({k, delta, currentX, currentF, "+", ""});

                    previousPreviousX = previousX;
                    previousX = currentX;
                    previousF = currentF;
                    k++;
                    stepPower++;
                }
                else
                {
                    a0 = min(currentX, previousPreviousX);
                    b0 = max(currentX, previousPreviousX);
                    table.push_back({k, delta, currentX, currentF, "-", makeInterval(a0, b0)});
                    break;
                }
            }
        }
    }

    std::cout << fixed << setprecision(4);

    std::cout << "Function: f(x) = x*ln(x) - 2*sin(x/2) - x\n";
    std::cout << "x0 = " << x0 << '\n';
    std::cout << "Delta0 = " << delta0 << '\n';
    printTable(table);
    std::cout << "\nInitial uncertainty interval:\n";
    std::cout << "[a0; b0] = [" << a0 << "; " << b0 << "]\n";
}