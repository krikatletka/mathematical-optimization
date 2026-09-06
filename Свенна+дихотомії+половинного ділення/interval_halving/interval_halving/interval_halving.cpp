#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

double f(double x){ return x * log(x) - 2.0 * sin(x / 2.0) - x;}
void printLine(){std::cout << "+-----+----------+----------+----------+-------------+-------------+-------------+-----------------------+----------+\n";}

int main()
{
    double a = 0.75;
    double b = 2.75;
    double sigma = 0.1;
    int k = 0;
    double xm = (a + b) / 2.0;
    double fm = f(xm);

    std::cout << fixed << setprecision(4);

    std::cout << "\nHalf division method\n\n";

    printLine();

    std::cout << "| " << left
        << setw(4) << "k"
        << "| " << setw(9) << "x1"
        << "| " << setw(9) << "xm"
        << "| " << setw(9) << "x2"
        << "| " << setw(12) << "f(x1)"
        << "| " << setw(12) << "f(xm)"
        << "| " << setw(12) << "f(x2)"
        << "| " << setw(22) << "[a_k, b_k]"
        << "| " << setw(9) << "L_k"
        << "|\n";

    printLine();

    std::cout << "| " << left
        << setw(4) << k
        << "| " << setw(9) << "-"
        << "| " << setw(9) << "-"
        << "| " << setw(9) << "-"
        << "| " << setw(12) << "-"
        << "| " << setw(12) << "-"
        << "| " << setw(12) << "-"
        << "| [" << setw(8) << a
        << "; " << setw(8) << b << "]"
        << "| " << setw(9) << b - a
        << "|\n";

    while ((b - a) > sigma)
    {
        k++;

        double L = b - a;

        double x1 = a + L / 4.0;
        double x2 = b - L / 4.0;

        double fx1 = f(x1);
        double fx2 = f(x2);

        double currentXm = xm;
        double currentFm = fm;

        if (fx1 < currentFm)
        {
            b = currentXm;
            xm = x1;
            fm = fx1;
        }
        else if (fx2 < currentFm)
        {
            a = currentXm;
            xm = x2;
            fm = fx2;
        }
        else
        {
            a = x1;
            b = x2;
        }

        double newL = b - a;

        cout << "| " << left
            << setw(4) << k
            << "| " << setw(9) << x1
            << "| " << setw(9) << currentXm
            << "| " << setw(9) << x2
            << "| " << setw(12) << fx1
            << "| " << setw(12) << currentFm
            << "| " << setw(12) << fx2
            << "| [" << setw(8) << a
            << "; " << setw(8) << b << "]"
            << "| " << setw(9) << newL
            << "|\n";
    }
    printLine();
    std::cout << "\nFinal interval: [" << a << "; " << b << "]\n";
    std::cout << "Minimum point x* = " << xm << "\n";
    std::cout << "Function value f(x*) = " << fm << "\n";

}