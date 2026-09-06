#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

double f(double x){return x * log(x) - 2.0 * sin(x / 2.0) - x;}

int main()
{
    double a = 0.75;
    double b = 2.75;

    double sigma = 0.1;
    double epsilon = 0.01;

    int k = 0;
    cout << fixed << setprecision(4);
    cout << "Dichotomy method\n\n";
    cout << "+-----+----------+----------+-------------+-------------+-----------------------+----------+\n";
    cout << "| k   | x1       | x2       | f(x1)       | f(x2)       | [a_k, b_k]            | L_k      |\n";
    cout << "+-----+----------+----------+-------------+-------------+-----------------------+----------+\n";

    cout << "| " << left << setw(4) << k
        << "| " << setw(9) << "-"
        << "| " << setw(9) << "-"
        << "| " << setw(12) << "-"
        << "| " << setw(12) << "-"
        << "| [" << setw(8) << a
        << "; " << setw(8) << b << "]"
        << "| " << setw(9) << b - a << "|\n";

    while ((b - a) > sigma)
    {
        k++;

        double x1 = (a + b) / 2.0 - epsilon / 2.0;
        double x2 = x1 + epsilon;

        double fx1 = f(x1);
        double fx2 = f(x2);
        if (fx1 < fx2)
        {
            b = x2;
        }
        else
        {
            a = x1;
        }

        double L = b - a;

        cout << "| " << left << setw(4) << k
            << "| " << setw(9) << x1
            << "| " << setw(9) << x2
            << "| " << setw(12) << fx1
            << "| " << setw(12) << fx2
            << "| [" << setw(8) << a
            << "; " << setw(8) << b << "]"
            << "| " << setw(9) << L << "|\n";
    }

    cout << "+-----+----------+----------+-------------+-------------+-----------------------+----------+\n";

    double xMin = (a + b) / 2.0;
    cout << "\nFinal interval: [" << a << "; " << b << "]\n";
    cout << "Minimum point x* = " << xMin << "\n";
    cout << "Function value f(x*) = " << f(xMin) << "\n";
}