#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>

using namespace std;

double f(double x)
{
    return x * log(x) - 2.0 * sin(x / 2.0) - x;
}

string fraction(int numerator, int denominator)
{
    ostringstream result;
    result << numerator << "/" << denominator;
    return result.str();
}

void printSearchLine()
{
    cout << "+-----+----------+----------------+-------------+\n";
}

void printTableLine()
{
    cout << "+----+----+---------------+----------+----------+----------+----------+-------------+-------------+----------+-----------------------------+\n";
}

int main()
{
    const double initialA = 0.75;
    const double initialB = 2.75;
    const double epsilon = 0.1;
    const double delta = epsilon / 10.0;

    double a = initialA;
    double b = initialB;

    const double initialLength = b - a;
    const double R = initialLength / epsilon;
    vector<long long> fibonacci = { 0, 1 };

    while (fibonacci.back() < R)
    {
        int size = fibonacci.size();

        fibonacci.push_back( fibonacci[size - 1] + fibonacci[size - 2]);
    }

    int N = static_cast<int>(fibonacci.size()) - 1;

    cout << fixed << setprecision(6);

    cout << "Fibonacci method\n\n";
    cout << "epsilon = " << epsilon << '\n';
    cout << "delta = epsilon / 10 = " << delta << '\n';
    cout << "L0 = " << initialLength << '\n';
    cout << "R = L0 / epsilon = " << R << "\n\n";

    /*
     * Таблиця пошуку N.
     */
    cout << "Table for finding N\n";

    printSearchLine();

    cout << "| " << left
        << setw(4) << "N"
        << "| " << setw(9) << "F_N"
        << "| " << setw(15) << "F_N >= R?"
        << "| " << setw(12) << "L0 / F_N"
        << "|\n";

    printSearchLine();

    for (int i = 2; i <= N; i++)
    {
        cout << "| " << left
            << setw(4) << i
            << "| " << setw(9) << fibonacci[i]
            << "| " << setw(15)
            << (fibonacci[i] >= R ? "yes" : "no")
            << "| " << setw(12)
            << initialLength / fibonacci[i]
            << "|\n";
    }

    printSearchLine();

    int m = N;
    double x1 = a + static_cast<double>(fibonacci[m - 2]) / fibonacci[m] * (b - a);
    double x2 = a + static_cast<double>(fibonacci[m - 1]) / fibonacci[m] * (b - a);

    double f1 = f(x1);
    double f2 = f(x2);
    int evaluations = 2;

    printTableLine();

    cout << "| " << left
        << setw(3) << "k"
        << "| " << setw(3) << "m"
        << "| " << setw(14) << "Fractions"
        << "| " << setw(9) << "a_k"
        << "| " << setw(9) << "b_k"
        << "| " << setw(9) << "x1"
        << "| " << setw(9) << "x2"
        << "| " << setw(12) << "f(x1)"
        << "| " << setw(12) << "f(x2)"
        << "| " << setw(9) << "L_k"
        << "| " << setw(28) << "Decision"
        << "|\n";

    printTableLine();

    for (int k = 0; k < N - 2; k++)
    {
        m = N - k;

        string fractions;

        if (m == 3)
        {
            double middle = (a + b) / 2.0;
            if (abs(x1 - middle) <= abs(x2 - middle))
            {
                x1 = middle;
            }
            else
            {
                x1 = middle;
                f1 = f2;
            }

            x2 = x1 + delta;
            f2 = f(x2);

            evaluations++;

            fractions = "1/2 ; 1/2";
        }
        else
        {
            fractions =
                fraction(fibonacci[m - 2], fibonacci[m])
                + " ; "
                + fraction(fibonacci[m - 1], fibonacci[m]);
        }

        double length = b - a;
        string decision;

        bool minimumOnLeft = f1 <= f2;

        if (minimumOnLeft)
        {
            decision = "f1 <= f2: b = x2";
        }
        else
        {
            decision = "f1 > f2: a = x1";
        }

        cout << "| " << left
            << setw(3) << k + 1
            << "| " << setw(3) << m
            << "| " << setw(14) << fractions
            << "| " << setw(9) << a
            << "| " << setw(9) << b
            << "| " << setw(9) << x1
            << "| " << setw(9) << x2
            << "| " << setw(12) << f1
            << "| " << setw(12) << f2
            << "| " << setw(9) << length
            << "| " << setw(28) << decision
            << "|\n";


        if (minimumOnLeft)
        {
            b = x2;
        }
        else
        {
            a = x1;
        }
        if (m > 4)
        {
            int nextM = m - 1;

            if (minimumOnLeft)
            {
                x2 = x1;
                f2 = f1;
                x1 = a + static_cast<double>(fibonacci[nextM - 2]) / fibonacci[nextM] * (b - a);
                f1 = f(x1);
            }
            else
            {
                x1 = x2;
                f1 = f2;
                x2 = a + static_cast<double>(fibonacci[nextM - 1]) / fibonacci[nextM] * (b - a);
                f2 = f(x2);
            }

            evaluations++;
        }
    }

    printTableLine();

    double finalLength = b - a;
    double xStar = (a + b) / 2.0;
    double fStar = f(xStar);
    double guaranteedLength = initialLength / fibonacci[N];

    cout << "\nFinal interval: [" << a << "; " << b << "]\n";
    cout << "L_final = " << finalLength << '\n';
    cout << "x* = " << xStar << '\n';
    cout << "f(x*) = " << fStar << '\n';
    cout << "Nf = " << evaluations << '\n';
    cout << "\nGuarantee check:\n";

    cout << "L0 / F_N = " << initialLength << " / " << fibonacci[N] << " = " << guaranteedLength << '\n';
    cout << "L_final = " << finalLength << '\n';

    if (finalLength <= guaranteedLength + 1e-9)
    {
        cout << "L_final <= L0 / F_N: condition is satisfied.\n";
    }
    else
    {
        cout << "L_final > L0 / F_N: condition is not satisfied.\n";
    }

    /*
     * Порівняння з методом золотого перерізу.
     * Дані золотого перерізу взято з лабораторної роботи №3.
     * Обидва методи виконано при epsilon = 0.1.
     */

    const double trueMinimum = 1.835810;

    // Результати методу золотого перерізу з ЛР №3
    const int goldenEvaluations = 9;
    const double goldenFinalLength = 0.068884;
    const double goldenXStar = 1.840170;


    double fibonacciError = abs(xStar - trueMinimum);
    double goldenError = abs(goldenXStar - trueMinimum);

    cout << "\nPorivnanna with golden section method\n";
    cout << "+----------------------+----------+--------------+--------------+--------------+\n";
    cout << "| " << left
        << setw(21) << "Method"
        << "| " << setw(9) << "Nf"
        << "| " << setw(13) << "Final length"
        << "| " << setw(13) << "x*"
        << "| " << setw(13) << "Pohibka"
        << "|\n";

    cout << "+----------------------+----------+--------------+--------------+--------------+\n";

    cout << "| " << left
        << setw(21) << "Fibonacci"
        << "| " << setw(9) << evaluations
        << "| " << setw(13) << finalLength
        << "| " << setw(13) << xStar
        << "| " << setw(13) << fibonacciError
        << "|\n";

    cout << "| " << left
        << setw(21) << "Golden section"
        << "| " << setw(9) << goldenEvaluations
        << "| " << setw(13) << goldenFinalLength
        << "| " << setw(13) << goldenXStar
        << "| " << setw(13) << goldenError
        << "|\n";

    cout << "+----------------------+----------+--------------+--------------+--------------+\n";
}
