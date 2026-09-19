#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>

using namespace std;

double f(double x){ return x * log(x) - 2.0 * sin(x / 2.0) - x;}
void printLine(){ cout << "+----+----------+----------+----------+----------+-------------+-------------+----------+-----------------------------+\n";}
int main()
{
    const double initialA = 0.75;
    const double initialB = 2.75;
    const double epsilon = 0.1;
    const double delta = 0.01;

    const double tau = (sqrt(5.0) - 1.0) / 2.0;

    double a = initialA;
    double b = initialB;

    double x1 = a + (1.0 - tau) * (b - a);
    double x2 = a + tau * (b - a);

    double f1 = f(x1);
    double f2 = f(x2);

    int k = 1;
    int evaluations = 2;

    cout << fixed << setprecision(5);

    cout << "Golden section method\n\n";
    cout << "Initial interval: [" << a << "; " << b << "]\n";
    cout << "epsilon = " << epsilon << '\n';
    cout << "tau = " << tau << "\n\n";

    printLine();

    cout << "| " << left
        << setw(3) << "k"
        << "| " << setw(9) << "a_k"
        << "| " << setw(9) << "b_k"
        << "| " << setw(9) << "x1"
        << "| " << setw(9) << "x2"
        << "| " << setw(12) << "f(x1)"
        << "| " << setw(12) << "f(x2)"
        << "| " << setw(9) << "L_k"
        << "| " << setw(28) << "Decision"
        << "|\n";

    printLine();

    while (true)
    {
        double length = b - a;
        string decision;

        if (length <= epsilon)
        {
            decision = "L_k <= epsilon -> STOP";

            cout << "| " << left
                << setw(3) << k
                << "| " << setw(9) << a
                << "| " << setw(9) << b
                << "| " << setw(9) << x1
                << "| " << setw(9) << x2
                << "| " << setw(12) << f1
                << "| " << setw(12) << f2
                << "| " << setw(9) << length
                << "| " << setw(28) << decision
                << "|\n";

            break;
        }

        if (f1 <= f2)
        {
            decision = "f1 <= f2: b = x2";

            cout << "| " << left
                << setw(3) << k
                << "| " << setw(9) << a
                << "| " << setw(9) << b
                << "| " << setw(9) << x1
                << "| " << setw(9) << x2
                << "| " << setw(12) << f1
                << "| " << setw(12) << f2
                << "| " << setw(9) << length
                << "| " << setw(28) << decision
                << "|\n";

            b = x2;

            // успадковуємо стару точку x1
            x2 = x1;
            f2 = f1;

            x1 = a + (1.0 - tau) * (b - a);
            f1 = f(x1);

            evaluations++;
        }
        else
        {
            decision = "f1 > f2: a = x1";

            cout << "| " << left
                << setw(3) << k
                << "| " << setw(9) << a
                << "| " << setw(9) << b
                << "| " << setw(9) << x1
                << "| " << setw(9) << x2
                << "| " << setw(12) << f1
                << "| " << setw(12) << f2
                << "| " << setw(9) << length
                << "| " << setw(28) << decision
                << "|\n";

            a = x1;
            
			// тут успадковуємо x2
            x1 = x2;
            f1 = f2;

            x2 = a + tau * (b - a);
            f2 = f(x2);

            evaluations++;
        }

        k++;
    }

    printLine();

    double xStar = (a + b) / 2.0;
    double fStar = f(xStar);

    cout << "\nFinal interval: [" << a << "; " << b << "]\n";
    cout << "Final interval length: " << b - a << '\n';
    cout << "Minimum point x* = " << xStar << '\n';
    cout << "Function value f(x*) = " << fStar << '\n';
    cout << "Number of iterations = " << k << '\n';
    cout << "Number of function evaluations Nf = " << evaluations << '\n';
    
    /*
     * Порівняння методів при однаковій точності epsilon = 0.1.
     * Результати дихотомії та половинного ділення взято
     * з попередньої лабораторної роботи.
     */

    double initialLength = initialB - initialA;

    // Результати попередньої лабораторної роботи
    int dichotomyIterations = 5;
    int dichotomyNf = 10;
    double dichotomyLength = 0.0721875;

    int halfDivisionIterations = 5;
    int halfDivisionNf = 11;
    double halfDivisionLength = 0.0625;

    // Результати методу золотого перерізу
    int goldenIterations = k;
    int goldenNf = evaluations;
    double goldenLength = b - a;


    double dichotomyEta = pow(dichotomyLength / initialLength, 1.0 / dichotomyNf);
    double halfDivisionEta = pow(halfDivisionLength / initialLength, 1.0 / halfDivisionNf);
    double goldenEta = pow(goldenLength / initialLength, 1.0 / goldenNf);

    cout << "\nComparison of optimization methods\n";
    cout << "Accuracy epsilon = " << epsilon << "\n\n";

    cout << "+----------------------+------------+----------+----------+----------+\n";
    cout << "| " << left
        << setw(21) << "Method"
        << "| " << setw(11) << "Iterations"
        << "| " << setw(9) << "Nf"
        << "| " << setw(9) << "L_n"
        << "| " << setw(9) << "eta"
        << "|\n";

    cout << "+----------------------+------------+----------+----------+----------+\n";

    cout << "| " << left
        << setw(21) << "Dichotomy"
        << "| " << setw(11) << dichotomyIterations
        << "| " << setw(9) << dichotomyNf
        << "| " << setw(9) << dichotomyLength
        << "| " << setw(9) << dichotomyEta
        << "|\n";

    cout << "| " << left
        << setw(21) << "Half division"
        << "| " << setw(11) << halfDivisionIterations
        << "| " << setw(9) << halfDivisionNf
        << "| " << setw(9) << halfDivisionLength
        << "| " << setw(9) << halfDivisionEta
        << "|\n";

    cout << "| " << left
        << setw(21) << "Golden section"
        << "| " << setw(11) << goldenIterations
        << "| " << setw(9) << goldenNf
        << "| " << setw(9) << goldenLength
        << "| " << setw(9) << goldenEta
        << "|\n";

    cout << "+----------------------+------------+----------+----------+----------+\n";
}