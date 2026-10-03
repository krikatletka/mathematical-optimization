#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

using namespace std;

double f(double x)
{
    return x * log(x) - 2.0 * sin(x / 2.0) - x;
}

struct MethodResult
{
    string name;
    double xStar;
    int evaluations;
    string accuracy;
};

void printIterationLine()
{
    cout << "+---+----------+----------+----------+-------------+-------------+-------------+"
        "-------------+-------------+----------+-------------+-------------+-------------+------------------------+\n";
}

int indexOfMinimum(const array<double, 3>& values)
{
    return static_cast<int>(min_element(values.begin(), values.end()) - values.begin());
}

int indexOfMaximum(const array<double, 3>& values)
{
    return static_cast<int>(max_element(values.begin(), values.end()) - values.begin());
}

int main()
{
    const double a0 = 0.75;
    const double b0 = 2.75;
    const double step = 0.50;
    const double sigmaF = 0.01;
    const double sigmaX = 0.001;
    const double tiny = 1e-12;
    const int maxIterations = 100;

    array<double, 3> x{};
    array<double, 3> fx{};

    x[0] = a0;
    x[1] = x[0] + step;
    fx[0] = f(x[0]);
    fx[1] = f(x[1]);

    if (fx[0] > fx[1])
        x[2] = x[0] + 2.0 * step;
    else
        x[2] = x[0] - step;

    if (x[2] <= 0.0)
    {
        cerr << "Error: the third point is outside the domain x > 0.\n";
        return 1;
    }

    fx[2] = f(x[2]);

    int evaluations = 3; 
    int iterations = 0;
    bool converged = false;
    double resultX = numeric_limits<double>::quiet_NaN();
    double resultF = numeric_limits<double>::quiet_NaN();

    cout << fixed << setprecision(6);
    cout << "Powell method\n\n";
    cout << "f(x) = x*ln(x) - 2*sin(x/2) - x\n";
    cout << "Sven interval [a0; b0] = [" << a0 << "; " << b0 << "]\n";
    cout << "x1 = " << x[0] << ", Delta x = " << step << '\n';
    cout << "sigma_f = " << sigmaF << ", sigma_x = " << sigmaX << "\n\n";

    printIterationLine();
    cout << "| " << left
        << setw(2) << "k"
        << "| " << setw(9) << "x1"
        << "| " << setw(9) << "x2"
        << "| " << setw(9) << "x3"
        << "| " << setw(12) << "f(x1)"
        << "| " << setw(12) << "f(x2)"
        << "| " << setw(12) << "f(x3)"
        << "| " << setw(12) << "a1"
        << "| " << setw(12) << "a2"
        << "| " << setw(9) << "x_hat"
        << "| " << setw(12) << "f(x_hat)"
        << "| " << setw(12) << "|df|"
        << "| " << setw(12) << "|dx|"
        << "| " << setw(23) << "Action"
        << "|\n";
    printIterationLine();

    for (int k = 1; k <= maxIterations; ++k)
    {
        iterations = k;

        if (abs(x[1] - x[0]) <= tiny ||
            abs(x[2] - x[0]) <= tiny ||
            abs(x[2] - x[1]) <= tiny)
        {
            cerr << "Error: two interpolation points coincided.\n";
            return 1;
        }

        const int minIndex = indexOfMinimum(fx);
        const int maxIndex = indexOfMaximum(fx);
        const double xMin = x[minIndex];
        const double fMin = fx[minIndex];

        const double a1 = (fx[1] - fx[0]) / (x[1] - x[0]);
        const double a2 = ((fx[2] - fx[0]) / (x[2] - x[0]) - a1) / (x[2] - x[1]);

        double xHat;
        if (a2 > tiny)
        {
            xHat = -a1 / (2.0 * a2) + (x[0] + x[1]) / 2.0;
        }
        else
        {
            xHat = 2.0 * x[minIndex] - x[maxIndex];
        }

        if (xHat <= 0.0 || !isfinite(xHat))
        {
            cerr << "Error: x_hat is outside the domain x > 0.\n";
            return 1;
        }

        const double fHat = f(xHat);
        ++evaluations;

        const double deltaF = abs(fMin - fHat);
        const double deltaX = abs(xMin - xHat);
        const bool stop = deltaF <= sigmaF && deltaX <= sigmaX;
        const string action = stop ? "STOP: both conditions"
            : "replace worst point";

        cout << "| " << left
            << setw(2) << k
            << "| " << setw(9) << x[0]
            << "| " << setw(9) << x[1]
            << "| " << setw(9) << x[2]
            << "| " << setw(12) << fx[0]
            << "| " << setw(12) << fx[1]
            << "| " << setw(12) << fx[2]
            << "| " << setw(12) << a1
            << "| " << setw(12) << a2
            << "| " << setw(9) << xHat
            << "| " << setw(12) << fHat
            << "| " << setw(12) << deltaF
            << "| " << setw(12) << deltaX
            << "| " << setw(23) << action
            << "|\n";

        if (stop)
        {
            converged = true;
            resultX = xHat;
            resultF = fHat;
            break;
        }
        x[maxIndex] = xHat;
        fx[maxIndex] = fHat;
    }

    printIterationLine();

    if (!converged)
    {
        cerr << "The method did not converge in " << maxIterations << " iterations.\n";
        return 1;
    }

    const double trueX = 1.835809799;

    cout << "\nBoth stopping conditions are satisfied:\n";
    cout << "|f_min - f(x_hat)| <= sigma_f and |x_min - x_hat| <= sigma_x\n";
    cout << "x* = " << resultX << '\n';
    cout << "f(x*) = " << resultF << '\n';
    cout << "Number of iterations n = " << iterations << '\n';
    cout << "Number of function evaluations Nf = " << evaluations
        << " (= 3 initial + " << iterations << " new)\n";
    cout << "Absolute error |x* - x_true| = " << abs(resultX - trueX) << "\n\n";


    const array<MethodResult, 5> comparison = { {
        {"Dichotomy",     1.843281, 10, "epsilon = 0.1"},
        {"Half division", 1.843750, 11, "epsilon = 0.1"},
        {"Golden section",1.840170,  9, "epsilon = 0.1"},
        {"Fibonacci",     1.845238,  7, "epsilon = 0.1"},
        {"Powell",        resultX, evaluations, "sf=0.01, sx=0.001"}
    } };

    cout << "Comparison with previous methods\n";
    cout << "+----------------+----------------+------+--------------------+----------------+\n";
    cout << "| " << left << setw(15) << "Method"
        << "| " << setw(15) << "x*"
        << "| " << setw(5) << "Nf"
        << "| " << setw(19) << "Accuracy"
        << "| " << setw(15) << "|x*-x_true|" << "|\n";
    cout << "+----------------+----------------+------+--------------------+----------------+\n";

    for (const MethodResult& method : comparison)
    {
        cout << "| " << left << setw(15) << method.name
            << "| " << setw(15) << method.xStar
            << "| " << setw(5) << method.evaluations
            << "| " << setw(19) << method.accuracy
            << "| " << setw(15) << abs(method.xStar - trueX)
            << "|\n";
    }
    cout << "+----------------+----------------+------+--------------------+----------------+\n";
}
