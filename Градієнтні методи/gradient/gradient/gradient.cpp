#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <algorithm>

using namespace std;

const double EPSILON = 0.01;
const double DERIVATIVE_STEP = 1e-5;
const double MIN_STEP = 1e-12;

struct Point
{
    double x1;
    double x2;
};

struct Gradient
{
    double g1;
    double g2;
};

struct Iteration
{
    int k;
    Point x;
    double value;
    Gradient gradient;
    double gradientNorm;
    Point direction;
    double step;
    int reductions;
    double trialValue;
    string decision;
};

struct Result
{
    string metric;
    double initialStep;

    Point solution;
    double finalValue;
    double finalGradientNorm;

    int iterations;
    int functionCalls;
    int gradientCalls;

    bool converged;
    vector<Iteration> table;
};

double F(const Point& p)
{
    return p.x1 * p.x1 + 3.0 * p.x2 * p.x2  + 2.0 * p.x1 * p.x2 - 9.0 * p.x1  - 17.0 * p.x2;
}

Gradient analyticGradient(const Point& p)
{
    Gradient result;
    result.g1 = 2.0 * p.x1 + 2.0 * p.x2 - 9.0;
    result.g2 = 6.0 * p.x2 + 2.0 * p.x1 - 17.0;

    return result;
}

double norm2(const Gradient& g)
{
    return sqrt(g.g1 * g.g1 + g.g2 * g.g2);
}

int signValue(double value)
{
    if (value > 0.0)
    {
        return 1;
    }

    if (value < 0.0)
    {
        return -1;
    }

    return 0;
}

Point add(const Point& a, const Point& b)
{
    return { a.x1 + b.x1, a.x2 + b.x2 };
}

Point multiply(const Point& p, double number)
{
    return { p.x1 * number, p.x2 * number };
}

Point directionForMetric(const Gradient& g,
    const string& metric)
{
    if (metric == "Spherical")
    {
        double length = norm2(g);

        return
        {
            -g.g1 / length,
            -g.g2 / length
        };
    }

    if (metric == "Cubic")
    {
        return
        {
            -static_cast<double>(signValue(g.g1)),
            -static_cast<double>(signValue(g.g2))
        };
    }

    if (metric == "Octahedral")
    {
        if (abs(g.g1) >= abs(g.g2))
        {
            return
            {
                -static_cast<double>(signValue(g.g1)),
                0.0
            };
        }

        return
        {
            0.0,
            -static_cast<double>(signValue(g.g2))
        };
    }

    throw invalid_argument("Unknown metric");
}

string number(double value, int precision = 6)
{
    ostringstream output;

    output << fixed
        << setprecision(precision)
        << value;

    return output.str();
}

void printBorder(const vector<int>& widths)
{
    cout << '+';

    for (int width : widths)
    {
        cout << string(width + 2, '-') << '+';
    }

    cout << '\n';
}

void printRow(const vector<string>& values,
    const vector<int>& widths)
{
    cout << '|';

    for (size_t i = 0; i < values.size(); ++i)
    {
        cout << ' '
            << left
            << setw(widths[i])
            << values[i]
            << " |";
    }

    cout << '\n';
}

void printDerivativeCheck(const Point& point)
{
    double h = DERIVATIVE_STEP;

    Point xPlusX1 = { point.x1 + h, point.x2 };
    Point xMinusX1 = { point.x1 - h, point.x2 };

    Point xPlusX2 = { point.x1, point.x2 + h };
    Point xMinusX2 = { point.x1, point.x2 - h };

    double numericalG1 =
        (F(xPlusX1) - F(xMinusX1)) / (2.0 * h);

    double numericalG2 =
        (F(xPlusX2) - F(xMinusX2)) / (2.0 * h);

    Gradient analytical = analyticGradient(point);

    cout << "\nDERIVATIVE CHECK\n";

    cout << "Point = ("
        << number(point.x1)
        << "; "
        << number(point.x2)
        << ")\n";

    cout << "Analytical dF/dx1 = "
        << number(analytical.g1)
        << '\n';

    cout << "Numerical dF/dx1  = "
        << number(numericalG1)
        << '\n';

    cout << "Difference         = "
        << number(abs(analytical.g1 - numericalG1))
        << "\n\n";

    cout << "Analytical dF/dx2 = "
        << number(analytical.g2)
        << '\n';

    cout << "Numerical dF/dx2  = "
        << number(numericalG2)
        << '\n';

    cout << "Difference         = "
        << number(abs(analytical.g2 - numericalG2))
        << '\n';
}

void printAnalyticalMinimum()
{
    Point minimum = { 2.5, 2.0 };

    cout << "\nANALYTICAL MINIMUM\n";
    cout << "x1* = " << number(minimum.x1) << '\n';
    cout << "x2* = " << number(minimum.x2) << '\n';
    cout << "F(x*) = " << number(F(minimum)) << '\n';
}

Result gradientMethod(const string& metric,
    const Point& initialPoint,
    double initialStep,
    double epsilon)
{
    Result result;

    result.metric = metric;
    result.initialStep = initialStep;
    result.iterations = 0;
    result.functionCalls = 1;
    result.gradientCalls = 0;
    result.converged = false;

    Point x = initialPoint;
    double currentValue = F(x);
    double step = initialStep;

    for (int k = 1; k <= 1000; ++k)
    {
        Gradient gradient = analyticGradient(x);
        result.gradientCalls++;

        double gradientLength = norm2(gradient);

        if (gradientLength <= epsilon)
        {
            result.solution = x;
            result.finalValue = currentValue;
            result.finalGradientNorm = gradientLength;
            result.converged = true;
            break;
        }

        Point direction =
            directionForMetric(gradient, metric);

        Point trialPoint;
        double trialValue;
        int reductions = 0;

        while (true)
        {
            trialPoint = add(x, multiply(direction, step));
            trialValue = F(trialPoint);
            result.functionCalls++;

            if (trialValue < currentValue)
            {
                break;
            }

            step /= 2.0;
            reductions++;

            if (step < MIN_STEP)
            {
                result.solution = x;
                result.finalValue = currentValue;
                result.finalGradientNorm = gradientLength;
                result.converged = false;
                return result;
            }
        }

        result.table.push_back( {k,  x, currentValue,  gradient,  gradientLength, direction, step, reductions, trialValue, "accepted"});

        x = trialPoint;
        currentValue = trialValue;
        result.iterations = k;
    }

    if (!result.converged)
    {
        Gradient finalGradient = analyticGradient(x);

        result.solution = x;
        result.finalValue = currentValue;
        result.finalGradientNorm = norm2(finalGradient);
    }

    return result;
}

void printIterationTable(const Result& result)
{
    vector<int> widths =
    {
        3, 11, 11, 13, 9, 11, 11,
        9, 9, 14, 14, 16
    };

    cout << "\n"
        << result.metric
        << " METRIC, h0 = "
        << number(result.initialStep)
        << '\n';

    printBorder(widths);

    printRow(
        {
            "k",
            "x1",
            "x2",
            "F(x)",
            "g1",
            "g2",
            "normG",
            "d1",
            "d2",
            "h",
            "reductions",
            "F(trial)"
        },
        widths);

    printBorder(widths);

    for (const Iteration& row : result.table)
    {
        printRow(
            {
                to_string(row.k),
                number(row.x.x1),
                number(row.x.x2),
                number(row.value),
                number(row.gradient.g1),
                number(row.gradient.g2),
                number(row.gradientNorm),
                number(row.direction.x1),
                number(row.direction.x2),
                number(row.step),
                to_string(row.reductions),
                number(row.trialValue)
            },
            widths);
    }

    printBorder(widths);

    cout << "Status = "
        << (result.converged ? "CONVERGED" : "NOT CONVERGED")
        << '\n';

    cout << "x1* = "
        << number(result.solution.x1)
        << '\n';

    cout << "x2* = "
        << number(result.solution.x2)
        << '\n';

    cout << "F(x*) = "
        << number(result.finalValue)
        << '\n';

    cout << "Final gradient norm = "
        << number(result.finalGradientNorm)
        << '\n';

    cout << "Iterations n = "
        << result.iterations
        << '\n';

    cout << "Function calls NF = "
        << result.functionCalls
        << '\n';

    cout << "Gradient calls NGrad = "
        << result.gradientCalls
        << '\n';

    cout << "Total calls = "
        << result.functionCalls + result.gradientCalls
        << '\n';
}

void printComparison(const vector<Result>& results)
{
    vector<int> widths =
    {
        16, 9, 9, 12, 12, 12, 12, 14
    };

    cout << "\nCOMPARISON TABLE\n";

    printBorder(widths);

    printRow(
        {
            "Metric",
            "h0",
            "n",
            "NF",
            "NGrad",
            "Total",
            "F(x*)",
            "Error to (2.5;2)"
        },
        widths);

    printBorder(widths);

    for (const Result& result : results)
    {
        double error =
            sqrt(
                (result.solution.x1 - 2.5)
                * (result.solution.x1 - 2.5)
                +
                (result.solution.x2 - 2.0)
                * (result.solution.x2 - 2.0)
            );

        printRow(
            {
                result.metric,
                number(result.initialStep),
                to_string(result.iterations),
                to_string(result.functionCalls),
                to_string(result.gradientCalls),
                to_string(result.functionCalls
                          + result.gradientCalls),
                number(result.finalValue),
                number(error)
            },
            widths);
    }

    printBorder(widths);
}

int main()
{
    try
    {
        const Point initialPoint = { -2.0, 6.0 };

        cout << fixed << setprecision(6);

        cout << "LABORATORY WORK 8 - VARIANT 8\n\n";

        cout << "Function:\n";
        cout << "F(x1, x2) = x1^2 + 3*x2^2 "
            "+ 2*x1*x2 - 9*x1 - 17*x2\n\n";

        cout << "Initial point:\n";
        cout << "x0 = ("
            << number(initialPoint.x1)
            << "; "
            << number(initialPoint.x2)
            << ")\n";

        cout << "epsilon = "
            << number(EPSILON)
            << '\n';

        cout << "Initial steps: h0 = 1.0 and h0 = 0.7\n";

        printDerivativeCheck(initialPoint);
        printAnalyticalMinimum();

        vector<Result> results;

        vector<string> metrics =
        {
            "Spherical",
            "Cubic",
            "Octahedral"
        };

        vector<double> initialSteps =
        {
            1.0,
            0.7
        };

        for (const string& metric : metrics)
        {
            for (double step : initialSteps)
            {
                Result result = gradientMethod( metric, initialPoint, step, EPSILON);

                results.push_back(result);
                printIterationTable(result);
            }
        }

        printComparison(results);

        cout << "\nFINAL CHECK\n";

        for (const Result& result : results)
        {
            cout << result.metric
                << ", h0 = "
                << number(result.initialStep)
                << ": ";

            if (result.finalGradientNorm <= EPSILON)
            {
                cout << "gradient criterion satisfied\n";
            }
            else
            {
                cout << "gradient criterion NOT satisfied\n";
            }
        }

        return 0;
    }
    catch (const exception& error)
    {
        cerr << "\nError: "
            << error.what()
            << '\n';

        return 1;
    }
}