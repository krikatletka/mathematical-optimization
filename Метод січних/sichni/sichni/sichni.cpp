#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

using namespace std;

double f(double x)
{
    return x * log(x) - 2.0 * sin(x / 2.0) - x;
}

double df(double x)
{
    return log(x) - cos(x / 2.0);
}

double ddf(double x)
{
    return 1.0 / x + 0.5 * sin(x / 2.0);
}

struct Row
{
    int k;
    double xPrevious;
    double dfPrevious;
    double xCurrent;
    double dfCurrent;
    double xNext;
    double dfNext;
    double fNext;
    string action;
};

struct Result
{
    double x = 0.0;
    double fx = 0.0;
    int iterations = 0;
    int functionCalls = 0;
    int derivativeCalls = 0;
    int secondDerivativeCalls = 0;
    bool converged = false;
    vector<Row> rows;
};

string formatNumber(double value)
{
    ostringstream output;
    output << fixed << setprecision(9) << value;
    return output.str();
}

void printBorder(const vector<int>& widths)
{
    cout << '+';
    for (int width : widths) cout << string(width + 2, '-') << '+';
    cout << '\n';
}

void printRow(const vector<string>& values,
    const vector<int>& widths)
{
    cout << '|';
    for (size_t i = 0; i < values.size(); ++i)   cout << ' ' << left << setw(widths[i]) << values[i]  << " |";
    cout << '\n';
}

Result secantMethod(double a,double b, double epsilon, bool pureSecant){
    Result result;
    double xPrevious = a;
    double xCurrent = b;
    double dfPrevious = df(xPrevious);
    double dfCurrent = df(xCurrent);
    result.derivativeCalls = 2;

    if (!(dfPrevious < 0.0 && dfCurrent > 0.0)) throw runtime_error("The initial interval does not contain a derivative sign change.");
    for (int k = 1; k <= 100; ++k)
    {
        double denominator = dfCurrent - dfPrevious;

        if (abs(denominator) < 1e-12)  throw runtime_error(   "Division by a value too close to zero.");

        double xNext = (dfCurrent * xPrevious -  dfPrevious * xCurrent) / denominator;

        if (!isfinite(xNext) || xNext <= 0.0) throw runtime_error(   "The new point is outside the domain x > 0." );

        double dfNext = df(xNext);
        double fNext = f(xNext);

        result.derivativeCalls++;
        result.functionCalls++;

        bool stop = abs(dfNext) <= epsilon;

        string action;

        if (stop)
        {
            action = "STOP: abs(df) <= epsilon";
        }
        else if (pureSecant)
        {
            action = "shift last two points";
        }
        else if (dfNext > 0.0)
        {
            action = "right = x_next";
        }
        else
        {
            action = "left = x_next";
        }

        result.rows.push_back( { k, xPrevious, dfPrevious, xCurrent,  dfCurrent, xNext, dfNext, fNext, action});

        result.x = xNext;
        result.fx = fNext;
        result.iterations = k;

        if (stop)
        {
            result.converged = true;
            break;
        }

        if (pureSecant)
        {
            xPrevious = xCurrent;
            dfPrevious = dfCurrent;

            xCurrent = xNext;
            dfCurrent = dfNext;
        }
        else if (dfNext > 0.0)
        {
            xCurrent = xNext;
            dfCurrent = dfNext;
        }
        else
        {
            xPrevious = xNext;
            dfPrevious = dfNext;
        }
    }

    return result;
}

Result newtonMethod(double x, double epsilon)
{
    Result result;

    double derivative = df(x);
    result.derivativeCalls = 1;

    while (abs(derivative) > epsilon && result.iterations < 100)
    {
        double secondDerivative = ddf(x);
        result.secondDerivativeCalls++;

        if (abs(secondDerivative) < 1e-12) throw runtime_error( "Second derivative is too close to zero." );

        x = x - derivative / secondDerivative;

        if (!isfinite(x) || x <= 0.0)  throw runtime_error(  "Newton method produced an invalid point." );

        derivative = df(x);
        result.derivativeCalls++;
        result.iterations++;
    }

    result.x = x;
    result.fx = f(x);
    result.functionCalls = 1;
    result.converged = abs(derivative) <= epsilon;

    return result;
}

double referenceMinimum(double a, double b)
{
    for (int i = 0; i < 100; ++i)
    {
        double middle = (a + b) / 2.0;

        if (df(middle) > 0.0)
        {
            b = middle;
        }
        else
        {
            a = middle;
        }
    }

    return (a + b) / 2.0;
}

void printSecantTable(const string& title, const Result& result, double reference)
{
    vector<int> widths = { 3, 13, 15, 13, 13, 13, 15, 15, 28 };

    cout << '\n' << title << '\n';

    printBorder(widths);

    printRow(
        {
            "k",
            "x_(k-1)",
            "df(x_(k-1))",
            "x_k",
            "df(x_k)",
            "x_(k+1)",
            "df(x_(k+1))",
            "f(x_(k+1))",
            "Action"
        },
        widths);

    printBorder(widths);

    for (const Row& row : result.rows)
    {
        printRow(
            {
                to_string(row.k),
                formatNumber(row.xPrevious),
                formatNumber(row.dfPrevious),
                formatNumber(row.xCurrent),
                formatNumber(row.dfCurrent),
                formatNumber(row.xNext),
                formatNumber(row.dfNext),
                formatNumber(row.fNext),
                row.action
            },
            widths);
    }

    printBorder(widths);

    cout << "Status: " << (result.converged ? "CONVERGED" : "ITERATION LIMIT") << '\n';
    cout << "x* = " << formatNumber(result.x)  << '\n';
    cout << "f(x*) = " << formatNumber(result.fx) << '\n';
    cout << "Iterations = " << result.iterations  << '\n';
    cout << "Derivative calls = " << result.derivativeCalls << '\n';
    cout << "Second derivative calls = " << result.secondDerivativeCalls << '\n';
    cout << "Function calls = " << result.functionCalls  << '\n';
    cout << "Absolute error = " << formatNumber(abs(result.x - reference))  << '\n';
}

int main()
{
    try
    {
        const double a = 0.75;
        const double b = 2.75;
        const double epsilon = 0.001;

        double reference = referenceMinimum(a, b);

        Result textbookSecant = secantMethod(a, b, epsilon, false);
        Result pureSecant = secantMethod(a, b, epsilon, true);
        Result newton = newtonMethod(a, epsilon);

        cout << fixed << setprecision(9);

        cout << "LABORATORY WORK 7 - VARIANT 8\n";
        cout << "f(x) = x*ln(x) - 2*sin(x/2) - x\n";
        cout << "df(x) = ln(x) - cos(x/2)\n";
        cout << "Initial interval: [" << a << "; " << b << "]\n";
        cout << "epsilon = " << epsilon << '\n';

        cout << "df(a) = " << df(a) << ", df(b) = " << df(b) << '\n';
        cout << "Reference minimum x = " << setprecision(12) << reference << "\n";

        cout << setprecision(9);

        printSecantTable("TEXTBOOK SECANT (SIGN REPLACEMENT)",  textbookSecant, reference);
        printSecantTable( "PURE SECANT (LAST TWO POINTS)",  pureSecant,  reference);
        cout << "\nCOMPARISON OF METHODS\n";

        vector<int> widths = { 25, 14, 12, 12, 12, 18 };
        printBorder(widths);

        printRow(
            {
                "Method",
                "x*",
                "Iterations",
                "N_df",
                "N_ddf",
                "Absolute error"
            },
            widths);

        printBorder(widths);

        auto printComparison =
            [&](const string& name,
                const Result& result)
            {
                printRow(
                    {
                        name,
                        formatNumber(result.x),
                        to_string(result.iterations),
                        to_string(result.derivativeCalls),
                        to_string(result.secondDerivativeCalls),
                        formatNumber(abs(result.x - reference))
                    },
                    widths);
            };

        printComparison("Textbook secant", textbookSecant);
        printComparison("Pure secant", pureSecant);
        printComparison(  "Newton, x0 = 0.75",  newton);
        printBorder(widths);

        cout << "\nCHECK OF INITIAL INTERVAL\n";

        if (df(a) * df(b) < 0.0)
        {
            cout << "df(a) * df(b) < 0: condition is satisfied.\n";
        }
        else
        {
            cout << "df(a) * df(b) >= 0: condition is not satisfied.\n";
        }

        cout << "\nFINAL RESULTS\n";
        cout << "Textbook secant x* = "  << formatNumber(textbookSecant.x) << '\n';

        cout << "Pure secant x* = " << formatNumber(pureSecant.x)  << '\n';

        cout << "Newton x* = " << formatNumber(newton.x)  << '\n';

        return 0;
    }
    catch (const exception& error)
    {
        cerr << "Error: "  << error.what()  << '\n';

        return 1;
    }
}