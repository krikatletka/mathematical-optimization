#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace std;

// f(x) = x ln(x) - 2 sin(x/2) - x, x > 0
double f(double x)
{
    return x * log(x) - 2.0 * sin(x / 2.0) - x;
}

double df(double x)
{
    return log(x) - cos(x / 2.0);
}

double d2f(double x)
{
    return 1.0 / x + 0.5 * sin(x / 2.0);
}

struct Calls
{
    int f = 0;
    int df = 0;
    int d2f = 0;
};

struct NewtonRow
{
    int k;
    double x;
    double fx;
    double d1;
    double absD1;
    double d2;
    bool hasD2;
};

struct RegulatedRow
{
    int k;
    double x;
    double fx;
    double d1;
    double d2;
    double t;
    double xTrial;
    double fTrial;
    double d1Trial;
    double actualDecrease;
    double requiredDecrease;
    string decision;
};

struct Result
{
    bool converged = false;
    string message;
    double xStar = numeric_limits<double>::quiet_NaN();
    double fStar = numeric_limits<double>::quiet_NaN();
    int iterations = 0;
    Calls calls;
    vector<NewtonRow> newtonRows;
    vector<RegulatedRow> regulatedRows;
};

Result classicalNewton(double x0, double epsilon, int maxIterations)
{
    Result result;

    if (x0 <= 0.0)
    {
        result.message = "x0 is outside the domain x > 0";
        return result;
    }

    double x = x0;

    for (int k = 0; k <= maxIterations; ++k)
    {
        const double fx = f(x);
        const double d1 = df(x);
        ++result.calls.f;  
        ++result.calls.df;

        if (abs(d1) <= epsilon)
        {
            result.newtonRows.push_back({ k, x, fx, d1, abs(d1), 0.0, false });
            result.converged = true;
            result.message = "STOP: |f'(x)| <= epsilon";
            result.xStar = x;
            result.fStar = fx;
            result.iterations = k;
            return result;
        }

        if (k == maxIterations)
        {
            result.message = "maximum number of iterations reached";
            result.xStar = x;
            result.fStar = fx;
            result.iterations = k;
            return result;
        }

        const double d2 = d2f(x);
        ++result.calls.d2f;
        result.newtonRows.push_back({ k, x, fx, d1, abs(d1), d2, true });

        if (d2 <= 1e-12)
        {
            result.message =
                "f''(x) <= 0: Newton direction is not a minimum direction";

            result.xStar = x;
            result.fStar = fx;
            result.iterations = k;

            return result;
        }

        const double xNext = x - d1 / d2;
        if (xNext <= 0.0 || !isfinite(xNext))
        {
            result.message = "Newton step left the domain x > 0";
            result.xStar = x;
            result.fStar = fx;
            result.iterations = k;
            return result;
        }

        x = xNext;
    }

    return result;
}

Result regulatedNewton(double x0, double epsilon, double alpha,
    double tMin, int maxIterations)
{
    Result result;

    if (x0 <= 0.0 || alpha <= 0.0 || alpha >= 1.0)
    {
        result.message = "invalid x0 or alpha";
        return result;
    }

    double x = x0;
    double fx = f(x);
    double d1 = df(x);
    ++result.calls.f;
    ++result.calls.df;

    for (int k = 0; k < maxIterations; ++k)
    {
        if (abs(d1) <= epsilon)
        {
            result.converged = true;
            result.message = "STOP: |f'(x)| <= epsilon";
            result.xStar = x;
            result.fStar = fx;
            result.iterations = k;
            return result;
        }

        const double d2 = d2f(x);
        ++result.calls.d2f;

        // Для задачі мінімізації потрібна додатна кривина.
        if (d2 <= 1e-12)
        {
            result.message = "f''(x) <= 0: Newton direction is not a minimum direction";
            result.xStar = x;
            result.fStar = fx;
            result.iterations = k;
            return result;
        }

        double t = 1.0;

        while (t >= tMin)
        {
            const double xTrial = x - t * d1 / d2;

            if (xTrial <= 0.0 || !isfinite(xTrial))
            {
                t /= 2.0;
                continue;
            }

            const double fTrial = f(xTrial);
            const double d1Trial = df(xTrial);
            ++result.calls.f;
            ++result.calls.df;

            const double actual = fTrial - fx;
            const double required = -alpha * t * d1 * d1 / d2;

            if (abs(d1Trial) <= epsilon)
            {
                result.regulatedRows.push_back({
                    k + 1, x, fx, d1, d2, t, xTrial, fTrial, d1Trial,
                    actual, required, "STOP: |f'(x_trial)| <= epsilon"
                    });
                result.converged = true;
                result.message = "STOP: |f'(x)| <= epsilon";
                result.xStar = xTrial;
                result.fStar = fTrial;
                result.iterations = k + 1;
                return result;
            }

            if (actual <= required)
            {
                result.regulatedRows.push_back({
                    k + 1, x, fx, d1, d2, t, xTrial, fTrial, d1Trial,
                    actual, required, "accepted"
                    });
                x = xTrial;
                fx = fTrial;
                d1 = d1Trial;
                break;
            }

            result.regulatedRows.push_back({
                k + 1, x, fx, d1, d2, t, xTrial, fTrial, d1Trial,
                actual, required, "rejected: t = t / 2"
                });
            t /= 2.0;
        }

        if (t < tMin)
        {
            result.message = "step control failed: t < t_min";
            result.xStar = x;
            result.fStar = fx;
            result.iterations = k;
            return result;
        }
    }

    result.message = "maximum number of iterations reached";
    result.xStar = x;
    result.fStar = fx;
    result.iterations = maxIterations;
    return result;
}

void printNewtonTable(const Result& result)
{
    cout << "+----+-------------+-------------+-------------+-------------+-------------+\n";
    cout << "| " << left << setw(3) << "k"
        << "| " << setw(12) << "x_k"
        << "| " << setw(12) << "f(x_k)"
        << "| " << setw(12) << "f'(x_k)"
        << "| " << setw(12) << "|f'(x_k)|"
        << "| " << setw(12) << "f''(x_k)" << "|\n";
    cout << "+----+-------------+-------------+-------------+-------------+-------------+\n";

    for (const NewtonRow& row : result.newtonRows)
    {
        cout << "| " << left << setw(3) << row.k
            << "| " << setw(12) << row.x
            << "| " << setw(12) << row.fx
            << "| " << setw(12) << row.d1
            << "| " << setw(12) << row.absD1;

        if (row.hasD2)
            cout << "| " << setw(12) << row.d2 << "|\n";
        else
            cout << "| " << setw(12) << "-" << "|\n";
    }

    cout << "+----+-------------+-------------+-------------+-------------+-------------+\n";
}

void printRegulatedTable(const Result& result)
{
    cout << "+----+----------+-----------+-----------+-----------+--------+----------+-----------+-----------+-----------+-----------+-------------------------------+\n";
    cout << "| " << left << setw(3) << "k"
        << "| " << setw(9) << "x_k"
        << "| " << setw(10) << "f(x_k)"
        << "| " << setw(10) << "f'(x_k)"
        << "| " << setw(10) << "f''(x_k)"
        << "| " << setw(7) << "t"
        << "| " << setw(9) << "x_trial"
        << "| " << setw(10) << "f_trial"
        << "| " << setw(10) << "f'_trial"
        << "| " << setw(10) << "delta_f"
        << "| " << setw(10) << "bound"
        << "| " << setw(30) << "Decision" << "|\n";
    cout << "+----+----------+-----------+-----------+-----------+--------+----------+-----------+-----------+-----------+-----------+-------------------------------+\n";

    for (const RegulatedRow& row : result.regulatedRows)
    {
        cout << "| " << left << setw(3) << row.k
            << "| " << setw(9) << row.x
            << "| " << setw(10) << row.fx
            << "| " << setw(10) << row.d1
            << "| " << setw(10) << row.d2
            << "| " << setw(7) << row.t
            << "| " << setw(9) << row.xTrial
            << "| " << setw(10) << row.fTrial
            << "| " << setw(10) << row.d1Trial
            << "| " << setw(10) << row.actualDecrease
            << "| " << setw(10) << row.requiredDecrease
            << "| " << setw(30) << row.decision << "|\n";
    }

    cout << "+----+----------+-----------+-----------+-----------+--------+----------+-----------+-----------+-----------+-----------+-------------------------------+\n";
}

void printResult(const Result& result)
{
    cout << result.message << '\n';
    cout << "x* = " << result.xStar << '\n';
    cout << "f(x*) = " << result.fStar << '\n';
    cout << "iterations = " << result.iterations << '\n';
    cout << "calls: f = " << result.calls.f
        << ", f' = " << result.calls.df
        << ", f'' = " << result.calls.d2f << "\n\n";
}

void printDerivativeCheck(double x, double h)
{
    const double numericD1 = (f(x + h) - f(x - h)) / (2.0 * h);
    const double numericD2 = (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h);

    cout << "DERIVATIVE CHECK AT x = " << x << "\n";
    cout << "f'(x)  analytic = " << df(x)
        << ", numeric = " << numericD1
        << ", difference = " << abs(df(x) - numericD1) << '\n';
    cout << "f''(x) analytic = " << d2f(x)
        << ", numeric = " << numericD2
        << ", difference = " << abs(d2f(x) - numericD2) << "\n\n";
}

int main()
{
    const double x0 = 0.75;
    const double epsilon = 0.001;
    const double alpha = 0.5;
    const double tMin = 1e-8;
    const int maxIterations = 50;
    const double trueX = 1.835809799;

    cout << fixed << setprecision(6);
    cout << "LABORATORY WORK 6\n";
    cout << "f(x)   = x*ln(x) - 2*sin(x/2) - x\n";
    cout << "f'(x)  = ln(x) - cos(x/2)\n";
    cout << "f''(x) = 1/x + 0.5*sin(x/2)\n";
    cout << "x0 = " << x0 << ", epsilon = " << epsilon
        << ", alpha = " << alpha << "\n\n";

    printDerivativeCheck(x0, 1e-5);

    cout << "CLASSICAL NEWTON METHOD\n";
    Result newton = classicalNewton(x0, epsilon, maxIterations);
    printNewtonTable(newton);
    printResult(newton);

    cout << "NEWTON-RAPHSON WITH STEP CONTROL\n";
    Result regulated = regulatedNewton(x0, epsilon, alpha, tMin, maxIterations);
    printRegulatedTable(regulated);
    printResult(regulated);

    cout << "STARTING-POINT EXPERIMENT\n";
    cout << "+----------+----------------+-----------+-------------+------------+-------------------------------+\n";
    cout << "| " << left << setw(9) << "x0"
        << "| " << setw(15) << "Method"
        << "| " << setw(10) << "x*"
        << "| " << setw(12) << "iterations"
        << "| " << setw(11) << "|x*-true|"
        << "| " << setw(30) << "Result" << "|\n";
    cout << "+----------+----------------+-----------+-------------+------------+-------------------------------+\n";

    const vector<double> starts = { 0.75, 2.75, 4.00 };
    for (double start : starts)
    {
        const Result ordinary = classicalNewton(start, epsilon, maxIterations);
        const Result safe = regulatedNewton(start, epsilon, alpha, tMin, maxIterations);

        for (const auto& item : vector<pair<string, Result>>{
                 {"Newton", ordinary}, {"Newton-Raphson", safe} })
        {
            cout << "| " << left << setw(9) << start
                << "| " << setw(15) << item.first
                << "| " << setw(10) << item.second.xStar
                << "| " << setw(12) << item.second.iterations
                << "| " << setw(11) << abs(item.second.xStar - trueX)
                << "| " << setw(30) << item.second.message << "|\n";
        }
    }
    cout << "+----------+----------------+-----------+-------------+------------+-------------------------------+\n\n";

    cout << "COMPARISON WITH PREVIOUS METHODS\n";
    cout << "+----------------------+------------+-------------------+-------------+\n";
    cout << "| " << left << setw(21) << "Method"
        << "| " << setw(11) << "x*"
        << "| " << setw(18) << "Calls"
        << "| " << setw(12) << "|x*-true|" << "|\n";
    cout << "+----------------------+------------+-------------------+-------------+\n";
    cout << "| " << setw(21) << "Dichotomy"
        << "| " << setw(11) << 1.843281
        << "| " << setw(18) << "10 f"
        << "| " << setw(12) << abs(1.843281 - trueX) << "|\n";
    cout << "| " << setw(21) << "Half division"
        << "| " << setw(11) << 1.843750
        << "| " << setw(18) << "11 f"
        << "| " << setw(12) << abs(1.843750 - trueX) << "|\n";
    cout << "| " << setw(21) << "Golden section"
        << "| " << setw(11) << 1.840170
        << "| " << setw(18) << "9 f"
        << "| " << setw(12) << abs(1.840170 - trueX) << "|\n";
    cout << "| " << setw(21) << "Fibonacci"
        << "| " << setw(11) << 1.845238
        << "| " << setw(18) << "7 f"
        << "| " << setw(12) << abs(1.845238 - trueX) << "|\n";
    cout << "| " << setw(21) << "Powell"
        << "| " << setw(11) << 1.835807
        << "| " << setw(18) << "7 f"
        << "| " << setw(12) << abs(1.835807 - trueX) << "|\n";
    cout << "| " << setw(21) << "Newton"
        << "| " << setw(11) << newton.xStar
        << "| " << setw(18) << (to_string(newton.calls.df) + " f', " +
            to_string(newton.calls.d2f) + " f''")
        << "| " << setw(12) << abs(newton.xStar - trueX) << "|\n";
    cout << "| " << setw(21) << "Newton-Raphson"
        << "| " << setw(11) << regulated.xStar
        << "| " << setw(18) << (to_string(regulated.calls.f) + " f, " +
            to_string(regulated.calls.df) + " f', " +
            to_string(regulated.calls.d2f) + " f''")
        << "| " << setw(12) << abs(regulated.xStar - trueX) << "|\n";
    cout << "+----------------------+------------+-------------------+-------------+\n";

    return 0;
}
