#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Vector = std::array<double, 2>;
using Function = std::function<double(const Vector&)>;
using Gradient = std::function<Vector(const Vector&)>;

constexpr double EPSILON = 0.01;
constexpr double LINE_EPSILON = 1e-6;
constexpr double SVENN_DELTA = 0.01;

Vector operator+(Vector a, const Vector& b) {
    a[0] += b[0];
    a[1] += b[1];
    return a;
}

Vector operator-(const Vector& a) {
    return { -a[0], -a[1] };
}

Vector operator*(double value, Vector a) {
    a[0] *= value;
    a[1] *= value;
    return a;
}

double dot(const Vector& a, const Vector& b) {
    return a[0] * b[0] + a[1] * b[1];
}

double vectorNorm(const Vector& a) {
    return std::sqrt(dot(a, a));
}

Vector multiplyByHessian(const Vector& v) {
    return { 2.0 * v[0] + 2.0 * v[1],
            2.0 * v[0] + 6.0 * v[1] };
}

double objective(const Vector& x) {
    return x[0] * x[0] + 3.0 * x[1] * x[1]
        + 2.0 * x[0] * x[1] - 9.0 * x[0] - 17.0 * x[1];
}

Vector objectiveGradient(const Vector& x) {
    return { 2.0 * x[0] + 2.0 * x[1] - 9.0,
            2.0 * x[0] + 6.0 * x[1] - 17.0 };
}

double rosenbrock(const Vector& x) {
    return std::pow(1.0 - x[0], 2)
        + 100.0 * std::pow(x[1] - x[0] * x[0], 2);
}

Vector rosenbrockGradient(const Vector& x) {
    return {
        -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]),
        200.0 * (x[1] - x[0] * x[0])
    };
}

struct Counter {
    long functionCalls = 0;
    long gradientCalls = 0;
};

struct IterationRow {
    int k;
    Vector x;
    double fx;
    double gradientNorm;
    double h;
    double beta;
    bool restart;
    double orthogonality;
    double conjugacy;
    double descent;
};

struct Result {
    Vector x{};
    double fx = 0.0;
    int iterations = 0;
    long functionCalls = 0;
    long gradientCalls = 0;
    std::vector<IterationRow> rows;
};

std::pair<double, double> svenn(
    const std::function<double(double)>& phi,
    double x0 = 0.0,
    double delta = SVENN_DELTA
) {
    const double f0 = phi(x0);
    const double fRight = phi(x0 + delta);
    const double fLeft = phi(x0 - delta);

    if (fLeft >= f0 && fRight >= f0) {
        return { x0 - delta, x0 + delta };
    }

    double previous = x0;
    double current;
    double currentValue;

    if (fLeft < fRight) {
        delta = -delta;
        current = x0 + delta;
        currentValue = fLeft;
    }
    else {
        current = x0 + delta;
        currentValue = fRight;
    }

    for (int power = 1; power <= 60; ++power) {
        const double candidate = current + delta * std::pow(2.0, power);
        const double candidateValue = phi(candidate);

        if (candidateValue >= currentValue) {
            return std::minmax(previous, candidate);
        }

        previous = current;
        current = candidate;
        currentValue = candidateValue;
    }

    throw std::runtime_error("Svenn method could not bracket the minimum");
}

double goldenSection(
    const std::function<double(double)>& phi,
    double left,
    double right,
    double epsilon = LINE_EPSILON
) {
    const double tau = (std::sqrt(5.0) - 1.0) / 2.0;
    double x1 = left + (1.0 - tau) * (right - left);
    double x2 = left + tau * (right - left);
    double f1 = phi(x1);
    double f2 = phi(x2);

    while (right - left > epsilon) {
        if (f1 <= f2) {
            right = x2;
            x2 = x1;
            f2 = f1;
            x1 = left + (1.0 - tau) * (right - left);
            f1 = phi(x1);
        }
        else {
            left = x1;
            x1 = x2;
            f1 = f2;
            x2 = left + tau * (right - left);
            f2 = phi(x2);
        }
    }

    return (left + right) / 2.0;
}

double lineSearch(const Function& function,
    const Vector& x,
    const Vector& direction) {
    const auto phi = [&](double h) {
        return function(x + h * direction);
        };

    const auto [left, right] = svenn(phi);
    return goldenSection(phi, left, right);
}

Result fletcherReeves(
    const Function& rawFunction,
    const Gradient& rawGradient,
    Vector x,
    double epsilon = EPSILON,
    int restartPeriod = 2,
    bool calculateConjugacy = false
) {
    Counter counter;
    Function function = [&](const Vector& value) {
        ++counter.functionCalls;
        return rawFunction(value);
        };
    Gradient gradient = [&](const Vector& value) {
        ++counter.gradientCalls;
        return rawGradient(value);
        };

    Vector g = gradient(x);
    Vector direction = -g;
    Result result;

    for (int k = 0; k < 5000; ++k) {
        const double fx = function(x);
        const double gradientNorm = vectorNorm(g);

        if (gradientNorm <= epsilon) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            result.rows.push_back({ k, x, fx, gradientNorm, nan, nan,false, nan, nan, nan });
            result.iterations = k;
            break;
        }

        const Vector oldDirection = direction;
        const double h = lineSearch(function, x, oldDirection);
        const Vector newX = x + h * oldDirection;
        const Vector newGradient = gradient(newX);

        double beta = dot(newGradient, newGradient) / dot(g, g);
        const bool restart = (restartPeriod > 0) && ((k + 1) % restartPeriod == 0);
        if (restart) {
            beta = 0.0;
        }

        const Vector newDirection = -newGradient + beta * oldDirection;
        const double conjugacy = calculateConjugacy ? dot(newDirection, multiplyByHessian(oldDirection)) : std::numeric_limits<double>::quiet_NaN();

        result.rows.push_back({
            k, x, fx, gradientNorm, h, beta, restart,
            dot(newGradient, oldDirection),
            conjugacy,
            dot(newGradient, newDirection)
            });

        x = newX;
        g = newGradient;
        direction = newDirection;
    }

    result.x = x;
    result.fx = function(x);
    result.functionCalls = counter.functionCalls;
    result.gradientCalls = counter.gradientCalls;
    return result;
}

Result antigradientExact(const Function& rawFunction, const Gradient& rawGradient, Vector x) {
    Counter counter;
    Function function = [&](const Vector& value) {
        ++counter.functionCalls;
        return rawFunction(value);
        };
    Gradient gradient = [&](const Vector& value) {
        ++counter.gradientCalls;
        return rawGradient(value);
        };

    int iterations = 0;
    while (iterations < 5000) {
        const Vector g = gradient(x);
        (void)function(x);
        if (vectorNorm(g) <= EPSILON) {
            break;
        }

        const Vector direction = -g;
        x = x + lineSearch(function, x, direction) * direction;
        ++iterations;
    }

    return { x, function(x), iterations,
            counter.functionCalls, counter.gradientCalls,{} };
}

Result gradientBacktracking(const Function& rawFunction, const Gradient& rawGradient, Vector x) {
    Counter counter;
    Function function = [&](const Vector& value) { ++counter.functionCalls; return rawFunction(value);};
    Gradient gradient = [&](const Vector& value) {++counter.gradientCalls; return rawGradient(value); };

    int iterations = 0;
    while (iterations < 5000) {
        const Vector g = gradient(x);
        if (vectorNorm(g) <= EPSILON) {
            break;
        }

        double h = 1.0;
        const double fx = function(x);
        while (function(x + (-h) * g) > fx - 1e-4 * h * dot(g, g)) 
        {
            h *= 0.5;
        }

        x = x + (-h) * g;
        ++iterations;
    }

    return { x, function(x), iterations, counter.functionCalls, counter.gradientCalls,{} };
}

void printIterationTable(const Result& result) {
    std::cout
        << "+---+-------------------------+--------------+------------+----------+----------+---------+\n"
        << "| k | x_k                     | F(x_k)       | ||grad F|| | h_k      | beta_k   | Restart |\n"
        << "+---+-------------------------+--------------+------------+----------+----------+---------+\n";

    for (const auto& row : result.rows) {
        std::cout << "| " << std::setw(1) << row.k << " | ("
            << std::setw(9) << std::fixed << std::setprecision(6) << row.x[0]
            << "; " << std::setw(9) << row.x[1] << ") | "
            << std::setw(12) << row.fx << " | "
            << std::setw(10) << row.gradientNorm << " | ";

        if (std::isnan(row.h)) {
            std::cout << std::setw(8) << "-" << " | "
                << std::setw(8) << "-" << " | "
                << std::setw(7) << "-" << " |\n";
        }
        else {
            std::cout << std::setw(8) << row.h << " | "
                << std::setw(8) << row.beta << " | "
                << std::setw(7) << (row.restart ? "yes" : "no") << " |\n";
        }
    }

    std::cout
        << "+---+-------------------------+--------------+------------+----------+----------+---------+\n";
}

void printChecksTable(const Result& result) {
    std::cout
        << "+---+------------------+---------------------+--------------------+\n"
        << "| k | <g_next,d_k>     | d_next^T H d_k      | <g_next,d_next>    |\n"
        << "+---+------------------+---------------------+--------------------+\n";

    for (const auto& row : result.rows) {
        if (!std::isnan(row.orthogonality)) {
            std::cout << "| " << row.k << " | "
                << std::setw(16) << std::scientific << row.orthogonality << " | "
                << std::setw(19) << row.conjugacy << " | "
                << std::setw(18) << row.descent << " |\n";
        }
    }

    std::cout
        << "+---+------------------+---------------------+--------------------+\n";
}

void printComparisonTable(const Result& gradient,
    const Result& antigradient,
    const Result& fletcher) {
    std::cout
        << "+-------------------------------+------+-------+-------+-------------------------+\n"
        << "| Method                        | Iter | Nf    | Ngrad | x*                      |\n"
        << "+-------------------------------+------+-------+-------+-------------------------+\n";

    const auto printRow = [](const std::string& name, const Result& result) {
        std::cout << "| " << std::left << std::setw(29) << name
            << " | " << std::right << std::setw(4) << result.iterations
            << " | " << std::setw(5) << result.functionCalls
            << " | " << std::setw(5) << result.gradientCalls
            << " | (" << std::fixed << std::setprecision(6)
            << std::setw(9) << result.x[0] << "; "
            << std::setw(9) << result.x[1] << ") |\n";
        };

    printRow("Gradient with backtracking", gradient);
    printRow("Antigradient + exact search", antigradient);
    printRow("Fletcher-Reeves", fletcher);

    std::cout<< "+-------------------------------+------+-------+-------+-------------------------+\n";
}

void printRosenbrockTable(const Result& withRestart,
    const Result& withoutRestart) {
    std::cout
        << "+-------------------------------+------+-------+-------+-------------------------+\n"
        << "| Method                        | Iter | Nf    | Ngrad | x*                      |\n"
        << "+-------------------------------+------+-------+-------+-------------------------+\n";

    const auto printRow = [](const std::string& name, const Result& result) {
        std::cout << "| " << std::left << std::setw(29) << name
            << " | " << std::right << std::setw(4) << result.iterations
            << " | " << std::setw(5) << result.functionCalls
            << " | " << std::setw(5) << result.gradientCalls
            << " | (" << std::fixed << std::setprecision(6)
            << std::setw(9) << result.x[0] << "; "
            << std::setw(9) << result.x[1] << ") |\n";
        };

    printRow("Fletcher-Reeves, restart 2", withRestart);
    printRow("Fletcher-Reeves, no restart", withoutRestart);

    std::cout<< "+-------------------------------+------+-------+-------+-------------------------+\n";
}

int main() {
    const Vector start{ -2.0, 6.0 };
    const Vector optimum{ 2.5, 2.0 };

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "LABORATORY WORK 9 - VARIANT 8\n"
        << "F(x1,x2) = x1^2 + 3*x2^2 + 2*x1*x2 - 9*x1 - 17*x2\n"
        << "Initial point = (-2.000000; 6.000000)\n"
        << "epsilon = 0.010000\n"
        << "Analytic minimum = (2.500000; 2.000000)\n"
        << "F(minimum) = " << objective(optimum) << "\n";

    const Vector g0 = objectiveGradient(start);
    const Vector d0 = -g0;
    const double analyticH = -dot(g0, d0) / dot(d0, multiplyByHessian(d0));
    const double numericalH = lineSearch(objective, start, d0);

    std::cout << "\nFIRST LINE SEARCH CHECK\n"
        << "analytic h* = " << std::setprecision(9) << analyticH << "\n"
        << "Svenn + golden h* = " << numericalH << "\n"
        << "absolute difference = " << std::scientific
        << std::abs(analyticH - numericalH) << "\n";

    const Result fletcher = fletcherReeves(
        objective, objectiveGradient, start, EPSILON, 2, true
    );
    const Result antigradient = antigradientExact(
        objective, objectiveGradient, start
    );
    const Result gradient = gradientBacktracking(
        objective, objectiveGradient, start
    );

    std::cout << "\nFLETCHER-REEVES ITERATIONS\n";
    printIterationTable(fletcher);

    std::cout << "\nTHREE REQUIRED CHECKS\n";
    printChecksTable(fletcher);

    std::cout << "\nMETHOD COMPARISON\n";
    printComparisonTable(gradient, antigradient, fletcher);

    const Result rosenbrockRestart = fletcherReeves(
        rosenbrock, rosenbrockGradient, { -1.2, 1.0 }, EPSILON, 2
    );
    const Result rosenbrockNoRestart = fletcherReeves(
        rosenbrock, rosenbrockGradient, { -1.2, 1.0 }, EPSILON, 0
    );

    std::cout << "\nROSENBROCK EXPERIMENT\n";
    printRosenbrockTable(rosenbrockRestart, rosenbrockNoRestart);
}
