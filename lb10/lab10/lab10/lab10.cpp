#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>

using Vector = std::array<double, 2>;
using Function = std::function<double(const Vector&)>;

constexpr double EPSILON = 0.01;

Vector operator+(Vector a, const Vector& b) {
    a[0] += b[0];
    a[1] += b[1];
    return a;
}

Vector operator-(Vector a, const Vector& b) {
    a[0] -= b[0];
    a[1] -= b[1];
    return a;
}

double distance(const Vector& a, const Vector& b) {
    return std::hypot(a[0] - b[0], a[1] - b[1]);
}

double objective(const Vector& x) {
    return x[0] * x[0] + 3.0 * x[1] * x[1]
        + 2.0 * x[0] * x[1] - 9.0 * x[0] - 17.0 * x[1];
}

double nonsmooth(const Vector& x) {
    return 2.0 * std::abs(x[0] - 1.0)
        + 3.0 * std::abs(x[1] + 2.0)
        + 0.5 * std::pow(x[0] - 1.0, 2);
}

double rosenbrock(const Vector& x) {
    return std::pow(1.0 - x[0], 2)
        + 100.0 * std::pow(x[1] - x[0] * x[0], 2);
}

struct CountedFunction {
    Function function;
    long calls = 0;
    std::vector<double> values;

    double operator()(const Vector& x) {
        ++calls;
        const double value = function(x);
        values.push_back(value);
        return value;
    }
};

struct ProtocolRow {
    int iteration;
    std::string action;
    Vector point;
    double value;
    double delta;
    std::string decision;
    long functionCalls;
};

struct Result {
    Vector x{};
    double fx = 0.0;
    double finalDelta = 0.0;
    int iterations = 0;
    int bases = 0;
    long functionCalls = 0;
    bool converged = false;
    std::vector<ProtocolRow> rows;
    std::vector<double> values;
};

std::pair<Vector, double> exploratorySearch(
    CountedFunction& function,
    Vector point,
    double pointValue,
    double delta,
    int iteration,
    std::vector<ProtocolRow>& rows,
    const std::string& origin
) {
    for (int coordinate = 0; coordinate < 2; ++coordinate) {
        Vector plus = point;
        plus[coordinate] += delta;
        const double plusValue = function(plus);

        if (plusValue < pointValue) {
            rows.push_back({ iteration,
                            origin + " +x" + std::to_string(coordinate + 1),
                            plus, plusValue, delta, "accepted", function.calls });
            point = plus;
            pointValue = plusValue;
            continue;
        }

        rows.push_back({ iteration,
                        origin + " +x" + std::to_string(coordinate + 1),
                        plus, plusValue, delta, "rejected", function.calls });

        Vector minus = point;
        minus[coordinate] -= delta;
        const double minusValue = function(minus);

        if (minusValue < pointValue) {
            rows.push_back({ iteration,
                            origin + " -x" + std::to_string(coordinate + 1),
                            minus, minusValue, delta, "accepted", function.calls });
            point = minus;
            pointValue = minusValue;
        }
        else {
            rows.push_back({ iteration,
                            origin + " -x" + std::to_string(coordinate + 1),
                            minus, minusValue, delta, "rejected", function.calls });
        }
    }

    return { point, pointValue };
}

std::pair<Vector, double> patternMove(
    CountedFunction& function,
    const Vector& oldBase,
    const Vector& newBase,
    double delta,
    int iteration,
    std::vector<ProtocolRow>& rows
) {
    const Vector pattern = newBase + (newBase - oldBase);
    const double patternValue = function(pattern);

    rows.push_back({ iteration, "pattern point", pattern, patternValue,
                    delta, "probe", function.calls });

    return { pattern, patternValue };
}

Result hookeJeeves(const Function& rawFunction, Vector start, double delta0 = 1.0, double epsilon = EPSILON,  int maxIterations = 10000,  long maxFunctionCalls = 200000) {
    if (delta0 <= 0.0 || epsilon <= 0.0) {
        throw std::invalid_argument("delta0 and epsilon must be positive");
    }

    CountedFunction function{ rawFunction, 0,{} };
    Vector base = start;
    double baseValue = function(base);
    double delta = delta0;

    Vector searchPoint = base;
    double searchValue = baseValue;
    bool fromPattern = false;
    int iteration = 0;
    int bases = 0;

    std::vector<ProtocolRow> rows{
        { 0, "initial base", base, baseValue, delta, "start", function.calls }
    };

    while (delta >= epsilon
        && iteration < maxIterations
        && function.calls < maxFunctionCalls) {
        ++iteration;

        const std::string origin = fromPattern
            ? "explore from pattern"
            : "explore from base";

        auto [candidate, candidateValue] = exploratorySearch(
            function, searchPoint, searchValue, delta, iteration, rows, origin
        );

        if (candidateValue < baseValue) {
            const Vector oldBase = base;
            base = candidate;
            baseValue = candidateValue;
            ++bases;

            rows.push_back({ iteration, "new base", base, baseValue, delta,
                            "accepted against base", function.calls });

            auto [pattern, patternValue] = patternMove(
                function, oldBase, base, delta, iteration, rows
            );

            searchPoint = pattern;
            searchValue = patternValue;
            fromPattern = true;
        }
        else if (fromPattern) {
            rows.push_back({ iteration, "return to base", base, baseValue, delta,
                            "pattern failed; step unchanged", function.calls });
            searchPoint = base;
            searchValue = baseValue;
            fromPattern = false;
        }
        else {
            delta /= 2.0;
            rows.push_back({ iteration, "reduce step", base, baseValue, delta,
                            "delta = delta / 2", function.calls });
            searchPoint = base;
            searchValue = baseValue;
        }
    }

    return {base, baseValue, delta, iteration, bases,  function.calls,  delta < epsilon, rows, function.values };
}

std::string formatPoint(const Vector& point) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(6) << "(" << point[0] << "; " << point[1] << ")";
    return stream.str();
}

void printProtocolTable(const Result& result) {
    std::cout
        << "+----+---------------------------+-------------------------+--------------+----------+--------------------------------+------+\n"
        << "| k  | Action                    | Point                   | F            | Delta    | Decision                       | Nf   |\n"
        << "+----+---------------------------+-------------------------+--------------+----------+--------------------------------+------+\n";

    for (const auto& row : result.rows) {
        std::cout << "| " << std::left << std::setw(2) << row.iteration
            << " | " << std::setw(25) << row.action
            << " | " << std::setw(23) << formatPoint(row.point)
            << " | " << std::right << std::setw(12)
            << std::fixed << std::setprecision(6) << row.value
            << " | " << std::setw(8) << row.delta
            << " | " << std::left << std::setw(30) << row.decision
            << " | " << std::right << std::setw(4) << row.functionCalls
            << " |\n";
    }

    std::cout
        << "+----+---------------------------+-------------------------+--------------+----------+--------------------------------+------+\n";
}

void printComparisonTable(const std::vector<std::pair<std::string, Result>>& results,
    const std::vector<Vector>& exactPoints) {
    std::cout
        << "+----------------------------+-------+------+--------+-------------------------+--------------+\n"
        << "| Experiment                 | Bases | Nf   | Error  | x*                      | F(x*)        |\n"
        << "+----------------------------+-------+------+--------+-------------------------+--------------+\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& [name, result] = results[i];
        std::cout << "| " << std::left << std::setw(26) << name
            << " | " << std::right << std::setw(5) << result.bases
            << " | " << std::setw(4) << result.functionCalls
            << " | " << std::setw(6) << std::scientific << std::setprecision(2)
            << distance(result.x, exactPoints[i])
            << " | " << std::left << std::setw(23) << formatPoint(result.x)
            << " | " << std::right << std::setw(12)
            << std::scientific << std::setprecision(4) << result.fx
            << " |\n";
    }

    std::cout
        << "+----------------------------+-------+------+--------+-------------------------+--------------+\n";
}

void printRosenbrockComparison(const Result& loose, const Result& precise) {
    std::cout
        << "+-------------------------------+-------+------+-------------------------+--------------+\n"
        << "| Method                        | Bases | Nf   | x*                      | F(x*)        |\n"
        << "+-------------------------------+-------+------+-------------------------+--------------+\n"
        << "| Spherical metric (Lab 8)      |  3797 | 7607 | -                       | -            |\n"
        << "| Fletcher-Reeves (Lab 9)       |    31 |  984 | -                       | -            |\n"
        << "| Hooke-Jeeves, eps=0.01        | "
        << std::setw(5) << loose.bases << " | " << std::setw(4) << loose.functionCalls
        << " | " << std::left << std::setw(23) << formatPoint(loose.x)
        << " | " << std::right << std::scientific << std::setprecision(4)
        << std::setw(12) << loose.fx << " |\n"
        << "| Hooke-Jeeves, eps=0.0001      | "
        << std::setw(5) << precise.bases << " | " << std::setw(4) << precise.functionCalls
        << " | " << std::left << std::setw(23) << formatPoint(precise.x)
        << " | " << std::right << std::scientific << std::setprecision(4)
        << std::setw(12) << precise.fx << " |\n"
        << "+-------------------------------+-------+------+-------------------------+--------------+\n";
}


int main() {
    const Vector start{ -2.0, 6.0 };
    const Vector exactMinimum{ 2.5, 2.0 };

    std::cout << std::fixed << std::setprecision(6)
        << "LABORATORY WORK 10 - VARIANT 8\n"
        << "HOOKE-JEEVES DIRECT SEARCH METHOD\n"
        << "F(x1,x2) = x1^2 + 3*x2^2 + 2*x1*x2 - 9*x1 - 17*x2\n"
        << "Initial point = " << formatPoint(start) << "\n"
        << "epsilon = 0.010000\n"
        << "Analytic minimum = " << formatPoint(exactMinimum) << "\n"
        << "F(minimum) = " << objective(exactMinimum) << "\n";

    const Result delta1 = hookeJeeves(objective, start, 1.0, EPSILON);
    const Result delta07 = hookeJeeves(objective, start, 0.7, EPSILON);

    const Function slideFunction = [](const Vector& x) {
        return 2.0 * x[0] * x[0] + 3.0 * x[1] * x[1] + x[0] * x[1];
        };
    const Result control = hookeJeeves(slideFunction, { 5.0, -3.0 }, 1.0, EPSILON);
    const std::vector<double> expected{ 62, 81, 47, 36, 18, 31, 9, 8, 12 };
    bool controlPassed = control.values.size() >= expected.size();
    for (std::size_t i = 0; i < expected.size() && controlPassed; ++i) {
        controlPassed = std::abs(control.values[i] - expected[i]) < 1e-12;
    }

    std::cout << "\nCONTROL CHECK: "
        << (controlPassed ? "PASSED" : "FAILED") << "\n";

    std::cout << "\nFULL PROTOCOL: DELTA0 = 1.0\n";
    printProtocolTable(delta1);

    const Result nonsmoothResult = hookeJeeves(
        nonsmooth, { -5.0, -3.0 }, 1.0, EPSILON
    );
    const Result rosenbrockLoose = hookeJeeves(
        rosenbrock, { -1.2, 1.0 }, 1.0, 0.01
    );
    const Result rosenbrockPrecise = hookeJeeves(
        rosenbrock, { -1.2, 1.0 }, 1.0, 0.0001, 20000
    );

    std::cout << "\nEXPERIMENT COMPARISON\n";
    printComparisonTable(
        {
            { "Variant 8, delta0=1.0", delta1 },
            { "Variant 8, delta0=0.7", delta07 },
            { "Nonsmooth function", nonsmoothResult },
            { "Rosenbrock, eps=0.01", rosenbrockLoose },
            { "Rosenbrock, eps=0.0001", rosenbrockPrecise }
        },
        { exactMinimum, exactMinimum,{ 1.0, -2.0 },{ 1.0, 1.0 },{ 1.0, 1.0 } }
    );

    std::cout << "\nROSENBROCK: LABS 8-10\n";
    printRosenbrockComparison(rosenbrockLoose, rosenbrockPrecise);
}
