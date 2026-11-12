#pragma once

#include <functional>
#include <string>
#include <vector>

namespace lr11 {

struct Vec2 {
    double x = 0.0;
    double y = 0.0;
};

Vec2 operator+(Vec2 a, Vec2 b);
Vec2 operator-(Vec2 a, Vec2 b);
Vec2 operator-(Vec2 a);
Vec2 operator*(double scalar, Vec2 value);
Vec2 operator*(Vec2 value, double scalar);
Vec2 operator/(Vec2 value, double scalar);
double dot(Vec2 a, Vec2 b);
double norm(Vec2 value);

double objective(Vec2 x);
Vec2 objectiveGradient(Vec2 x);

double positivePart(double value);
double g1(Vec2 x);                    // 4 - x1 - x2 <= 0
double g2(Vec2 x);                    // x2 - 1 <= 0
double circleConstraint(Vec2 x, double radius = 3.0);

struct Constraint {
    std::string name;
    std::function<double(Vec2)> value;
    std::function<Vec2(Vec2)> gradient;
};

Constraint linearConstraint1();
Constraint linearConstraint2();
Constraint outsideCircle(double radius);

struct InnerRow {
    int iteration = 0;
    Vec2 point{};
    double value = 0.0;
    double gradientNorm = 0.0;
    double step = 0.0;
    double beta = 0.0;
    bool restart = false;
};

struct InnerResult {
    Vec2 point{};
    int iterations = 0;
    int functionCalls = 0;
    int gradientCalls = 0;
    bool converged = false;
    std::vector<InnerRow> rows;
};

struct OuterRow {
    int iteration = 0;
    double parameter = 0.0;
    Vec2 start{};
    Vec2 point{};
    double objectiveValue = 0.0;
    double violation = 0.0;
    double weightedPenalty = 0.0;
    std::vector<double> multipliers;
    int innerIterations = 0;
    int functionCalls = 0;
    int gradientCalls = 0;
};

struct MethodResult {
    Vec2 point{};
    std::vector<OuterRow> rows;
    int totalFunctionCalls = 0;
    int totalGradientCalls = 0;
    bool converged = false;
};

struct RunConfig {
    double r0 = 1.0;
    double multiplier = 10.0;
    double outerTolerance = 0.01;
    double innerTolerance = 0.01;
    int maxOuterIterations = 30;
    int maxInnerIterations = 2000;
};

InnerResult fletcherReeves(const std::function<double(Vec2)>& function,
                           const std::function<Vec2(Vec2)>& gradient,
                           Vec2 start,
                           double tolerance = 0.01,
                           int maxIterations = 2000);

InnerResult sphericalGradient(const std::function<double(Vec2)>& function,
                              const std::function<Vec2(Vec2)>& gradient,
                              Vec2 start,
                              double tolerance = 0.01,
                              double initialStep = 1.0,
                              int maxIterations = 100000);

MethodResult externalPenalty(Vec2 start,
                             const std::vector<Constraint>& inequalities,
                             const RunConfig& config = {});

MethodResult equalityPenalty(Vec2 start,
                             const Constraint& equality,
                             const RunConfig& config = {});

MethodResult interiorBarrier(Vec2 start,
                             const std::vector<Constraint>& inequalities,
                             double mu0 = 10.0,
                             double divisor = 10.0,
                             double outerTolerance = 0.01,
                             double innerTolerance = 0.01);

Vec2 analyticSingleConstraint(double r);
Vec2 projectToG1Boundary(Vec2 point);

}  // namespace lr11
