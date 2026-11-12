#include "optimization.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lr11 {

Vec2 operator+(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
Vec2 operator-(Vec2 a, Vec2 b) { return {a.x - b.x, a.y - b.y}; }
Vec2 operator-(Vec2 a) { return {-a.x, -a.y}; }
Vec2 operator*(double s, Vec2 v) { return {s * v.x, s * v.y}; }
Vec2 operator*(Vec2 v, double s) { return s * v; }
Vec2 operator/(Vec2 v, double s) { return {v.x / s, v.y / s}; }
double dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
double norm(Vec2 v) { return std::sqrt(dot(v, v)); }

double objective(Vec2 p) {
    return 2.0 * p.x * p.x + 3.0 * p.y * p.y + p.x * p.y;
}

Vec2 objectiveGradient(Vec2 p) {
    return {4.0 * p.x + p.y, p.x + 6.0 * p.y};
}

double positivePart(double value) { return std::max(0.0, value); }
double g1(Vec2 p) { return 4.0 - p.x - p.y; }
double g2(Vec2 p) { return p.y - 1.0; }
double circleConstraint(Vec2 p, double radius) {
    return radius * radius - dot(p, p);
}

Constraint linearConstraint1() {
    return {"4-x1-x2<=0", [](Vec2 p) { return g1(p); },
            [](Vec2) { return Vec2{-1.0, -1.0}; }};
}

Constraint linearConstraint2() {
    return {"x2-1<=0", [](Vec2 p) { return g2(p); },
            [](Vec2) { return Vec2{0.0, 1.0}; }};
}

Constraint outsideCircle(double radius) {
    return {"R^2-x1^2-x2^2<=0",
            [radius](Vec2 p) { return circleConstraint(p, radius); },
            [](Vec2 p) { return Vec2{-2.0 * p.x, -2.0 * p.y}; }};
}

namespace {

struct Bracket { double left; double right; };

Bracket svennBracket(const std::function<double(double)>& phi) {
    const double f0 = phi(0.0);
    double delta = 0.01;
    double right = delta;
    double fRight = phi(right);

    for (int i = 0; i < 60 && (!std::isfinite(fRight) || fRight >= f0); ++i) {
        delta *= 0.5;
        right = delta;
        fRight = phi(right);
    }
    if (!std::isfinite(fRight) || fRight >= f0) return {0.0, delta};

    double previous = 0.0;
    double current = right;
    double fCurrent = fRight;
    for (int k = 1; k <= 60; ++k) {
        const double next = current + std::ldexp(delta, k);
        const double fNext = phi(next);
        if (!std::isfinite(fNext) || fNext >= fCurrent) return {previous, next};
        previous = current;
        current = next;
        fCurrent = fNext;
    }
    return {previous, current};
}

double goldenSection(const std::function<double(double)>& phi, double left, double right,
                     double tolerance = 1e-6) {
    constexpr double tau = 0.6180339887498948482;
    double x1 = right - tau * (right - left);
    double x2 = left + tau * (right - left);
    double f1 = phi(x1);
    double f2 = phi(x2);
    for (int k = 0; k < 200 && right - left > tolerance; ++k) {
        if (f1 <= f2) {
            right = x2; x2 = x1; f2 = f1;
            x1 = right - tau * (right - left); f1 = phi(x1);
        } else {
            left = x1; x1 = x2; f1 = f2;
            x2 = left + tau * (right - left); f2 = phi(x2);
        }
    }
    return 0.5 * (left + right);
}

double lineSearch(const std::function<double(Vec2)>& evaluate, Vec2 x, Vec2 direction) {
    auto phi = [&](double alpha) { return evaluate(x + alpha * direction); };
    const Bracket bracket = svennBracket(phi);
    return goldenSection(phi, bracket.left, bracket.right);
}

}  // namespace

InnerResult fletcherReeves(const std::function<double(Vec2)>& function,
                           const std::function<Vec2(Vec2)>& gradient,
                           Vec2 start, double tolerance, int maxIterations) {
    InnerResult result;
    auto f = [&](Vec2 p) { ++result.functionCalls; return function(p); };
    auto grad = [&](Vec2 p) { ++result.gradientCalls; return gradient(p); };

    Vec2 x = start;
    Vec2 g = grad(x);
    Vec2 direction = -g;
    result.rows.push_back({0, x, f(x), norm(g), 0.0, 0.0, false});

    for (int k = 0; k < maxIterations && norm(g) > tolerance; ++k) {
        if (dot(g, direction) >= 0.0) direction = -g;
        const double alpha = lineSearch(f, x, direction);
        const Vec2 next = x + alpha * direction;
        const Vec2 nextGradient = grad(next);
        double beta = dot(nextGradient, nextGradient) / std::max(dot(g, g), 1e-300);
        const bool restart = ((k + 1) % 2 == 0);
        if (restart) beta = 0.0;
        Vec2 nextDirection = -nextGradient + beta * direction;
        if (dot(nextGradient, nextDirection) >= 0.0) {
            nextDirection = -nextGradient;
            beta = 0.0;
        }
        result.rows.push_back({k + 1, next, f(next), norm(nextGradient), alpha, beta, restart});
        x = next;
        g = nextGradient;
        direction = nextDirection;
    }
    result.point = x;
    result.iterations = static_cast<int>(result.rows.size()) - 1;
    result.converged = norm(g) <= tolerance;
    return result;
}

InnerResult sphericalGradient(const std::function<double(Vec2)>& function,
                              const std::function<Vec2(Vec2)>& gradient,
                              Vec2 start, double tolerance, double initialStep,
                              int maxIterations) {
    InnerResult result;
    auto f = [&](Vec2 p) { ++result.functionCalls; return function(p); };
    auto grad = [&](Vec2 p) { ++result.gradientCalls; return gradient(p); };
    Vec2 x = start;
    double fx = f(x);
    double step = initialStep;
    for (int k = 0; k <= maxIterations; ++k) {
        const Vec2 g = grad(x);
        result.rows.push_back({k, x, fx, norm(g), step, 0.0, false});
        if (norm(g) <= tolerance) { result.converged = true; break; }
        Vec2 candidate = x - step * g;
        double candidateValue = f(candidate);
        while (candidateValue >= fx && step > 1e-15) {
            step *= 0.5;
            candidate = x - step * g;
            candidateValue = f(candidate);
        }
        if (step <= 1e-15) break;
        x = candidate;
        fx = candidateValue;
    }
    result.point = x;
    result.iterations = static_cast<int>(result.rows.size()) - 1;
    return result;
}

MethodResult externalPenalty(Vec2 start, const std::vector<Constraint>& cons,
                             const RunConfig& config) {
    MethodResult result;
    Vec2 x = start;
    double r = config.r0;
    for (int outer = 1; outer <= config.maxOuterIterations; ++outer) {
        const Vec2 innerStart = x;
        auto phi = [=](Vec2 p) {
            double value = objective(p);
            for (const auto& c : cons) value += r * std::pow(positivePart(c.value(p)), 2);
            return value;
        };
        auto gradPhi = [=](Vec2 p) {
            Vec2 value = objectiveGradient(p);
            for (const auto& c : cons) {
                const double s = positivePart(c.value(p));
                value = value + (2.0 * r * s) * c.gradient(p);
            }
            return value;
        };
        const InnerResult inner = fletcherReeves(phi, gradPhi, x,
                                                  config.innerTolerance,
                                                  config.maxInnerIterations);
        x = inner.point;
        double penalty = 0.0;
        double maximumViolation = 0.0;
        std::vector<double> multipliers;
        for (const auto& c : cons) {
            const double s = positivePart(c.value(x));
            penalty += s * s;
            maximumViolation = std::max(maximumViolation, s);
            multipliers.push_back(2.0 * r * s);
        }
        result.rows.push_back({outer, r, innerStart, x, objective(x), maximumViolation,
                               r * penalty, multipliers, inner.iterations,
                               inner.functionCalls, inner.gradientCalls});
        result.totalFunctionCalls += inner.functionCalls;
        result.totalGradientCalls += inner.gradientCalls;
        if (r * penalty <= config.outerTolerance) { result.converged = true; break; }
        r *= config.multiplier;
    }
    result.point = x;
    return result;
}

MethodResult equalityPenalty(Vec2 start, const Constraint& equality,
                             const RunConfig& config) {
    MethodResult result;
    Vec2 x = start;
    double r = config.r0;
    for (int outer = 1; outer <= config.maxOuterIterations; ++outer) {
        const Vec2 innerStart = x;
        auto phi = [=](Vec2 p) {
            const double h = equality.value(p);
            return objective(p) + r * h * h;
        };
        auto gradPhi = [=](Vec2 p) {
            const double h = equality.value(p);
            return objectiveGradient(p) + (2.0 * r * h) * equality.gradient(p);
        };
        const InnerResult inner = fletcherReeves(phi, gradPhi, x,
                                                  config.innerTolerance,
                                                  config.maxInnerIterations);
        x = inner.point;
        const double h = equality.value(x);
        result.rows.push_back({outer, r, innerStart, x, objective(x), std::abs(h), r * h * h,
                               {2.0 * r * h}, inner.iterations,
                               inner.functionCalls, inner.gradientCalls});
        result.totalFunctionCalls += inner.functionCalls;
        result.totalGradientCalls += inner.gradientCalls;
        if (r * h * h <= config.outerTolerance) { result.converged = true; break; }
        r *= config.multiplier;
    }
    result.point = x;
    return result;
}

MethodResult interiorBarrier(Vec2 start, const std::vector<Constraint>& cons,
                             double mu0, double divisor, double outerTolerance,
                             double innerTolerance) {
    for (const auto& c : cons) {
        if (c.value(start) >= 0.0) throw std::invalid_argument("Barrier start must be strictly feasible");
    }
    MethodResult result;
    Vec2 x = start;
    double mu = mu0;
    for (int outer = 1; outer <= 50; ++outer) {
        const Vec2 innerStart = x;
        auto phi = [=](Vec2 p) {
            double value = objective(p);
            for (const auto& c : cons) {
                const double gv = c.value(p);
                if (gv >= 0.0) return std::numeric_limits<double>::infinity();
                value -= mu * std::log(-gv);
            }
            return value;
        };
        auto gradPhi = [=](Vec2 p) {
            Vec2 value = objectiveGradient(p);
            for (const auto& c : cons) value = value - (mu / c.value(p)) * c.gradient(p);
            return value;
        };
        const InnerResult inner = fletcherReeves(phi, gradPhi, x, innerTolerance, 5000);
        x = inner.point;
        double closest = 0.0;
        std::vector<double> multipliers;
        for (const auto& c : cons) {
            closest = std::max(closest, -c.value(x));
            multipliers.push_back(-mu / c.value(x));
        }
        result.rows.push_back({outer, mu, innerStart, x, objective(x), closest,
                               mu * static_cast<double>(cons.size()), multipliers,
                               inner.iterations, inner.functionCalls, inner.gradientCalls});
        result.totalFunctionCalls += inner.functionCalls;
        result.totalGradientCalls += inner.gradientCalls;
        if (mu * static_cast<double>(cons.size()) <= outerTolerance) {
            result.converged = true; break;
        }
        mu /= divisor;
    }
    result.point = x;
    return result;
}

Vec2 analyticSingleConstraint(double r) {
    const double denominator = 23.0 + 16.0 * r;
    return {40.0 * r / denominator, 24.0 * r / denominator};
}

Vec2 projectToG1Boundary(Vec2 p) {
    const double violation = g1(p);
    return {p.x + 0.5 * violation, p.y + 0.5 * violation};
}

}  // namespace lr11
