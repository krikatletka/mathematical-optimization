#include "optimization.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

static bool close(double a, double b, double eps) {
    return std::abs(a - b) <= eps;
}

int main() {
    using namespace lr11;

    const Vec2 p{2.5, 1.5};
    assert(close(objective(p), 23.0, 1e-12));
    const Vec2 gp = objectiveGradient(p);
    assert(close(gp.x, 11.5, 1e-12));
    assert(close(gp.y, 11.5, 1e-12));

    assert(close(g1(p), 0.0, 1e-12));
    assert(close(g2({3.0, 1.0}), 0.0, 1e-12));
    assert(close(circleConstraint({3.0, 0.0}), 0.0, 1e-12));

    const Vec2 analytic = analyticSingleConstraint(10000.0);
    assert(close(analytic.x, 2.49964, 5e-4));
    assert(close(analytic.y, 1.49978, 5e-4));

    RunConfig config;
    config.r0 = 1.0;
    config.multiplier = 10.0;
    config.outerTolerance = 0.01;
    config.innerTolerance = 0.01;
    const auto result = externalPenalty({5.0, -3.0}, {linearConstraint1()}, config);
    assert(result.rows.size() == 5);
    assert(close(result.point.x, 2.5, 0.01));
    assert(close(result.point.y, 1.5, 0.01));
    assert(positivePart(g1(result.point)) < 0.001);

    const Vec2 polished = projectToG1Boundary(result.point);
    assert(close(g1(polished), 0.0, 1e-12));
    assert(close(objective(polished), 23.0, 1e-3));

    const auto two = externalPenalty({5.0, -3.0},
                                     {linearConstraint1(), linearConstraint2()}, config);
    assert(close(two.point.x, 3.0, 0.01));
    assert(close(two.point.y, 1.0, 0.01));

    const auto circleA = externalPenalty({5.0, -3.0}, {outsideCircle(3.0)}, config);
    const auto circleB = externalPenalty({-5.0, 3.0}, {outsideCircle(3.0)}, config);
    assert(close(norm(circleA.point), 3.0, 0.01));
    assert(close(norm(circleB.point), 3.0, 0.01));
    assert(circleA.point.x * circleB.point.x < 0.0);

    const auto barrier = interiorBarrier({5.0, 1.0}, {linearConstraint1()}, 10.0, 10.0,
                                         0.01, 0.01);
    assert(barrier.rows.size() == 4);
    assert(g1(barrier.point) < 0.0);
    assert(close(barrier.point.x, 2.5, 0.02));
    assert(close(barrier.point.y, 1.5, 0.02));

    std::cout << "All LR11 control tests passed.\n";
}
