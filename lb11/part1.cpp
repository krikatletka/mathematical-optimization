#include "optimization.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using lr11::Vec2;

namespace {

void printPoint(Vec2 p) {
    std::cout << '(' << std::fixed << std::setprecision(6) << p.x << "; " << p.y << ')';
}

void printInnerProtocol(const lr11::InnerResult& result) {
    std::cout << "\nFULL INNER PROTOCOL FOR r = 1\n";
    std::cout << "+----+-------------------------+-------------+-------------+-----------+----------+---------+\n";
    std::cout << "| k  | x                       | Phi(x)      | ||grad Phi|| | h         | beta     | restart |\n";
    std::cout << "+----+-------------------------+-------------+-------------+-----------+----------+---------+\n";
    for (const auto& row : result.rows) {
        std::ostringstream point;
        point << '(' << std::fixed << std::setprecision(6) << row.point.x << "; " << row.point.y << ')';
        std::cout << "| " << std::setw(2) << row.iteration << " | " << std::left << std::setw(23)
                  << point.str() << std::right << " | " << std::setw(11) << std::fixed
                  << std::setprecision(6) << row.value << " | " << std::setw(11)
                  << row.gradientNorm << " | " << std::setw(9) << row.step << " | "
                  << std::setw(8) << row.beta << " | " << std::setw(7)
                  << (row.restart ? "yes" : "no") << " |\n";
    }
    std::cout << "+----+-------------------------+-------------+-------------+-----------+----------+---------+\n";
    std::cout << "Iterations = " << result.iterations << ", Nphi = " << result.functionCalls
              << ", Ngrad = " << result.gradientCalls << "\n";
}

void printOuterProtocol(const lr11::MethodResult& result) {
    std::cout << "\nEXTERNAL PENALTY PROTOCOL\n";
    std::cout << "+---+---------+-------------------------+------------+------------+------------+----------+----------+\n";
    std::cout << "| k | r       | x(r)                    | F(x)       | g+         | r*P        | 2r*g     | Nphi sum |\n";
    std::cout << "+---+---------+-------------------------+------------+------------+------------+----------+----------+\n";
    int cumulative = 0;
    for (const auto& row : result.rows) {
        cumulative += row.functionCalls;
        std::ostringstream point;
        point << '(' << std::fixed << std::setprecision(6) << row.point.x << "; " << row.point.y << ')';
        std::cout << "| " << row.iteration << " | " << std::setw(7) << std::scientific
                  << std::setprecision(0) << row.parameter << " | " << std::left << std::setw(23)
                  << point.str() << std::right << " | " << std::setw(10) << std::fixed
                  << std::setprecision(6) << row.objectiveValue << " | " << std::setw(10)
                  << row.violation << " | " << std::setw(10) << row.weightedPenalty << " | "
                  << std::setw(8) << row.multipliers.at(0) << " | " << std::setw(8)
                  << cumulative << " |\n";
    }
    std::cout << "+---+---------+-------------------------+------------+------------+------------+----------+----------+\n";
}

}  // namespace

int main() {
    using namespace lr11;
    std::cout << "LABORATORY WORK 11 - PART 1\n";
    std::cout << "EXTERNAL QUADRATIC PENALTY METHOD\n";
    std::cout << "F(x1,x2) = 2*x1^2 + 3*x2^2 + x1*x2\n";
    std::cout << "Start = (5; -3)\n";
    std::cout << "g(x) = 4 - x1 - x2 <= 0  (x1+x2 >= 4)\n";
    std::cout << "Control solution: x* = (2.5; 1.5), F* = 23, lambda = 11.5\n";

    const Constraint constraint = linearConstraint1();
    auto phi1 = [constraint](Vec2 p) {
        const double s = positivePart(constraint.value(p));
        return objective(p) + s * s;
    };
    auto gradPhi1 = [constraint](Vec2 p) {
        const double s = positivePart(constraint.value(p));
        return objectiveGradient(p) + (2.0 * s) * constraint.gradient(p);
    };
    const InnerResult firstInner = fletcherReeves(phi1, gradPhi1, {5.0, -3.0}, 0.01);
    printInnerProtocol(firstInner);

    RunConfig config;
    config.r0 = 1.0;
    config.multiplier = 10.0;
    config.outerTolerance = 0.01;
    config.innerTolerance = 0.01;
    const MethodResult result = externalPenalty({5.0, -3.0}, {constraint}, config);
    printOuterProtocol(result);

    std::cout << "\nANALYTIC CHECK x(r) = (40r;24r)/(23+16r)\n";
    for (const auto& row : result.rows) {
        const Vec2 exact = analyticSingleConstraint(row.parameter);
        std::cout << "r=" << std::scientific << row.parameter << ": numerical ";
        printPoint(row.point);
        std::cout << ", analytic ";
        printPoint(exact);
        std::cout << ", distance=" << std::scientific << norm(row.point - exact) << '\n';
    }

    const Vec2 polished = projectToG1Boundary(result.point);
    std::cout << "\nFINAL PENALTY POINT = "; printPoint(result.point);
    std::cout << "\nF = " << std::fixed << std::setprecision(9) << objective(result.point)
              << ", violation = " << positivePart(g1(result.point)) << '\n';
    std::cout << "Projected to active boundary = "; printPoint(polished);
    std::cout << "\nF(projected) = " << objective(polished)
              << ", violation = " << positivePart(g1(polished)) << '\n';

    const double directR = 1e6;
    auto directPhi = [=](Vec2 p) {
        const double s = positivePart(g1(p));
        return objective(p) + directR * s * s;
    };
    auto directGrad = [=](Vec2 p) {
        const double s = positivePart(g1(p));
        return objectiveGradient(p) + (2.0 * directR * s) * Vec2{-1.0, -1.0};
    };
    const InnerResult direct = fletcherReeves(directPhi, directGrad, {5.0, -3.0}, 0.01, 5000);
    std::cout << "\nCOMPARISON WITH r=1e6 DIRECTLY FROM THE ORIGINAL START\n";
    std::cout << "Warm sequence: outer=" << result.rows.size() << ", Nphi="
              << result.totalFunctionCalls << ", Ngrad=" << result.totalGradientCalls << '\n';
    std::cout << "Direct r=1e6: iterations=" << direct.iterations << ", Nphi="
              << direct.functionCalls << ", Ngrad=" << direct.gradientCalls
              << ", converged=" << (direct.converged ? "yes" : "no") << '\n';

    std::filesystem::create_directories("outputs");
    std::ofstream csv("outputs/part1_outer_protocol.csv");
    csv << "k,r,x1,x2,F,violation,rP,lambda_est,inner_iterations,Nphi,Ngrad\n";
    for (const auto& row : result.rows) {
        csv << row.iteration << ',' << row.parameter << ',' << row.point.x << ',' << row.point.y
            << ',' << row.objectiveValue << ',' << row.violation << ',' << row.weightedPenalty
            << ',' << row.multipliers.at(0) << ',' << row.innerIterations << ','
            << row.functionCalls << ',' << row.gradientCalls << '\n';
    }
    std::cout << "Saved data for plots: outputs/part1_outer_protocol.csv\n";
}
