#include "optimization.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using lr11::Vec2;

namespace {

struct Summary {
    std::string experiment;
    int outer = 0;
    int nf = 0;
    int ng = 0;
    Vec2 point{};
    double f = 0.0;
    double violation = 0.0;
};

void printPoint(Vec2 p) {
    std::cout << '(' << std::fixed << std::setprecision(6) << p.x << "; " << p.y << ')';
}

Summary summarize(std::string name, const lr11::MethodResult& result) {
    return {std::move(name), static_cast<int>(result.rows.size()), result.totalFunctionCalls,
            result.totalGradientCalls, result.point, lr11::objective(result.point),
            result.rows.empty() ? 0.0 : result.rows.back().violation};
}

void printExperiment(const std::string& name, const lr11::MethodResult& result) {
    std::cout << std::left << std::setw(30) << name << std::right
              << " outer=" << std::setw(2) << result.rows.size()
              << " Nphi=" << std::setw(6) << result.totalFunctionCalls
              << " Ngrad=" << std::setw(5) << result.totalGradientCalls << " x=";
    printPoint(result.point);
    std::cout << " F=" << std::setw(11) << std::fixed << std::setprecision(6)
              << lr11::objective(result.point)
              << " violation=" << std::scientific << std::setprecision(3)
              << (result.rows.empty() ? 0.0 : result.rows.back().violation) << '\n';
}

lr11::RunConfig config(double c = 10.0, double innerEps = 0.01) {
    lr11::RunConfig value;
    value.r0 = 1.0;
    value.multiplier = c;
    value.outerTolerance = 0.01;
    value.innerTolerance = innerEps;
    value.maxInnerIterations = 5000;
    return value;
}

}  // namespace

int main() {
    using namespace lr11;
    std::cout << "LABORATORY WORK 11 - PART 2\n";
    std::cout << "PENALTY AND BARRIER METHOD EXPERIMENTS\n";
    std::cout << "Control problem: F=2*x1^2+3*x2^2+x1*x2, start=(5;-3)\n\n";
    std::vector<Summary> summary;

    std::cout << "EXPERIMENT 1 - PENALTY MULTIPLIER c\n";
    for (double c : {2.0, 4.0, 10.0, 100.0}) {
        const auto result = externalPenalty({5.0, -3.0}, {linearConstraint1()}, config(c));
        std::ostringstream name; name << "c=" << c;
        printExperiment(name.str(), result);
        summary.push_back(summarize(name.str(), result));
    }

    std::cout << "\nEXPERIMENT 2 - INNER ACCURACY\n";
    for (double eps : {0.1, 0.01, 0.001}) {
        const auto result = externalPenalty({5.0, -3.0}, {linearConstraint1()}, config(10.0, eps));
        std::ostringstream name; name << "inner eps=" << eps;
        printExperiment(name.str(), result);
        summary.push_back(summarize(name.str(), result));
    }

    std::cout << "\nEXPERIMENT 3 - EQUALITY h(x)=4-x1-x2=0\n";
    const auto equality = equalityPenalty({5.0, -3.0}, linearConstraint1(), config());
    printExperiment("equality", equality);
    summary.push_back(summarize("equality", equality));

    std::cout << "\nEXPERIMENT 4 - LOGARITHMIC BARRIER\n";
    const auto barrier = interiorBarrier({5.0, 1.0}, {linearConstraint1()}, 10.0, 10.0, 0.01, 0.01);
    std::cout << "Strictly feasible start = (5;1)\n";
    for (const auto& row : barrier.rows) {
        std::cout << "mu=" << std::setw(8) << std::fixed << std::setprecision(4) << row.parameter
                  << " x="; printPoint(row.point);
        std::cout << " F=" << std::setw(10) << row.objectiveValue
                  << " g=" << std::scientific << g1(row.point)
                  << " lambda_est=" << row.multipliers.at(0) << '\n';
    }
    printExperiment("log barrier", barrier);
    summary.push_back(summarize("log barrier", barrier));

    std::cout << "\nEXPERIMENT 5 - TWO INEQUALITIES\n";
    const auto two = externalPenalty({5.0, -3.0},
                                     {linearConstraint1(), linearConstraint2()}, config());
    printExperiment("g1 and g2", two);
    std::cout << "Control: x*=(3;1), F*=24, lambda=(13;4)\nEstimated lambda = ("
              << two.rows.back().multipliers.at(0) << "; "
              << two.rows.back().multipliers.at(1) << ")\n";
    summary.push_back(summarize("two inequalities", two));

    std::cout << "\nEXPERIMENT 6 - NONLINEAR BOUNDARY, OUTSIDE CIRCLE R=3\n";
    const auto circlePositive = externalPenalty({5.0, -3.0}, {outsideCircle(3.0)}, config());
    const auto circleNegative = externalPenalty({-5.0, 3.0}, {outsideCircle(3.0)}, config());
    printExperiment("circle start (5,-3)", circlePositive);
    printExperiment("circle start (-5,3)", circleNegative);
    std::cout << "Control: ||x*||=3, F*=16.136, two symmetric minima\n";
    summary.push_back(summarize("circle positive start", circlePositive));
    summary.push_back(summarize("circle negative start", circleNegative));

    std::cout << "\nEXPERIMENT 7 - INNER METHOD AT r=100\n";
    const double r = 100.0;
    auto phi = [=](Vec2 p) {
        const double s = positivePart(g1(p));
        return objective(p) + r * s * s;
    };
    auto gradPhi = [=](Vec2 p) {
        const double s = positivePart(g1(p));
        return objectiveGradient(p) + (2.0 * r * s) * Vec2{-1.0, -1.0};
    };
    const auto fr = fletcherReeves(phi, gradPhi, {5.0, -3.0}, 0.01, 5000);
    const auto spherical = sphericalGradient(phi, gradPhi, {5.0, -3.0}, 0.01, 1.0, 100000);
    std::cout << "Fletcher-Reeves: iterations=" << fr.iterations << " Nphi=" << fr.functionCalls
              << " Ngrad=" << fr.gradientCalls << " x="; printPoint(fr.point); std::cout << '\n';
    std::cout << "Spherical gradient: iterations=" << spherical.iterations
              << " Nphi=" << spherical.functionCalls << " Ngrad=" << spherical.gradientCalls
              << " x="; printPoint(spherical.point); std::cout << '\n';

    std::cout << "\nEXPERIMENT 8 - SUMMARY\n";
    std::cout << "+------------------------------+-------+--------+-------+-------------------------+------------+------------+\n";
    std::cout << "| Experiment                   | Outer | Nphi   | Ngrad | x                       | F          | violation  |\n";
    std::cout << "+------------------------------+-------+--------+-------+-------------------------+------------+------------+\n";
    for (const auto& row : summary) {
        std::ostringstream point;
        point << '(' << std::fixed << std::setprecision(5) << row.point.x << "; " << row.point.y << ')';
        std::cout << "| " << std::left << std::setw(28) << row.experiment << std::right
                  << " | " << std::setw(5) << row.outer << " | " << std::setw(6) << row.nf
                  << " | " << std::setw(5) << row.ng << " | " << std::left << std::setw(23)
                  << point.str() << std::right << " | " << std::setw(10) << std::fixed
                  << std::setprecision(5) << row.f << " | " << std::setw(10) << std::scientific
                  << std::setprecision(3) << row.violation << " |\n";
    }
    std::cout << "+------------------------------+-------+--------+-------+-------------------------+------------+------------+\n";

    std::filesystem::create_directories("outputs");
    std::ofstream csv("outputs/part2_summary.csv");
    csv << "experiment,outer,Nphi,Ngrad,x1,x2,F,violation\n";
    for (const auto& row : summary) {
        csv << row.experiment << ',' << row.outer << ',' << row.nf << ',' << row.ng << ','
            << row.point.x << ',' << row.point.y << ',' << row.f << ',' << row.violation << '\n';
    }
    std::cout << "Saved summary for plots: outputs/part2_summary.csv\n";
}
