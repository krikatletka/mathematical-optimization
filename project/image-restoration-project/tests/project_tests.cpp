#include "variant.h"
#include "image.h"
#include "rng.h"
#include "metrics.h"
#include "oracle.h"
#include "filters.h"
#include "lab1_svenn.h"
#include "lab2_search.h"
#include "lab3_nelder_mead.h"
#include "lab3_powell.h"
#include "lab4_gradient.h"
#include "lab5_constrained.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
int failures = 0;

void check(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void near(double actual, double expected, double tolerance, const char* message)
{
    check(std::fabs(actual - expected) <= tolerance, message);
}
}

int main()
{
    check(variant::STUDENT_ID == 8, "variant number");
    check(variant::SEED == 168081ULL, "variant seed");
    near(variant::NOISE_SIGMA, 0.1449, 1e-15, "noise sigma");

    Image clean = clean_image();
    check(clean.h == 128 && clean.w == 128, "clean image size");
    near(clean.at(60, 60), 0.4930, 5e-5, "clean central pixel");

    Rng reference(437327);
    const double normals[] = {1.533208, 0.235860, -1.375178, -0.453852, 0.323775, 0.199051};
    for (double expected : normals) near(reference.normal(), expected, 5e-7, "RNG reference sequence");

    Noisy data = add_noise(clean, variant::NOISE_SIGMA, variant::SEED);
    near(mse(clean, data.noisy), 0.01786611, 5e-9, "variant 8 noisy MSE");
    near(psnr(clean, data.noisy), 17.4797, 5e-5, "variant 8 noisy PSNR");

    Image black(2, 2, 0.0), white(2, 2, 1.0);
    near(mse(black, white), 1.0, 1e-15, "black-white MSE");

    Oracle oracle([](double x) { return (x - 2.0) * (x - 2.0); });
    near(oracle.value(0.0), 4.0, 1e-15, "oracle value");
    near(oracle.peek(5.0), 9.0, 1e-15, "oracle peek");
    check(oracle.n_f == 1, "peek does not count");

    Image constant(7, 7, 0.42);
    Image blurred = gaussian_blur(constant, 1.0);
    near(mse(constant, blurred), 0.0, 1e-20, "Gaussian preserves constants");
    Image bilateral = bilateral_filter(constant, 1.5, 0.2);
    near(mse(constant, bilateral), 0.0, 1e-20, "bilateral preserves constants");

    Oracle quadratic([](double x) { return (x - 2.0) * (x - 2.0); });
    auto bracket = svenn(quadratic, 1.0, 0.2, -10.0, 10.0);
    check(bracket.a <= 2.0 && 2.0 <= bracket.b, "Svenn brackets minimum");
    quadratic.reset();
    auto golden = golden_section(quadratic, bracket.a, bracket.b, 1e-5);
    near(golden.x, 2.0, 1e-4, "golden-section minimum");

    auto nm = nelder_mead([](Point2 p) {
        return (p.x - 1.0) * (p.x - 1.0) + (p.y + 2.0) * (p.y + 2.0);
    }, {0.0, 0.0}, 0.5, 1e-6, 500);
    near(nm.point.x, 1.0, 2e-3, "Nelder-Mead x");
    near(nm.point.y, -2.0, 2e-3, "Nelder-Mead y");

    auto pw = powell([](Point2 p) {
        return (p.x - 1.0) * (p.x - 1.0) + (p.y + 2.0) * (p.y + 2.0);
    }, {0.0, 0.0}, 1e-5, 20);
    near(pw.point.x, 1.0, 2e-3, "Powell x");
    near(pw.point.y, -2.0, 2e-3, "Powell y");

    Image raw(3, 3, 0.5);
    auto gd = gradient_descent(raw, 0.2, 1e-8, 100);
    check(gd.image.px.size() == raw.px.size(), "gradient result size");
    auto pg = projected_gradient(raw, 0.2, 1e-8, 100);
    check(max_box_violation(pg.image) == 0.0, "projection is feasible");

    if (failures == 0) std::cout << "All project tests passed.\n";
    return failures == 0 ? 0 : 1;
}
