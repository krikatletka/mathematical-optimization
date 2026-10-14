#include "filters.h"
#include "lab1_svenn.h"
#include "lab2_search.h"
#include "lab3_nelder_mead.h"
#include "lab3_powell.h"
#include "lab4_gradient.h"
#include "lab5_constrained.h"
#include "metrics.h"
#include "variant.h"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

struct ProjectData {
    Image clean;
    Image raw;
    Image noisy;
};

ProjectData prepare(const fs::path& output)
{
    fs::create_directories(output);
    Image clean = clean_image();
    Noisy noisy = add_noise(clean, variant::NOISE_SIGMA, variant::SEED);
    write_pgm(clean, output / "00_clean.pgm");
    write_pgm(noisy.noisy, output / "01_noisy.pgm");
    std::cout << "PART 0 - PREPARATION\n"
              << "Variant = " << variant::STUDENT_ID << "\nSeed = " << variant::SEED
              << "\nNoise sigma = " << variant::NOISE_SIGMA
              << "\nMSE = " << std::setprecision(8) << mse(clean, noisy.noisy)
              << "\nPSNR = " << std::setprecision(4) << psnr(clean, noisy.noisy) << " dB"
              << "\nRaw pixels outside [0,1] = " << count_outside_box(noisy.raw) << "\n";
    const bool pass = std::fabs(mse(clean, noisy.noisy) - 0.01786611) <= 5e-9 && std::fabs(psnr(clean, noisy.noisy) - 17.4797) <= 5e-5;
    std::cout << "CONTROL CHECK: " << (pass ? "PASSED" : "FAILED") << "\n\n";
    return {std::move(clean), std::move(noisy.raw), std::move(noisy.noisy)};
}

SvennResult run_lab1(const ProjectData& data)
{
    Oracle oracle([&](double sigma) { return mse(gaussian_blur(data.noisy, sigma), data.clean); });
    SvennResult result = svenn(oracle, variant::X0, variant::DELTA0);
    std::cout << "LAB 1 - SVENN\nInterval = [" << result.a << "; " << result.b << "]\nNf = " << result.evaluations << "\n\n";
    return result;
}

SearchResult run_lab2(const ProjectData& data, const SvennResult& bracket, const fs::path& output)
{
    auto objective = [&](double sigma) { return mse(gaussian_blur(data.noisy, sigma), data.clean); };
    Oracle od(objective); SearchResult d = dichotomy(od, bracket.a, bracket.b, variant::EPS);
    Oracle og(objective); SearchResult g = golden_section(og, bracket.a, bracket.b, variant::EPS);
    Image restored = gaussian_blur(data.noisy, g.x); write_pgm(restored, output / "02_gaussian.pgm");
    std::cout << "LAB 2 - ONE-DIMENSIONAL SEARCH\n"
              << "Dichotomy: sigma* = " << d.x << ", iterations = " << d.iterations << ", Nf = " << d.evaluations << '\n'
              << "Golden:    sigma* = " << g.x << ", iterations = " << g.iterations << ", Nf = " << g.evaluations << '\n'
              << "Restored PSNR = " << psnr(data.clean, restored) << " dB\n\n";
    return g;
}

void run_lab3(const ProjectData& data, const fs::path& output)
{
    auto objective = [&](Point2 p) {
        if (p.x <= 0.05 || p.y <= 0.005 || p.x > 8.0 || p.y > 2.0) return 1e9;
        return mse(bilateral_filter(data.noisy, p.x, p.y), data.clean);
    };
    Point2 start{1.3383, 0.1645};
    NMResult nm = nelder_mead(objective, start, variant::SIMPLEX_EDGE, variant::EPS, 200);
    PowellResult pw = powell(objective, start, variant::EPS, 12);
    Image restored = bilateral_filter(data.noisy, nm.point.x, nm.point.y); write_pgm(restored, output / "03_bilateral.pgm");
    std::cout << "LAB 3 - NELDER-MEAD AND POWELL\n"
              << "Nelder-Mead: (" << nm.point.x << ", " << nm.point.y << "), iterations = " << nm.iterations << ", Nf = " << nm.evaluations << '\n'
              << "Powell:      (" << pw.point.x << ", " << pw.point.y << "), cycles = " << pw.cycles << ", Nf = " << pw.evaluations << '\n'
              << "Restored PSNR = " << psnr(data.clean, restored) << " dB\n\n";
}

GradientResult run_lab4(const ProjectData& data, const fs::path& output)
{
    constexpr double lambda = 0.477352;
    GradientResult gd = gradient_descent(data.noisy, lambda, 1e-6, 300);
    GradientResult hb = heavy_ball(data.noisy, lambda, 0.1399, 1e-6, 300);
    GradientResult ng = nesterov(data.noisy, lambda, 1e-6, 300);
    write_pgm(ng.image, output / "04_gradient.pgm");
    std::cout << "LAB 4 - GRADIENT METHODS\nGradient iterations = " << gd.iterations
              << "\nHeavy-ball iterations = " << hb.iterations
              << "\nNesterov iterations = " << ng.iterations
              << "\nNesterov PSNR = " << psnr(data.clean, ng.image) << " dB\n\n";
    return ng;
}

void run_lab5(const ProjectData& data, const fs::path& output)
{
    constexpr double lambda = 0.477352;
    GradientResult projected = projected_gradient(data.raw, lambda, 1e-6, 300);
    PenaltyResult penalty = penalty_method(data.raw, lambda);
    write_pgm(projected.image, output / "05_constrained.pgm");
    std::cout << "LAB 5 - CONSTRAINED OPTIMIZATION\nProjected iterations = " << projected.iterations
              << ", calls = " << projected.n_f + projected.n_grad
              << "\nProjected violation = " << max_box_violation(projected.image)
              << "\nPenalty outer iterations = " << penalty.outer_iterations
              << ", calls = " << penalty.n_f + penalty.n_grad
              << "\nPenalty violation = " << max_box_violation(penalty.image)
              << "\nProjected PSNR = " << psnr(data.clean, projected.image) << " dB\n\n";
}

int main(int argc, char** argv)
{
    try {
        int part = 0;
        bool all = false;
        if (argc == 3 && std::string(argv[1]) == "--part") part = std::stoi(argv[2]);
        else if (argc == 2 && std::string(argv[1]) == "--all") all = true;
        else if (argc != 1) throw std::invalid_argument("usage: image_restoration [--part 0..5 | --all]");
        if (part < 0 || part > 5) throw std::invalid_argument("part must be from 0 to 5");
        const fs::path output = "outputs";
        ProjectData data = prepare(output);
        if (!all && part == 0) return 0;
        SvennResult bracket = run_lab1(data);
        if (!all && part == 1) return 0;
        run_lab2(data, bracket, output);
        if (!all && part == 2) return 0;
        run_lab3(data, output);
        if (!all && part == 3) return 0;
        run_lab4(data, output);
        if (!all && part == 4) return 0;
        run_lab5(data, output);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
