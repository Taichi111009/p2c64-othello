#include "p2c64_data.hpp"
#include "p2c64_io.hpp"
#include "p2c64_search.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace p2c64;

struct RegStats {
    double slope = 0.0;
    double intercept = 0.0;
    double sigma = 0.0;
    double r2 = 0.0;
    size_t n = 0;
};

static RegStats fit_linear_regression(const std::vector<double>& x, const std::vector<double>& y) {
    RegStats st{};
    st.n = x.size();
    if (st.n < 2) return st;

    double sx = 0.0, sy = 0.0;
    for (size_t i = 0; i < st.n; ++i) { sx += x[i]; sy += y[i]; }
    double mx = sx / st.n;
    double my = sy / st.n;

    double sxx = 0.0, sxy = 0.0, syy = 0.0;
    for (size_t i = 0; i < st.n; ++i) {
        double dx = x[i] - mx;
        double dy = y[i] - my;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    st.slope     = (sxx > 1e-12) ? (sxy / sxx) : 0.0;
    st.intercept = my - st.slope * mx;

    double sse = 0.0;
    for (size_t i = 0; i < st.n; ++i) {
        double res = y[i] - (st.slope * x[i] + st.intercept);
        sse += res * res;
    }
    st.sigma = std::sqrt(sse / static_cast<double>(st.n));
    st.r2    = (syy > 1e-12) ? (1.0 - sse / syy) : 0.0;
    return st;
}

static void print_usage(const char* prog) {
    std::printf("Usage: %s --data <val.bin> --models <m1.p2c> [m2.p2c ...] [options]\n", prog);
    std::printf("  --num-pos <N>    Max unseen positions to sample (default: 2000)\n");
    std::printf("  --shallow <d>    Shallow search depth (default: 2)\n");
    std::printf("  --deep <d>       Deep search depth (default: 6)\n");
    std::printf("  --tt-mb <MB>     TT size in MiB (default: 64)\n");
    std::printf("  --m25 <discs>    WACA-G2 m25 gate margin (default: 2.5)\n");
    std::printf("  --m3 <discs>     WACA-G2 m3 gate margin (default: 1.5)\n");
    std::printf("  --tt-guard <d>   WIDE_SAFE_BND TT safety margin (default: 4.5)\n");
    std::printf("  --gate-mode <m>  safe_bnd|fail_hard|full|wide_safe_bnd (default: safe_bnd)\n");
    std::printf("  --csv <path>     Write MPC sigma comparison CSV\n");
}

int main(int argc, char** argv) {
    std::string data_path, csv_path, gate_mode_str = "safe_bnd";
    std::vector<std::string> model_paths;
    size_t num_pos    = 2000;
    int d_shallow     = 2;
    int d_deep        = 6;
    size_t tt_mb      = 64;
    double m25_d      = 2.5;
    double m3_d       = 1.5;
    double tt_guard_d = 4.5;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc)           data_path = argv[++i];
        else if (arg == "--num-pos" && i + 1 < argc)   num_pos = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        else if (arg == "--shallow" && i + 1 < argc)   d_shallow = std::atoi(argv[++i]);
        else if (arg == "--deep" && i + 1 < argc)      d_deep = std::atoi(argv[++i]);
        else if (arg == "--tt-mb" && i + 1 < argc)     tt_mb = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        else if (arg == "--m25" && i + 1 < argc)       m25_d = std::atof(argv[++i]);
        else if (arg == "--m3" && i + 1 < argc)        m3_d = std::atof(argv[++i]);
        else if (arg == "--tt-guard" && i + 1 < argc)  tt_guard_d = std::atof(argv[++i]);
        else if (arg == "--gate-mode" && i + 1 < argc) gate_mode_str = argv[++i];
        else if (arg == "--csv" && i + 1 < argc)       csv_path = argv[++i];
        else if (arg == "--models") {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                model_paths.emplace_back(argv[++i]);
            }
        } else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "Unknown arg: %s\n", arg.c_str()); print_usage(argv[0]); return 1; }
    }

    if (data_path.empty() || model_paths.empty()) {
        std::fprintf(stderr, "ERROR: --data and --models are required.\n");
        print_usage(argv[0]);
        return 1;
    }

    GateMode gm = GateMode::SAFE_BND;
    if (gate_mode_str == "fail_hard") gm = GateMode::FAIL_HARD;
    else if (gate_mode_str == "full") gm = GateMode::FULL;
    else if (gate_mode_str == "wide_safe_bnd") gm = GateMode::WIDE_SAFE_BND;

    std::ifstream ifs(data_path, std::ios::binary | std::ios::ate);
    if (!ifs) { std::fprintf(stderr, "ERROR: Cannot open '%s'\n", data_path.c_str()); return 1; }
    size_t file_bytes = static_cast<size_t>(ifs.tellg());
    size_t total_rec  = file_bytes / sizeof(PackedSample);
    ifs.seekg(0, std::ios::beg);

    std::vector<PackedSample> all(total_rec);
    ifs.read(reinterpret_cast<char*>(all.data()), static_cast<std::streamsize>(total_rec * sizeof(PackedSample)));

    std::vector<PackedSample> samples;
    samples.reserve(std::min(num_pos, total_rec));
    if (total_rec <= num_pos) {
        samples = all;
    } else {
        double stride = static_cast<double>(total_rec) / static_cast<double>(num_pos);
        for (size_t k = 0; k < num_pos; ++k) {
            size_t idx = std::min(total_rec - 1, static_cast<size_t>(k * stride));
            samples.push_back(all[idx]);
        }
    }

    std::printf("=== P2C-64 Multi-ProbCut (MPC) Sigma Measurer (depth %d -> %d) ===\n", d_shallow, d_deep);
    std::printf("Dataset : %s (%zu sampled positions out of %zu)\n\n", data_path.c_str(), samples.size(), total_rec);
    std::printf("%-28s | %8s | %8s | %8s | %8s | %6s | %6s | %6s\n",
                "Model / Pair", "All sig", "0-19", "20-39", "40-59", "Slope", "Bias", "R^2");
    std::printf("-----------------------------+----------+----------+----------+----------+--------+--------+-------\n");

    std::ofstream csv_ofs;
    if (!csv_path.empty()) {
        csv_ofs.open(csv_path, std::ios::trunc);
        if (csv_ofs) {
            csv_ofs << "model_path,mode,shallow,deep,n_pos,sigma_all,sigma_0_19,sigma_20_39,sigma_40_59,slope,intercept,r2\n";
        }
    }

    auto emit_row = [&](const std::string& mp, const std::string& label,
                        const std::vector<double>& ax, const std::vector<double>& ay,
                        const std::vector<double> bx[3], const std::vector<double> by[3]) {
        RegStats r_all = fit_linear_regression(ax, ay);
        RegStats r_b0  = fit_linear_regression(bx[0], by[0]);
        RegStats r_b1  = fit_linear_regression(bx[1], by[1]);
        RegStats r_b2  = fit_linear_regression(bx[2], by[2]);
        std::printf("%-28s | %8.3f | %8.3f | %8.3f | %8.3f | %6.3f | %+6.3f | %6.3f\n",
                    label.c_str(), r_all.sigma, r_b0.sigma, r_b1.sigma, r_b2.sigma,
                    r_all.slope, r_all.intercept, r_all.r2);
        if (csv_ofs) {
            csv_ofs << mp << "," << label << "," << d_shallow << "," << d_deep << ","
                    << r_all.n << "," << r_all.sigma << "," << r_b0.sigma << "," << r_b1.sigma << ","
                    << r_b2.sigma << "," << r_all.slope << "," << r_all.intercept << "," << r_all.r2 << "\n";
        }
    };

    for (const auto& mp : model_paths) {
        LoadedModel m;
        std::string err;
        if (!load_p2c_model(mp, m, &err)) {
            std::fprintf(stderr, "ERROR loading '%s': %s\n", mp.c_str(), err.c_str());
            return 1;
        }
        PVSEngine eng(&m, tt_mb, 0);
        eng.set_waca_gate(gm, m25_d, m3_d, tt_guard_d);
        const double inv_scale = 1.0 / static_cast<double>(m.score_scale);

        std::vector<double> ax, ay, ax_v1;
        std::vector<double> bx[3], by[3], bx_v1[3];
        ax.reserve(samples.size()); ay.reserve(samples.size());
        if (m.is_any_waca()) ax_v1.reserve(samples.size());

        for (const auto& s : samples) {
            if (s.ply > 59) continue;
            int empties = 60 - static_cast<int>(s.ply);
            if (empties <= d_deep) continue;

            eng.clear_tt();
            double vs = static_cast<double>(eng.eval_fixed_depth(s.black, s.white, s.stm, d_shallow, false)) * inv_scale;
            eng.clear_tt();
            double vd = static_cast<double>(eng.eval_fixed_depth(s.black, s.white, s.stm, d_deep, false)) * inv_scale;

            int bkt = (s.ply < 20) ? 0 : ((s.ply < 40) ? 1 : 2);
            ax.push_back(vs);
            ay.push_back(vd);
            bx[bkt].push_back(vs);
            by[bkt].push_back(vd);

            if (m.is_any_waca()) {
                eng.clear_tt();
                double vs_v1 = static_cast<double>(eng.eval_fixed_depth(s.black, s.white, s.stm, d_shallow, true)) * inv_scale;
                ax_v1.push_back(vs_v1);
                bx_v1[bkt].push_back(vs_v1);
            }
        }

        emit_row(mp, eval_mode_name(m.mode), ax, ay, bx, by);
        if (m.is_any_waca()) {
            std::string pure_lbl = std::string(eval_mode_name(m.mode)) + "[V1->Full]";
            emit_row(mp, pure_lbl, ax_v1, ay, bx_v1, by);
        }
    }
    return 0;
}
