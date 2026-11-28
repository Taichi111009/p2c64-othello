#include "p2c64_data.hpp"
#include "p2c64_io.hpp"
#include "p2c64_search.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace p2c64;

static constexpr int16_t RELABEL_SENTINEL = INT16_MIN; // -32768: use original final label
static constexpr int32_t RELABEL_MAX_ABS  = 8192;      // +/- 64 discs * 128

static void print_usage(const char* prog) {
    std::printf("Usage: %s --in <data.bin> --out <relabel.i16> --model <model.p2c> [options]\n", prog);
    std::printf("  --depth <d>            Fixed search depth (default: 6)\n");
    std::printf("  --ply-min <p>          Min ply to relabel inclusive (default: 12)\n");
    std::printf("  --ply-max <p>          Max ply to relabel inclusive (default: 49)\n");
    std::printf("  --mpc / --no-mpc       Enable/disable MPC during fixed-depth relabel (default: off)\n");
    std::printf("  --mpc-t <val>          MPC sigma multiplier (default: 1.50)\n");
    std::printf("  --mpc-cal <csv>        Optional MPC calibration CSV\n");
    std::printf("  --mo static|b4_hybrid  Move ordering mode (default: b4_hybrid)\n");
    std::printf("  --gate-mode <m>        safe_bnd|fail_hard|full|wide_safe_bnd (default: wide_safe_bnd)\n");
    std::printf("  --m25 <discs>          WACA Stage 1->2.5 margin (default: 2.5)\n");
    std::printf("  --m3 <discs>           WACA Stage 2.5->3 margin (default: 1.5)\n");
    std::printf("  --tt-guard <discs>     WIDE_SAFE_BND TT safety margin (default: 5.5)\n");
    std::printf("  --tt-mb <MB>           Per-thread TT size in MiB (default: 32)\n");
    std::printf("  --threads <T>          OpenMP thread count, 0=auto (default: 0)\n");
    std::printf("  --shard-size <N>       Records per flush/resume shard (default: 1000000)\n");
    std::printf("  --max-samples <N>      Optional cap on records processed, 0=all (default: 0)\n");
    std::printf("  --resume               Resume from last completed shard boundary in --out\n");
}

int main(int argc, char** argv) {
    std::string in_path, out_path, model_path, mpc_cal_path;
    std::string gate_mode_str = "wide_safe_bnd";
    std::string mo_str        = "b4_hybrid";
    int depth                 = 6;
    int ply_min               = 12;
    int ply_max               = 49;
    bool use_mpc              = false;
    double mpc_t              = 1.50;
    double m25_d              = 2.5;
    double m3_d               = 1.5;
    double tt_guard_d         = 5.5;
    size_t tt_mb              = 32;
    int threads               = 0;
    size_t shard_size         = 1000000;
    size_t max_samples        = 0;
    bool resume               = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--in" && i + 1 < argc)                in_path = argv[++i];
        else if (arg == "--out" && i + 1 < argc)          out_path = argv[++i];
        else if (arg == "--model" && i + 1 < argc)        model_path = argv[++i];
        else if (arg == "--depth" && i + 1 < argc)        depth = std::atoi(argv[++i]);
        else if (arg == "--ply-min" && i + 1 < argc)      ply_min = std::atoi(argv[++i]);
        else if (arg == "--ply-max" && i + 1 < argc)      ply_max = std::atoi(argv[++i]);
        else if (arg == "--mpc")                          use_mpc = true;
        else if (arg == "--no-mpc")                       use_mpc = false;
        else if (arg == "--mpc-t" && i + 1 < argc)        mpc_t = std::atof(argv[++i]);
        else if (arg == "--mpc-cal" && i + 1 < argc)      mpc_cal_path = argv[++i];
        else if (arg == "--mo" && i + 1 < argc)           mo_str = argv[++i];
        else if (arg == "--gate-mode" && i + 1 < argc)    gate_mode_str = argv[++i];
        else if (arg == "--m25" && i + 1 < argc)          m25_d = std::atof(argv[++i]);
        else if (arg == "--m3" && i + 1 < argc)           m3_d = std::atof(argv[++i]);
        else if (arg == "--tt-guard" && i + 1 < argc)     tt_guard_d = std::atof(argv[++i]);
        else if (arg == "--tt-mb" && i + 1 < argc)        tt_mb = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        else if (arg == "--threads" && i + 1 < argc)      threads = std::atoi(argv[++i]);
        else if (arg == "--shard-size" && i + 1 < argc)   shard_size = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        else if (arg == "--max-samples" && i + 1 < argc)  max_samples = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        else if (arg == "--resume")                       resume = true;
        else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "Unknown arg: %s\n", arg.c_str()); print_usage(argv[0]); return 1; }
    }

    if (in_path.empty() || out_path.empty() || model_path.empty()) {
        std::fprintf(stderr, "ERROR: --in, --out, and --model are required.\n");
        print_usage(argv[0]);
        return 1;
    }
    if (shard_size == 0) {
        std::fprintf(stderr, "ERROR: --shard-size must be > 0.\n");
        return 1;
    }

    GateMode gm = GateMode::WIDE_SAFE_BND;
    if (gate_mode_str == "safe_bnd")           gm = GateMode::SAFE_BND;
    else if (gate_mode_str == "fail_hard")     gm = GateMode::FAIL_HARD;
    else if (gate_mode_str == "full")          gm = GateMode::FULL;
    else if (gate_mode_str == "wide_safe_bnd") gm = GateMode::WIDE_SAFE_BND;
    else { std::fprintf(stderr, "ERROR: Invalid --gate-mode '%s'\n", gate_mode_str.c_str()); return 1; }

    MoveOrderMode mo = MoveOrderMode::B4_HYBRID;
    if (mo_str == "static")         mo = MoveOrderMode::STATIC;
    else if (mo_str == "b4_hybrid") mo = MoveOrderMode::B4_HYBRID;
    else { std::fprintf(stderr, "ERROR: Invalid --mo '%s'\n", mo_str.c_str()); return 1; }

    LoadedModel model;
    std::string err;
    if (!load_p2c_model(model_path, model, &err)) {
        std::fprintf(stderr, "ERROR loading model '%s': %s\n", model_path.c_str(), err.c_str());
        return 1;
    }

    std::ifstream ifs(in_path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        std::fprintf(stderr, "ERROR: Cannot open input '%s'\n", in_path.c_str());
        return 1;
    }
    const size_t file_bytes = static_cast<size_t>(ifs.tellg());
    size_t total_rec = file_bytes / sizeof(PackedSample);
    if (max_samples > 0 && max_samples < total_rec) {
        total_rec = max_samples;
    }
    if (total_rec == 0) {
        std::fprintf(stderr, "ERROR: No records in '%s'\n", in_path.c_str());
        return 1;
    }

    // Determine resume position aligned strictly to completed shard boundary
    size_t start_rec = 0;
    std::FILE* out_fp = nullptr;
    if (resume) {
        std::FILE* chk_fp = std::fopen(out_path.c_str(), "rb");
        if (chk_fp) {
            std::fseek(chk_fp, 0, SEEK_END);
            const long existing_bytes = std::ftell(chk_fp);
            std::fclose(chk_fp);
            if (existing_bytes > 0) {
                const size_t existing_recs = static_cast<size_t>(existing_bytes) / sizeof(int16_t);
                if (existing_recs >= total_rec) {
                    std::printf("[RESUME] Output '%s' already contains %zu >= %zu records. Nothing to do.\n",
                                out_path.c_str(), existing_recs, total_rec);
                    return 0;
                }
                const size_t completed_shards = existing_recs / shard_size;
                start_rec = std::min(total_rec, completed_shards * shard_size);
                out_fp = std::fopen(out_path.c_str(), "r+b");
                if (out_fp) {
                    std::fseek(out_fp, static_cast<long>(start_rec * sizeof(int16_t)), SEEK_SET);
                }
            }
        }
    }
    if (!out_fp) {
        start_rec = 0;
        out_fp = std::fopen(out_path.c_str(), "wb");
        if (!out_fp) {
            std::fprintf(stderr, "ERROR: Cannot open output '%s' for writing\n", out_path.c_str());
            return 1;
        }
    }

    int num_threads = 1;
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
    num_threads = omp_get_max_threads();
#else
    (void)threads;
#endif

    // Pre-allocate one independent PVSEngine per OpenMP thread (zero shared mutable state!)
    std::vector<std::unique_ptr<PVSEngine>> engines(static_cast<size_t>(num_threads));
    for (int t = 0; t < num_threads; ++t) {
        engines[static_cast<size_t>(t)] = std::make_unique<PVSEngine>(&model, tt_mb, 0);
        auto& eng = *engines[static_cast<size_t>(t)];
        eng.set_mpc(use_mpc, mpc_t);
        eng.set_move_order_mode(mo);
        eng.set_waca_gate(gm, m25_d, m3_d, tt_guard_d);
        if (!mpc_cal_path.empty()) {
            if (!eng.load_mpc_calibration(mpc_cal_path, &err)) {
                std::fprintf(stderr, "ERROR loading --mpc-cal: %s\n", err.c_str());
                std::fclose(out_fp);
                return 1;
            }
        }
    }

    const double scale_to_128 = 128.0 / static_cast<double>(model.score_scale);
    const bool exact_128 = (std::fabs(model.score_scale - 128.0f) < 1e-6f);

    std::printf("=== P2C-64 v4.0 OpenMP Fixed-Depth Dataset Relabeler ===\n");
    std::printf("Input   : %s (%zu records)\n", in_path.c_str(), total_rec);
    std::printf("Output  : %s (int16 sidecar, sentinel=%d, resume_from=%zu)\n",
                out_path.c_str(), static_cast<int>(RELABEL_SENTINEL), start_rec);
    std::printf("Model   : %s (%s)\n", model_path.c_str(), eval_mode_name(model.mode));
    std::printf("Config  : depth=%d | ply=[%d..%d] | MPC=%s(t=%.2f) | mo=%s | gate=%s(m25=%.2f,m3=%.2f,tt_guard=%.2f) | TT=%zu MiB x %d threads | shard=%zu\n",
                depth, ply_min, ply_max, use_mpc ? "ON" : "OFF", mpc_t, move_order_mode_name(mo),
                gate_mode_str.c_str(), m25_d, m3_d, tt_guard_d, tt_mb, num_threads, shard_size);
    std::fflush(stdout);

    std::vector<PackedSample> shard_in;
    std::vector<int16_t> shard_out;
    const auto t_start = std::chrono::steady_clock::now();
    size_t processed_this_run = 0;
    size_t relabeled_this_run = 0;

    ifs.seekg(static_cast<std::streamoff>(start_rec * sizeof(PackedSample)), std::ios::beg);

    for (size_t offset = start_rec; offset < total_rec; offset += shard_size) {
        const size_t cnt = std::min(shard_size, total_rec - offset);
        shard_in.resize(cnt);
        shard_out.assign(cnt, RELABEL_SENTINEL);

        if (!ifs.read(reinterpret_cast<char*>(shard_in.data()),
                      static_cast<std::streamsize>(cnt * sizeof(PackedSample)))) {
            std::fprintf(stderr, "ERROR: Failed reading shard at offset %zu\n", offset);
            std::fclose(out_fp);
            return 1;
        }

        size_t shard_relabeled = 0;
        #pragma omp parallel reduction(+:shard_relabeled)
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            PVSEngine& eng = *engines[static_cast<size_t>(tid)];

            #pragma omp for schedule(dynamic, 256)
            for (size_t i = 0; i < cnt; ++i) {
                const PackedSample& s = shard_in[i];
                const int ply = static_cast<int>(s.ply);
                if (ply < ply_min || ply > ply_max) {
                    shard_out[i] = RELABEL_SENTINEL;
                    continue;
                }
                const int32_t raw_sc = eng.eval_fixed_depth(s.black, s.white, static_cast<int>(s.stm),
                                                            depth, false, use_mpc);
                const int32_t sc128 = exact_128
                    ? raw_sc
                    : static_cast<int32_t>(std::lround(static_cast<double>(raw_sc) * scale_to_128));
                shard_out[i] = static_cast<int16_t>(std::clamp<int32_t>(sc128, -RELABEL_MAX_ABS, RELABEL_MAX_ABS));
                shard_relabeled++;
            }
        }

        const size_t written = std::fwrite(shard_out.data(), sizeof(int16_t), cnt, out_fp);
        if (written != cnt) {
            std::fprintf(stderr, "ERROR: Failed writing shard at offset %zu\n", offset);
            std::fclose(out_fp);
            return 1;
        }
        std::fflush(out_fp);

        processed_this_run += cnt;
        relabeled_this_run += shard_relabeled;
        const size_t done_total = offset + cnt;
        const double elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        const double pos_per_s = (elapsed_s > 1e-6) ? (static_cast<double>(processed_this_run) / elapsed_s) : 0.0;
        const double rel_per_s = (elapsed_s > 1e-6) ? (static_cast<double>(relabeled_this_run) / elapsed_s) : 0.0;
        const size_t rem = total_rec - done_total;
        const double eta_s = (pos_per_s > 1e-6) ? (static_cast<double>(rem) / pos_per_s) : 0.0;

        std::printf("[Shard %zu] %zu / %zu (%.1f%%) | relabeled=%zu | %.1f pos/s (%.1f rel/s) | elapsed=%.1fs | ETA=%.1fs\n",
                    (offset / shard_size) + 1, done_total, total_rec,
                    100.0 * static_cast<double>(done_total) / static_cast<double>(total_rec),
                    shard_relabeled, pos_per_s, rel_per_s, elapsed_s, eta_s);
        std::fflush(stdout);
    }

    std::fclose(out_fp);
    std::printf("[DONE] Relabeled %zu / %zu records into '%s'.\n", relabeled_this_run, total_rec, out_path.c_str());
    return 0;
}
