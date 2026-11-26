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

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() noexcept { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    int range(int lo, int hi) noexcept { return lo + static_cast<int>(next() % static_cast<uint64_t>(hi - lo + 1)); }
};

struct Opening {
    uint64_t black;
    uint64_t white;
    int      stm;
};

static std::vector<Opening> generate_openings(int num_pairs, int opening_plies, uint64_t seed) {
    Rng rng(seed);
    std::vector<Opening> ops;
    ops.reserve(static_cast<size_t>(num_pairs));

    while (static_cast<int>(ops.size()) < num_pairs) {
        uint64_t B = (1ULL << rc(3, 4)) | (1ULL << rc(4, 3));
        uint64_t W = (1ULL << rc(3, 3)) | (1ULL << rc(4, 4));
        int stm = 1;
        bool ok = true;
        for (int p = 0; p < opening_plies; ++p) {
            uint64_t P = (stm > 0) ? B : W;
            uint64_t O = (stm > 0) ? W : B;
            uint64_t mv = generate_moves(P, O);
            if (!mv) {
                if (generate_moves(O, P) == 0) { ok = false; break; }
                stm = -stm;
                continue;
            }
            int cnt = __builtin_popcountll(mv);
            int pick = rng.range(0, cnt - 1);
            int sq = 0; uint64_t m = mv;
            for (int k = 0; k <= pick; ++k) { sq = __builtin_ctzll(m); m &= m - 1; }
            uint64_t fl = calc_flips(sq, P, O);
            if (stm > 0) { B |= (1ULL << sq) | fl; W ^= fl; }
            else         { W |= (1ULL << sq) | fl; B ^= fl; }
            stm = -stm;
        }
        if (ok) ops.push_back({B, W, stm});
    }
    return ops;
}

struct SideStats {
    uint64_t total_nodes = 0;
    double   total_ms    = 0.0;
    uint64_t total_depth = 0;
    uint64_t move_count  = 0;
};

// Play single game from opening; returns final disc diff from Black's perspective
static int play_game(const Opening& op,
                     PVSEngine& eng_black, PVSEngine& eng_white,
                     SideStats& st_black,  SideStats& st_white,
                     SearchLimitMode mode, int64_t limit_val) {
    uint64_t B = op.black;
    uint64_t W = op.white;
    int stm = op.stm;
    int pass_streak = 0;

    eng_black.clear_tt();
    eng_white.clear_tt();

    while (pass_streak < 2) {
        uint64_t P = (stm > 0) ? B : W;
        uint64_t O = (stm > 0) ? W : B;
        uint64_t mv = generate_moves(P, O);
        if (!mv) {
            pass_streak++;
            stm = -stm;
            continue;
        }
        pass_streak = 0;

        PVSEngine& eng = (stm > 0) ? eng_black : eng_white;
        SideStats& st  = (stm > 0) ? st_black  : st_white;

        SearchResult sr = eng.search(B, W, stm, mode, limit_val);
        int sq = sr.best_move;
        if (sq < 0 || ((mv >> sq) & 1ULL) == 0ULL) {
            sq = __builtin_ctzll(mv);
        }
        st.total_nodes += sr.nodes;
        st.total_ms    += sr.elapsed_ms;
        st.total_depth += static_cast<uint64_t>(sr.completed_depth);
        st.move_count++;

        uint64_t fl = calc_flips(sq, P, O);
        if (stm > 0) { B |= (1ULL << sq) | fl; W ^= fl; }
        else         { W |= (1ULL << sq) | fl; B ^= fl; }
        stm = -stm;
    }

    int bc = __builtin_popcountll(B);
    int wc = __builtin_popcountll(W);
    int em = 64 - bc - wc;
    if (bc > wc) return bc - wc + em;
    if (bc < wc) return bc - wc - em;
    return 0;
}

static double prob_to_elo(double p) {
    p = std::clamp(p, 1e-6, 1.0 - 1e-6);
    return -400.0 * std::log10(1.0 / p - 1.0);
}

static void print_usage(const char* prog) {
    std::printf("Usage: %s --p1 <model1.p2c> --p2 <model2.p2c> [options]\n", prog);
    std::printf("  --mode depth|nodes|time  Search limit mode (default: depth)\n");
    std::printf("  --limit <val>            Depth plies / node budget / ms per move (default: 4)\n");
    std::printf("  --pairs <N>              Number of paired openings (2*N games, default: 20)\n");
    std::printf("  --opening-plies <P>      Random opening plies (default: 8)\n");
    std::printf("  --endgame-empties <E>    Exact endgame solver threshold (default: 10)\n");
    std::printf("  --tt-mb <MB>             TT size per player in MiB (default: 64)\n");
    std::printf("  --seed <S>               Opening RNG seed (default: 20261125)\n");
    std::printf("  --csv <path>             Append summary row to CSV file\n");
}

int main(int argc, char** argv) {
    std::string p1_path, p2_path, csv_path;
    std::string mode_str = "depth";
    int64_t limit_val    = 4;
    int num_pairs        = 20;
    int opening_plies    = 8;
    int endgame_empties  = 10;
    size_t tt_mb         = 64;
    uint64_t seed        = 20261125ULL;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--p1" && i + 1 < argc)                   p1_path = argv[++i];
        else if (arg == "--p2" && i + 1 < argc)              p2_path = argv[++i];
        else if (arg == "--mode" && i + 1 < argc)            mode_str = argv[++i];
        else if (arg == "--limit" && i + 1 < argc)           limit_val = std::strtoll(argv[++i], nullptr, 10);
        else if (arg == "--pairs" && i + 1 < argc)           num_pairs = std::atoi(argv[++i]);
        else if (arg == "--opening-plies" && i + 1 < argc)   opening_plies = std::atoi(argv[++i]);
        else if (arg == "--endgame-empties" && i + 1 < argc) endgame_empties = std::atoi(argv[++i]);
        else if (arg == "--tt-mb" && i + 1 < argc)           tt_mb = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        else if (arg == "--seed" && i + 1 < argc)            seed = static_cast<uint64_t>(std::strtoull(argv[++i], nullptr, 10));
        else if (arg == "--csv" && i + 1 < argc)             csv_path = argv[++i];
        else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "Unknown arg: %s\n", arg.c_str()); print_usage(argv[0]); return 1; }
    }

    if (p1_path.empty() || p2_path.empty()) {
        std::fprintf(stderr, "ERROR: Both --p1 and --p2 .p2c paths are required.\n");
        print_usage(argv[0]);
        return 1;
    }

    SearchLimitMode mode = SearchLimitMode::DEPTH;
    if (mode_str == "depth")      mode = SearchLimitMode::DEPTH;
    else if (mode_str == "nodes") mode = SearchLimitMode::NODES;
    else if (mode_str == "time")  mode = SearchLimitMode::TIME;
    else { std::fprintf(stderr, "ERROR: Invalid --mode '%s'\n", mode_str.c_str()); return 1; }

    LoadedModel m1, m2;
    std::string err;
    if (!load_p2c_model(p1_path, m1, &err)) { std::fprintf(stderr, "ERROR loading p1: %s\n", err.c_str()); return 1; }
    if (!load_p2c_model(p2_path, m2, &err)) { std::fprintf(stderr, "ERROR loading p2: %s\n", err.c_str()); return 1; }

    PVSEngine eng1(&m1, tt_mb, endgame_empties);
    PVSEngine eng2(&m2, tt_mb, endgame_empties);

    auto openings = generate_openings(num_pairs, opening_plies, seed);

    int wins = 0, draws = 0, losses = 0;
    int penta[5] = {0, 0, 0, 0, 0}; // 0:LL, 1:LD, 2:WL/DD, 3:WD, 4:WW
    int64_t total_disc_diff = 0;
    SideStats st1, st2;

    std::printf("=== P2C-64 Pure-Scalar PVS Paired Match Engine ===\n");
    std::printf("P1 (%s): %s\n", eval_mode_name(m1.mode), p1_path.c_str());
    std::printf("P2 (%s): %s\n", eval_mode_name(m2.mode), p2_path.c_str());
    std::printf("Config: mode=%s limit=%lld | pairs=%d (%d games) | open=%d ply | endgame=%d empties | TT=%zu MiB\n",
                mode_str.c_str(), static_cast<long long>(limit_val), num_pairs, num_pairs * 2,
                opening_plies, endgame_empties, tt_mb);

    for (int i = 0; i < num_pairs; ++i) {
        // Game 1: P1 = Black, P2 = White
        int d1 = play_game(openings[static_cast<size_t>(i)], eng1, eng2, st1, st2, mode, limit_val);
        int r1 = (d1 > 0) ? 2 : ((d1 == 0) ? 1 : 0);
        // Game 2: P2 = Black, P1 = White
        int d2_black = play_game(openings[static_cast<size_t>(i)], eng2, eng1, st2, st1, mode, limit_val);
        int d2 = -d2_black; // from P1's perspective
        int r2 = (d2 > 0) ? 2 : ((d2 == 0) ? 1 : 0);

        if (r1 == 2) wins++; else if (r1 == 1) draws++; else losses++;
        if (r2 == 2) wins++; else if (r2 == 1) draws++; else losses++;
        penta[r1 + r2]++;
        total_disc_diff += (d1 + d2);
    }

    const int total_games = num_pairs * 2;
    const double win_rate = (wins + 0.5 * draws) / static_cast<double>(total_games);
    const double avg_diff = static_cast<double>(total_disc_diff) / static_cast<double>(total_games);

    // Pentanomial Elo and 95% confidence interval
    double mean_p = 0.0;
    for (int k = 0; k < 5; ++k) mean_p += (k * 0.25) * penta[k];
    mean_p /= static_cast<double>(num_pairs);

    double var_sum = 0.0;
    for (int k = 0; k < 5; ++k) {
        double dx = (k * 0.25) - mean_p;
        var_sum += penta[k] * dx * dx;
    }
    double se_p = (num_pairs > 1) ? std::sqrt(var_sum / (static_cast<double>(num_pairs) * (num_pairs - 1))) : 0.0;
    double elo_mid = prob_to_elo(mean_p);
    double elo_lo  = prob_to_elo(mean_p - 1.96 * se_p);
    double elo_hi  = prob_to_elo(mean_p + 1.96 * se_p);
    double elo_err = 0.5 * (elo_hi - elo_lo);

    double nps1 = (st1.total_ms > 0.0) ? (static_cast<double>(st1.total_nodes) / (st1.total_ms * 1e-3)) : 0.0;
    double nps2 = (st2.total_ms > 0.0) ? (static_cast<double>(st2.total_nodes) / (st2.total_ms * 1e-3)) : 0.0;
    double avg_d1 = st1.move_count ? static_cast<double>(st1.total_depth) / st1.move_count : 0.0;
    double avg_d2 = st2.move_count ? static_cast<double>(st2.total_depth) / st2.move_count : 0.0;

    std::printf("\n--- Match Summary (P1 vs P2) ---\n");
    std::printf("Result (W-D-L) : %d - %d - %d (WinRate: %.2f%%)\n", wins, draws, losses, win_rate * 100.0);
    std::printf("Pentanomial    : [LL=%d, LD=%d, WL/DD=%d, WD=%d, WW=%d]\n",
                penta[0], penta[1], penta[2], penta[3], penta[4]);
    std::printf("Elo Diff (95%%) : %+.1f +/- %.1f [%.1f, %.1f]\n", elo_mid, elo_err, elo_lo, elo_hi);
    std::printf("Avg Disc Diff  : %+.2f discs/game\n", avg_diff);
    std::printf("P1 Perf        : %.2f kNPS | avg depth %.2f | %llu nodes in %.2f s\n",
                nps1 * 1e-3, avg_d1, static_cast<unsigned long long>(st1.total_nodes), st1.total_ms * 1e-3);
    std::printf("P2 Perf        : %.2f kNPS | avg depth %.2f | %llu nodes in %.2f s\n",
                nps2 * 1e-3, avg_d2, static_cast<unsigned long long>(st2.total_nodes), st2.total_ms * 1e-3);

    if (!csv_path.empty()) {
        bool exists = std::ifstream(csv_path).good();
        std::ofstream ofs(csv_path, std::ios::app);
        if (ofs) {
            if (!exists) {
                ofs << "p1_mode,p2_mode,mode,limit,games,wins,draws,losses,win_rate,elo,elo_ci95,avg_disc_diff,p1_knps,p2_knps,p1_depth,p2_depth\n";
            }
            ofs << eval_mode_name(m1.mode) << "," << eval_mode_name(m2.mode) << ","
                << mode_str << "," << limit_val << "," << total_games << ","
                << wins << "," << draws << "," << losses << ","
                << win_rate << "," << elo_mid << "," << elo_err << "," << avg_diff << ","
                << (nps1 * 1e-3) << "," << (nps2 * 1e-3) << "," << avg_d1 << "," << avg_d2 << "\n";
        }
    }
    return 0;
}
