#include "p2c64_eval.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace p2c64;

// 256 MiB Transposition Table (16B/entry -> 16,777,216 entries)
struct alignas(16) TTEntry {
    uint64_t key;
    int64_t  val;
};
static constexpr size_t TT_SIZE = (256ULL * 1024 * 1024) / sizeof(TTEntry);
static constexpr size_t TT_MASK = TT_SIZE - 1;

struct RecordedLeaf {
    uint16_t idx[NUM_PATTERNS];
    uint64_t tt_hash;
    int8_t   stm;
    uint8_t  ply;
};

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() noexcept { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
};

// Record realistic midgame (ply 20..40) DFS Alpha-Beta search leaves
// Depth-8..10 locality is reproduced by traversing a bounded-branching (top-3/4 moves) DFS tree
// of depth 8..10 from real midgame roots (ply 20..32 -> leaves at ply 28..40), yielding 200,000 leaves!
static void dfs_collect(const PatternTables& pt,
                        uint64_t B, uint64_t W, int stm, int ply, int rem_depth,
                        uint64_t hash, const uint16_t idx[NUM_PATTERNS],
                        std::vector<RecordedLeaf>& out, size_t target_count) {
    if (out.size() >= target_count) return;
    if (rem_depth == 0 || ply >= NUM_PLIES) {
        RecordedLeaf lf;
        std::memcpy(lf.idx, idx, sizeof(lf.idx));
        lf.tt_hash = hash;
        lf.stm     = static_cast<int8_t>(stm);
        lf.ply     = static_cast<uint8_t>(std::clamp(ply, 0, NUM_PLIES - 1));
        out.push_back(lf);
        return;
    }

    uint64_t P  = (stm > 0) ? B : W;
    uint64_t O  = (stm > 0) ? W : B;
    uint64_t mv = generate_moves(P, O);
    if (!mv) {
        if (generate_moves(O, P) != 0) {
            dfs_collect(pt, B, W, -stm, ply, rem_depth - 1, hash ^ 0x55aa55aa55aa55aaULL, idx, out, target_count);
        }
        return;
    }

    // Mimic Alpha-Beta PV/Cut branching factor (~3 moves expanded per internal node at depth 8..10)
    int expanded = 0;
    const int max_branch = (rem_depth >= 5) ? 4 : 3;
    uint16_t child_idx[NUM_PATTERNS];
    while (mv && expanded < max_branch && out.size() < target_count) {
        int sq = __builtin_ctzll(mv);
        mv &= mv - 1;
        uint64_t flips = calc_flips(sq, P, O);
        uint64_t nB = (stm > 0) ? (B | (1ULL << sq) | flips) : (B ^ flips);
        uint64_t nW = (stm > 0) ? (W ^ flips) : (W | (1ULL << sq) | flips);
        uint64_t nh = (hash * 0x9e3779b97f4a7c15ULL) ^ (static_cast<uint64_t>(sq) * 0xbf58476d1ce4e5b9ULL) ^ flips;
        pt.update_indices_move(child_idx, idx, sq, flips, stm);
        dfs_collect(pt, nB, nW, -stm, ply + 1, rem_depth - 1, nh, child_idx, out, target_count);
        expanded++;
    }
}

static std::vector<RecordedLeaf> build_dfs_replay_trace(size_t total_leaves) {
    const auto& pt = get_pattern_tables();
    Rng rng(0xdecafbad12345678ULL);
    std::vector<RecordedLeaf> leaves;
    leaves.reserve(total_leaves);

    constexpr int NUM_ROOTS = 10;
    const size_t per_root = total_leaves / NUM_ROOTS;

    for (int r = 0; r < NUM_ROOTS; ++r) {
        uint64_t B = (1ULL << rc(3, 4)) | (1ULL << rc(4, 3));
        uint64_t W = (1ULL << rc(3, 3)) | (1ULL << rc(4, 4));
        int stm = 1;
        int root_ply = 20 + (r % 11); // roots at ply 20..30 -> depth 9 leaves at ply 29..39
        int cur_ply = 0;
        while (cur_ply < root_ply) {
            uint64_t P = (stm > 0) ? B : W;
            uint64_t O = (stm > 0) ? W : B;
            uint64_t mv = generate_moves(P, O);
            if (!mv) { stm = -stm; continue; }
            int cnt = __builtin_popcountll(mv);
            int pick = static_cast<int>(rng.next() % static_cast<uint64_t>(cnt));
            int sq = 0; uint64_t m = mv;
            for (int k = 0; k <= pick; ++k) { sq = __builtin_ctzll(m); m &= m - 1; }
            uint64_t fl = calc_flips(sq, P, O);
            if (stm > 0) { B |= (1ULL << sq) | fl; W ^= fl; }
            else         { W |= (1ULL << sq) | fl; B ^= fl; }
            stm = -stm;
            cur_ply++;
        }
        uint16_t root_idx[NUM_PATTERNS];
        pt.extract_indices(B, W, root_idx);
        const int depth = 9; // depth-9 DFS tree
        dfs_collect(pt, B, W, stm, cur_ply, depth, rng.next(), root_idx, leaves, leaves.size() + per_root);
    }
    return leaves;
}

template<typename EvalFn>
static void bench_one(const char* name,
                      const std::vector<RecordedLeaf>& leaves,
                      TTEntry* tt,
                      EvalFn eval_fn,
                      double& ref_eval_ns,
                      double& ref_tt_ns) {
    constexpr int PASSES_EVAL = 6;
    constexpr int PASSES_TT   = 4;
    volatile int64_t sink = 0;

    // Warmup 1 pass
    for (const auto& lf : leaves) sink += eval_fn(lf.idx, lf.stm, lf.ply);

    // 1) Pure Eval-only (realistic DFS search-tree locality)
    auto t0 = std::chrono::steady_clock::now();
    for (int pass = 0; pass < PASSES_EVAL; ++pass) {
        for (const auto& lf : leaves) {
            sink += eval_fn(lf.idx, lf.stm, lf.ply);
        }
    }
    double eval_ns = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                   * 1e9 / (static_cast<double>(PASSES_EVAL) * static_cast<double>(leaves.size()));

    // 2) Eval + Active 256 MiB TT Contention (random read + write per leaf)
    auto t1 = std::chrono::steady_clock::now();
    for (int pass = 0; pass < PASSES_TT; ++pass) {
        for (const auto& lf : leaves) {
            TTEntry& te = tt[lf.tt_hash & TT_MASK];
            const int64_t prev = te.val;
            const int32_t sc   = eval_fn(lf.idx, lf.stm, lf.ply);
            te.key = lf.tt_hash;
            te.val = static_cast<int64_t>(sc) + (prev & 1LL);
            sink += sc;
        }
    }
    double tt_ns = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count()
                 * 1e9 / (static_cast<double>(PASSES_TT) * static_cast<double>(leaves.size()));

    if (ref_eval_ns == 0.0) ref_eval_ns = eval_ns;
    if (ref_tt_ns   == 0.0) ref_tt_ns   = tt_ns;

    const double eval_mnps = 1000.0 / eval_ns;
    const double tt_mnps   = 1000.0 / tt_ns;
    std::printf("%-28s | %6.1f ns (%5.2f M/s, %+5.1f%%) | %6.1f ns (%5.2f M/s, %+5.1f%%) | chk:%02llx\n",
                name,
                eval_ns, eval_mnps, (eval_ns / ref_eval_ns - 1.0) * 100.0,
                tt_ns,   tt_mnps,   (tt_ns   / ref_tt_ns   - 1.0) * 100.0,
                static_cast<unsigned long long>(sink & 0xFF));
}

int main() {
    const auto& pt = get_pattern_tables();
    Rng rng(0x13579bdf2468ace0ULL);

    // Initialize RawNTuple64 (60 plies x 2B w + 60 plies x 1B corner-4 tempo = 48.05 MiB)
    RawNTuple64Weights raw;
    for (int ply = 0; ply < NUM_PLIES; ++ply) {
        for (int16_t& v : raw.w_ply[ply])        v = static_cast<int16_t>((rng.next() & 511U) - 256);
        for (int8_t&  v : raw.corner_tempo[ply]) v = static_cast<int8_t>((rng.next() & 63U) - 32);
    }

    // Initialize P2C64Weights v1.1 Asymmetric Layout (7.01 MiB Key-14 + 36.41 MiB NonKey-50 + 96.1 KiB Tempo + 30 KiB MacroLUT)
    P2C64Weights w;
    for (int b = 0; b < NUM_BLOCKS; ++b) {
        for (auto& e : w.key_blocks[b]) {
            for (int k = 0; k < PLIES_PER_BLOCK; ++k) e.w[k] = static_cast<int16_t>((rng.next() & 511U) - 256);
            for (int k = 0; k < 4; ++k) e.aux[k] = static_cast<uint8_t>(rng.next() & 0xFFU);
        }
        for (int8_t& t : w.corner_tempo[b]) {
            t = static_cast<int8_t>((rng.next() & 63U) - 32);
        }
        for (int g = 0; g < 3; ++g) {
            w.screlu[b].mu[g][0]      = 32;
            w.screlu[b].mu[g][1]      = 16;
            w.screlu[b].beta_sq[g][0] = 64;
            w.screlu[b].beta_sq[g][1] = -48;
            w.screlu[b].beta_pr[g]    = 32;
        }
        for (int k = 0; k < 512; ++k) {
            w.macro[b].t_cw[k] = static_cast<int16_t>((rng.next() & 255U) - 128);
            w.macro[b].t_ea[k] = static_cast<int16_t>((rng.next() & 255U) - 128);
            w.macro[b].t_cd[k] = static_cast<int16_t>((rng.next() & 255U) - 128);
        }
    }
    for (int ply = 0; ply < NUM_PLIES; ++ply) {
        for (int16_t& v : w.nonkey_w[ply]) {
            v = static_cast<int16_t>((rng.next() & 511U) - 256);
        }
    }

    TTEntry* tt = static_cast<TTEntry*>(std::aligned_alloc(64, TT_SIZE * sizeof(TTEntry)));
    if (!tt) { std::fprintf(stderr, "Failed to allocate 256 MiB TT\n"); return 1; }
    std::memset(tt, 0, TT_SIZE * sizeof(TTEntry));

    auto leaves = build_dfs_replay_trace(200000);
    const double raw_mib = (static_cast<double>(pt.total_raw_entries) * 2.0 + static_cast<double>(CORNER_RAW_SIZE))
                         * NUM_PLIES / (1024.0 * 1024.0);
    const double key_mib = static_cast<double>(pt.key_canon_entries) * sizeof(BlockEntry16) * NUM_BLOCKS / (1024.0 * 1024.0);
    const double nk_mib  = static_cast<double>(pt.nonkey_raw_entries) * sizeof(int16_t) * NUM_PLIES / (1024.0 * 1024.0);
    const double tmp_kib = static_cast<double>(CORNER_CANON_SIZE) * NUM_BLOCKS / 1024.0;
    const double p2c_mib = key_mib + nk_mib + tmp_kib / 1024.0;

    std::printf("=== P2C-64 v1.1 (Milestone 1.5) Pure-Scalar Evaluation Benchmark (Zero SIMD) ===\n");
    std::printf("Replay Trace : %zu midgame DFS leaves (roots ply 20..30, depth=9, leaves ply 29..39)\n", leaves.size());
    std::printf("Table Sizes  : RAW_NTUPLE_64 = %.2f MiB (2B w + 1B c4-tempo x 60)\n", raw_mib);
    std::printf("               P2C-64 v1.1   = %.2f MiB (Key-14 16B: %.2f MiB + NonKey-50 2B: %.2f MiB + Tempo: %.1f KiB + 30 KiB MacroLUT)\n",
                p2c_mib, key_mib, nk_mib, tmp_kib);
    std::printf("TT Contention: 256 MiB Transposition Table (16,777,216 x 16B random R/W per leaf)\n\n");
    std::printf("%-28s | %-30s | %-30s | %s\n",
                "Mode", "Eval-only (DFS tree)", "Eval + 256 MiB TT", "Chk");
    std::printf("-----------------------------+--------------------------------+--------------------------------+-----\n");

    double ref_eval = 0.0, ref_tt = 0.0;
    bench_one("1. RAW_NTUPLE_64 (2Bx60)", leaves, tt,
        [&](const uint16_t* idx, int stm, int ply) { return eval_raw_ntuple64(raw, pt.meta, idx, stm, ply); },
        ref_eval, ref_tt);
    bench_one("2. LINEAR_64 (Asym v1.1)", leaves, tt,
        [&](const uint16_t* idx, int stm, int ply) { return evaluate_p2c64<EvalMode::LINEAR_64>(w, pt.meta, idx, stm, ply); },
        ref_eval, ref_tt);
    bench_one("3. CAND_A_SCRELU_K2 (Fact)", leaves, tt,
        [&](const uint16_t* idx, int stm, int ply) { return evaluate_p2c64<EvalMode::CAND_A_SCRELU_K2>(w, pt.meta, idx, stm, ply); },
        ref_eval, ref_tt);
    bench_one("4. CAND_B_3W3B (Naive)", leaves, tt,
        [&](const uint16_t* idx, int stm, int ply) { return evaluate_p2c64<EvalMode::CAND_B_SCONED_3W3B_NAIVE>(w, pt.meta, idx, stm, ply); },
        ref_eval, ref_tt);
    bench_one("5. CAND_B_3W3B (Key-14 1st)", leaves, tt,
        [&](const uint16_t* idx, int stm, int ply) { return evaluate_p2c64<EvalMode::CAND_B_SCONED_3W3B>(w, pt.meta, idx, stm, ply); },
        ref_eval, ref_tt);
    bench_one("6. CAND_C_HYBRID (Fact+LUT)", leaves, tt,
        [&](const uint16_t* idx, int stm, int ply) { return evaluate_p2c64<EvalMode::CAND_C_HYBRID>(w, pt.meta, idx, stm, ply); },
        ref_eval, ref_tt);

    std::free(tt);
    return 0;
}
