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
        int root_ply = 20 + (r % 11);
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
        dfs_collect(pt, B, W, stm, cur_ply, 9, rng.next(), root_idx, leaves, leaves.size() + per_root);
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

    for (const auto& lf : leaves) sink += eval_fn(lf.idx, lf.stm, lf.ply);

    auto t0 = std::chrono::steady_clock::now();
    for (int pass = 0; pass < PASSES_EVAL; ++pass) {
        for (const auto& lf : leaves) {
            sink += eval_fn(lf.idx, lf.stm, lf.ply);
        }
    }
    double eval_ns = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                   * 1e9 / (static_cast<double>(PASSES_EVAL) * static_cast<double>(leaves.size()));

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

    // Initialize RawNTuple64 (46.92 MiB true PTM) and copy identical linear weights to P2C64Weights!
    RawNTuple64Weights raw;
    for (int ply = 0; ply < NUM_PLIES; ++ply) {
        for (int16_t& v : raw.w_ply[ply]) v = static_cast<int16_t>((rng.next() & 511U) - 256);
    }

    P2C64Weights w;
    w.load_linear_from_raw(raw);
    for (int b = 0; b < NUM_BLOCKS; ++b) {
        for (auto& e : w.key_blocks[b]) {
            for (int k = 0; k < 4; ++k) e.aux[k] = static_cast<uint8_t>(rng.next() & 0xFFU);
        }
    }
    for (int ply = 0; ply < NUM_PLIES; ++ply) {
        for (int g = 0; g < 3; ++g) {
            for (int k = 0; k < 4; ++k) {
                w.screlu[ply].mu[g][k]   = 16;
                w.screlu[ply].beta[g][k] = static_cast<int16_t>((k & 1) ? -48 : 64);
            }
        }
        for (int k = 0; k < 512; ++k) {
            w.macro[ply].t_cw[k] = static_cast<int16_t>((rng.next() & 255U) - 128);
            w.macro[ply].t_ea[k] = static_cast<int16_t>((rng.next() & 255U) - 128);
            w.macro[ply].t_cd[k] = static_cast<int16_t>((rng.next() & 255U) - 128);
        }
    }

    TTEntry* tt = static_cast<TTEntry*>(std::aligned_alloc(64, TT_SIZE * sizeof(TTEntry)));
    if (!tt) { std::fprintf(stderr, "Failed to allocate 256 MiB TT\n"); return 1; }
    std::memset(tt, 0, TT_SIZE * sizeof(TTEntry));

    auto leaves = build_dfs_replay_trace(200000);
    const double raw_mib = static_cast<double>(pt.total_raw_entries) * 2.0 * NUM_PLIES / (1024.0 * 1024.0);
    const double key_mib = static_cast<double>(pt.key_raw_entries) * sizeof(BlockEntry16) * NUM_BLOCKS / (1024.0 * 1024.0);
    const double nk_mib  = static_cast<double>(pt.nonkey_raw_entries) * sizeof(int16_t) * NUM_PLIES / (1024.0 * 1024.0);
    const double p2c_mib = key_mib + nk_mib;

    std::printf("=== P2C-64 v2.0 (Full PTM) Pure-Scalar Evaluation Benchmark (Zero SIMD) ===\n");
    std::printf("Replay Trace : %zu midgame DFS leaves (roots ply 20..30, depth=9, leaves ply 29..39)\n", leaves.size());
    std::printf("Table Sizes  : RAW_NTUPLE_64 = %.2f MiB (64-pat PTM x 2B x 60)\n", raw_mib);
    std::printf("               P2C-64 v2.0   = %.2f MiB (Key-14 16B: %.2f MiB + NonKey-50 2B: %.2f MiB | +2.81 KiB SCReLU / +180 KiB MacroLUT)\n",
                p2c_mib, key_mib, nk_mib);
    std::printf("TT Contention: 256 MiB Transposition Table (16,777,216 x 16B random R/W per leaf)\n\n");
    std::printf("%-28s | %-30s | %-30s | %s\n",
                "Mode", "Eval-only (DFS tree)", "Eval + 256 MiB TT", "Chk");
    std::printf("-----------------------------+--------------------------------+--------------------------------+-----\n");

    double ref_eval = 0.0, ref_tt = 0.0;
    bench_one("1. RAW_NTUPLE_64 (PTM 2B)", leaves, tt,
        [&](const uint16_t* idx, int stm, int ply) { return eval_raw_ntuple64(raw, pt.meta, idx, stm, ply); },
        ref_eval, ref_tt);
    bench_one("2. LINEAR_64 (PTM v2.0)", leaves, tt,
        [&](const uint16_t* idx, int stm, int ply) { return evaluate_p2c64<EvalMode::LINEAR_64>(w, pt.meta, idx, stm, ply); },
        ref_eval, ref_tt);
    bench_one("3. CAND_A_PTM_SCRELU4", leaves, tt,
        [&](const uint16_t* idx, int stm, int ply) { return evaluate_p2c64<EvalMode::CAND_A_PTM_SCRELU4>(w, pt.meta, idx, stm, ply); },
        ref_eval, ref_tt);
    bench_one("4. CAND_B_PTM_3W3B (Naive)", leaves, tt,
        [&](const uint16_t* idx, int stm, int ply) { return evaluate_p2c64<EvalMode::CAND_B_PTM_3W3B_NAIVE>(w, pt.meta, idx, stm, ply); },
        ref_eval, ref_tt);
    bench_one("5. CAND_B_PTM_3W3B (Key14)", leaves, tt,
        [&](const uint16_t* idx, int stm, int ply) { return evaluate_p2c64<EvalMode::CAND_B_PTM_3W3B>(w, pt.meta, idx, stm, ply); },
        ref_eval, ref_tt);
    bench_one("6. CAND_C_PTM_HYBRID", leaves, tt,
        [&](const uint16_t* idx, int stm, int ply) { return evaluate_p2c64<EvalMode::CAND_C_PTM_HYBRID>(w, pt.meta, idx, stm, ply); },
        ref_eval, ref_tt);

    // ------------------------------------------------------------------------
    // WACA-G2 v3.0 Standalone Stage Benchmarks (Stage 1 / Stage 2.5 / Stage 3)
    // ------------------------------------------------------------------------
    WacaG2Weights wg2;
    wg2.mlp_hidden = 16;
    wg2.tier1 = raw;
    for (int ply = 0; ply < NUM_PLIES; ++ply) {
        wg2.s25[ply].b = 128;
        for (int i = 0; i < WACA_F25; ++i) wg2.s25[ply].w[i] = static_cast<int16_t>(((ply + i) & 1) ? 45 : -35);
        wg2.screlu[ply] = w.screlu[ply];
        wg2.mlp[ply].b2 = 64;
        for (int j = 0; j < 16; ++j) {
            wg2.mlp[ply].b1[j] = 4096;
            wg2.mlp[ply].w2[j] = static_cast<int16_t>((j & 1) ? 24 : -18);
            for (int i = 0; i < WACA_F3; ++i) wg2.mlp[ply].w1[j][i] = static_cast<int8_t>(((j * 7 + i * 3) % 31) - 15);
        }
    }
    for (int b = 0; b < WACA_EO_BLOCKS; ++b) {
        for (auto& e : wg2.h_emb[b]) {
            e.h[0] = static_cast<int8_t>((rng.next() & 127U) - 64);
            e.h[1] = static_cast<int8_t>((rng.next() & 127U) - 64);
            e.h[2] = static_cast<int8_t>((rng.next() & 127U) - 64);
            e.h[3] = static_cast<int8_t>((rng.next() & 127U) - 64);
        }
    }

    // Reconstruct realistic midgame bitboards for Stage 2.5 / Stage 3 standalone microbenchmark
    std::vector<std::pair<uint64_t, uint64_t>> boards(leaves.size());
    for (size_t i = 0; i < leaves.size(); ++i) {
        uint64_t P = 0, O = 0;
        for (int sq = 0; sq < 64; ++sq) {
            uint64_t r = rng.next() % 3ULL;
            if (r == 1) P |= (1ULL << sq);
            else if (r == 2) O |= (1ULL << sq);
        }
        boards[i] = {P, O};
    }

    bench_one("7. WACA_G2 Stage 1 (Tier1)", leaves, tt,
        [&](const uint16_t* idx, int stm, int ply) {
            return (stm < 0) ? eval_waca_stage1_ptm<true>(wg2.tier1, pt.meta, idx, ply)
                             : eval_waca_stage1_ptm<false>(wg2.tier1, pt.meta, idx, ply);
        },
        ref_eval, ref_tt);

    volatile int64_t sink_w = 0;
    int32_t c_dum[4] = {120, -40, 85, -15}, e_dum[4] = {55, -25, 10, 30};
    int32_t f25_buf[WACA_F25]{};

    auto t_s25 = std::chrono::steady_clock::now();
    for (int pass = 0; pass < 6; ++pass) {
        for (size_t i = 0; i < leaves.size(); ++i) {
            sink_w += eval_waca_stage25_kernel(wg2, boards[i].first, boards[i].second,
                                               128, c_dum, e_dum, leaves[i].ply, f25_buf);
        }
    }
    const double s25_ns = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_s25).count()
                        * 1e9 / (6.0 * static_cast<double>(leaves.size()));

    auto t_s3 = std::chrono::steady_clock::now();
    for (int pass = 0; pass < 6; ++pass) {
        for (size_t i = 0; i < leaves.size(); ++i) {
            const auto& lf = leaves[i];
            sink_w += (lf.stm < 0)
                    ? eval_waca_stage3_ptm<true>(wg2, pt.meta, lf.idx, boards[i].first, boards[i].second,
                                                 c_dum, e_dum, f25_buf, 16, lf.ply)
                    : eval_waca_stage3_ptm<false>(wg2, pt.meta, lf.idx, boards[i].first, boards[i].second,
                                                  c_dum, e_dum, f25_buf, 16, lf.ply);
        }
    }
    const double s3_ns = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_s3).count()
                       * 1e9 / (6.0 * static_cast<double>(leaves.size()));

    std::printf("\n=== WACA-G2 v3.0 Standalone Stage Microbenchmark (H=16, Zero SIMD) ===\n");
    std::printf("  Stage 2.5 (16-dim global mob+geom linear) : %6.1f ns/call  (amortized @15%% = %4.1f ns/leaf)\n",
                s25_ns, s25_ns * 0.15);
    std::printf("  Stage 3   (20-blk EO Key14 + 24->16->1)   : %6.1f ns/call  (amortized @ 5%% = %4.1f ns/leaf)\n",
                s3_ns, s3_ns * 0.05);
    std::printf("  Amortized 2-Stage Overhead (15%% S2.5 + 5%% S3): %5.1f ns/leaf (chk:%02llx)\n",
                s25_ns * 0.15 + s3_ns * 0.05, static_cast<unsigned long long>(sink_w & 0xFF));

    std::free(tt);
    return 0;
}
