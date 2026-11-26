#include "p2c64_eval.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace p2c64;

static_assert(sizeof(BlockEntry16) == 16 && alignof(BlockEntry16) == 16, "BlockEntry16 must be 16B aligned");
static_assert(sizeof(SCReLUPlyParams) == 48, "SCReLUPlyParams must be 48B");
static_assert(sizeof(MacroLUTPly) == 3072, "MacroLUTPly must be 3072B");

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() noexcept { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    int32_t range(int32_t lo, int32_t hi) noexcept {
        return lo + static_cast<int32_t>(next() % static_cast<uint64_t>(hi - lo + 1));
    }
};

struct TestPos {
    uint64_t black;
    uint64_t white;
    int      stm;
    int      ply;
    bool     is_pass;
};

static void fill_random_weights(P2C64Weights& w, Rng& rng) {
    for (int b = 0; b < NUM_BLOCKS; ++b) {
        for (auto& e : w.key_blocks[b]) {
            for (int k = 0; k < PLIES_PER_BLOCK; ++k) {
                e.w[k] = static_cast<int16_t>(rng.range(-4000, 4000));
            }
            for (int k = 0; k < 4; ++k) {
                e.aux[k] = static_cast<uint8_t>(rng.next() & 0xFFU);
            }
        }
    }
    for (int ply = 0; ply < NUM_PLIES; ++ply) {
        for (int16_t& v : w.nonkey_w[ply]) {
            v = static_cast<int16_t>(rng.range(-4000, 4000));
        }
        for (int g = 0; g < 3; ++g) {
            for (int k = 0; k < 4; ++k) {
                w.screlu[ply].mu[g][k]   = static_cast<int16_t>(rng.range(8, 64));
                w.screlu[ply].beta[g][k] = static_cast<int16_t>(rng.range(-500, 500));
            }
        }
        for (int k = 0; k < 512; ++k) {
            w.macro[ply].t_cw[k] = static_cast<int16_t>(rng.range(-800, 800));
            w.macro[ply].t_ea[k] = static_cast<int16_t>(rng.range(-800, 800));
            w.macro[ply].t_cd[k] = static_cast<int16_t>(rng.range(-800, 800));
        }
    }
}

// Test 1: Structural & Incremental Index Update Verification (500 full random games)
static std::vector<TestPos> test_incremental_and_collect_positions() {
    const auto& pt = get_pattern_tables();
    if (pt.key_raw_entries != 91854 || pt.nonkey_raw_entries != 318168 || pt.total_raw_entries != 410022) {
        std::fprintf(stderr, "FAIL: table entry counts mismatch (key_raw=%d, nonkey_raw=%d, total_raw=%d)\n",
                     pt.key_raw_entries, pt.nonkey_raw_entries, pt.total_raw_entries);
        std::exit(1);
    }

    Rng rng(0x20261125ULL);
    std::vector<TestPos> openings, midgames, endgames, passes;
    uint64_t total_moves_checked = 0;

    for (int g = 0; g < 500; ++g) {
        uint64_t B = (1ULL << rc(3, 4)) | (1ULL << rc(4, 3));
        uint64_t W = (1ULL << rc(3, 3)) | (1ULL << rc(4, 4));
        int stm = 1;
        int pass_streak = 0;

        uint16_t cur_idx[NUM_PATTERNS];
        pt.extract_indices(B, W, cur_idx);

        while (pass_streak < 2) {
            int discs = __builtin_popcountll(B | W);
            int ply = discs - 4;
            if (ply >= NUM_PLIES) break;

            uint64_t P = (stm > 0) ? B : W;
            uint64_t O = (stm > 0) ? W : B;
            uint64_t mv = generate_moves(P, O);

            if (!mv) {
                if (generate_moves(O, P) != 0 && passes.size() < 150) {
                    passes.push_back({B, W, stm, ply, true});
                }
                pass_streak++;
                stm = -stm;
                continue;
            }
            pass_streak = 0;

            if (ply <= 14 && openings.size() < 250 && (rng.next() & 7U) == 0) {
                openings.push_back({B, W, stm, ply, false});
            } else if (ply >= 15 && ply <= 44 && midgames.size() < 400 && (rng.next() & 7U) == 0) {
                midgames.push_back({B, W, stm, ply, false});
            } else if (ply >= 45 && ply <= 59 && endgames.size() < 250 && (rng.next() & 3U) == 0) {
                endgames.push_back({B, W, stm, ply, false});
            }

            int cnt = __builtin_popcountll(mv);
            int pick = rng.range(0, cnt - 1);
            int sq = 0;
            uint64_t m = mv;
            for (int k = 0; k <= pick; ++k) { sq = __builtin_ctzll(m); m &= m - 1; }
            uint64_t flips = calc_flips(sq, P, O);

            uint16_t inc_idx[NUM_PATTERNS];
            pt.update_indices_move(inc_idx, cur_idx, sq, flips, stm);

            if (stm > 0) { B |= (1ULL << sq) | flips; W ^= flips; }
            else         { W |= (1ULL << sq) | flips; B ^= flips; }

            uint16_t ref_idx[NUM_PATTERNS];
            pt.extract_indices(B, W, ref_idx);
            for (int p = 0; p < NUM_PATTERNS; ++p) {
                if (inc_idx[p] != ref_idx[p]) {
                    std::fprintf(stderr, "FAIL: incremental index mismatch at game=%d ply=%d pat=%d\n", g, ply, p);
                    std::exit(1);
                }
            }
            std::memcpy(cur_idx, inc_idx, sizeof(cur_idx));
            total_moves_checked++;
            stm = -stm;
        }
    }

    std::vector<TestPos> all;
    all.reserve(1000);
    auto push_cap = [&](const std::vector<TestPos>& src, size_t limit) {
        for (size_t i = 0; i < src.size() && i < limit && all.size() < 1000; ++i) all.push_back(src[i]);
    };
    push_cap(passes,   150);
    push_cap(openings, 250);
    push_cap(endgames, 250);
    push_cap(midgames, 400);
    if (all.size() < 1000) {
        std::fprintf(stderr, "FAIL: collected only %zu positions (< 1000)\n", all.size());
        std::exit(1);
    }
    all.resize(1000);
    std::printf("[PASS] Test 1: Structural (key_raw=91,854 | nonkey_raw=318,168 | total=410,022) & Incremental (%llu moves / 500 games)\n",
                static_cast<unsigned long long>(total_moves_checked));
    return all;
}

// Test 2: Full 16-Symmetry (D4 x Z2) Bit-Exact Invariance Test across all modes
template<EvalMode MODE>
static void verify_mode_16sym(const char* name, const std::vector<TestPos>& positions, uint64_t seed) {
    const auto& pt = get_pattern_tables();
    P2C64Weights w;
    Rng rng(seed);
    fill_random_weights(w, rng);
    w.symmetrize_for_mode(MODE);

    int64_t non_zero_evals = 0;
    int64_t non_zero_res   = 0;
    int64_t max_err        = 0;

    for (size_t i = 0; i < positions.size(); ++i) {
        const TestPos& tp = positions[i];
        uint16_t base_idx[NUM_PATTERNS];
        pt.extract_indices(tp.black, tp.white, base_idx);
        const int32_t base_val = evaluate_p2c64<MODE>(w, pt.meta, base_idx, tp.stm, tp.ply);
        const int32_t lin_val  = evaluate_p2c64<EvalMode::LINEAR_64>(w, pt.meta, base_idx, tp.stm, tp.ply);
        if (base_val != 0) non_zero_evals++;
        if (base_val != lin_val) non_zero_res++;

        for (int sym = 0; sym < 16; ++sym) {
            uint64_t tb = 0, tw = 0;
            int tstm = 0;
            transform_board_d4z2(tp.black, tp.white, tp.stm, sym, tb, tw, tstm);
            uint16_t tidx[NUM_PATTERNS];
            pt.extract_indices(tb, tw, tidx);
            const int32_t val = evaluate_p2c64<MODE>(w, pt.meta, tidx, tstm, tp.ply);
            const int64_t err = std::llabs(static_cast<int64_t>(val) - static_cast<int64_t>(base_val));
            if (err > max_err) max_err = err;
            if (err != 0) {
                std::fprintf(stderr, "FAIL: %s symmetry mismatch pos=%zu ply=%d sym=%d (base=%d vs got=%d)\n",
                             name, i, tp.ply, sym, base_val, val);
                std::exit(1);
            }
        }
        if constexpr (MODE == EvalMode::CAND_B_PTM_3W3B) {
            const int32_t naive_val = evaluate_p2c64<EvalMode::CAND_B_PTM_3W3B_NAIVE>(w, pt.meta, base_idx, tp.stm, tp.ply);
            if (naive_val != base_val) {
                std::fprintf(stderr, "FAIL: CAND_B Sched vs Naive mismatch (%d != %d)\n", base_val, naive_val);
                std::exit(1);
            }
        }
    }

    if (non_zero_evals < 950) {
        std::fprintf(stderr, "FAIL: %s produced degenerate near-zero outputs (%lld/1000)\n",
                     name, static_cast<long long>(non_zero_evals));
        std::exit(1);
    }
    if constexpr (MODE != EvalMode::LINEAR_64) {
        if (non_zero_res < 950) {
            std::fprintf(stderr, "FAIL: %s residual was unexpectedly zero (%lld/1000)\n",
                         name, static_cast<long long>(non_zero_res));
            std::exit(1);
        }
    }
    std::printf("[PASS] Test 2 (%-22s): 16,000 D4xZ2 evals | max_err=%lld | non_zero=%lld/1000 | active_res=%lld/1000\n",
                name, static_cast<long long>(max_err), static_cast<long long>(non_zero_evals),
                static_cast<long long>(non_zero_res));
}

// Test 3: RAW_NTUPLE_64 16-symmetry AND exact bit-identity with LINEAR_64 (0 handicap guarantee!)
static void verify_raw_and_linear_identity(const std::vector<TestPos>& positions) {
    const auto& pt = get_pattern_tables();
    RawNTuple64Weights raw;
    Rng rng(99999ULL);
    for (int ply = 0; ply < NUM_PLIES; ++ply) {
        for (int16_t& v : raw.w_ply[ply]) v = static_cast<int16_t>(rng.range(-4000, 4000));
    }
    raw.symmetrize();

    P2C64Weights lin;
    lin.load_linear_from_raw(raw);

    int64_t non_zero = 0;
    for (size_t i = 0; i < positions.size(); ++i) {
        const TestPos& tp = positions[i];
        uint16_t base_idx[NUM_PATTERNS];
        pt.extract_indices(tp.black, tp.white, base_idx);
        const int32_t raw_val = eval_raw_ntuple64(raw, pt.meta, base_idx, tp.stm, tp.ply);
        const int32_t lin_val = evaluate_p2c64<EvalMode::LINEAR_64>(lin, pt.meta, base_idx, tp.stm, tp.ply);
        if (raw_val != lin_val) {
            std::fprintf(stderr, "FAIL: RAW != LINEAR at pos=%zu stm=%d ply=%d (%d != %d)\n",
                         i, tp.stm, tp.ply, raw_val, lin_val);
            std::exit(1);
        }
        // Also verify opposite stm
        const int32_t raw_opp = eval_raw_ntuple64(raw, pt.meta, base_idx, -tp.stm, tp.ply);
        const int32_t lin_opp = evaluate_p2c64<EvalMode::LINEAR_64>(lin, pt.meta, base_idx, -tp.stm, tp.ply);
        if (raw_opp != lin_opp) {
            std::fprintf(stderr, "FAIL: RAW != LINEAR (opp stm) at pos=%zu (%d != %d)\n", i, raw_opp, lin_opp);
            std::exit(1);
        }
        if (raw_val != 0) non_zero++;

        for (int sym = 0; sym < 16; ++sym) {
            uint64_t tb = 0, tw = 0; int tstm = 0;
            transform_board_d4z2(tp.black, tp.white, tp.stm, sym, tb, tw, tstm);
            uint16_t tidx[NUM_PATTERNS];
            pt.extract_indices(tb, tw, tidx);
            const int32_t val = eval_raw_ntuple64(raw, pt.meta, tidx, tstm, tp.ply);
            if (val != raw_val) {
                std::fprintf(stderr, "FAIL: RAW_NTUPLE_64 symmetry mismatch pos=%zu sym=%d (%d != %d)\n",
                             i, sym, raw_val, val);
                std::exit(1);
            }
        }
    }
    std::printf("[PASS] Test 2 (%-22s): 16,000 D4xZ2 evals | max_err=0 | non_zero=%lld/1000\n",
                "RAW_NTUPLE_64", static_cast<long long>(non_zero));
    std::printf("[PASS] Test 3 (RAW == LINEAR_64 Bit-Exact): 2,000 evals (both stm=+1/-1) | max_err=0\n");
}

// Test 4: Extreme Boundary int32_t Overflow Safety (exact match with int64_t reference)
static void test_extreme_overflow_safety() {
    const auto& pt = get_pattern_tables();
    P2C64Weights w;

    uint16_t idx_min[NUM_PATTERNS], idx_max[NUM_PATTERNS];
    for (int p = 0; p < NUM_PATTERNS; ++p) {
        idx_min[p] = 0;
        idx_max[p] = static_cast<uint16_t>(pt.meta[p].maxv);
    }

    auto set_extreme = [&](int16_t wval, int8_t hval, int16_t muval, int16_t bval, int16_t mval) {
        for (int b = 0; b < NUM_BLOCKS; ++b) {
            for (int t = 0; t < 4; ++t) {
                const int32_t off  = pt.type_raw_off[t];
                const int32_t maxv = pt.type_raw_size[t] - 1;
                for (int32_t loc : {0, maxv}) {
                    BlockEntry16& e = w.key_blocks[b][off + loc];
                    for (int k = 0; k < PLIES_PER_BLOCK; ++k) e.w[k] = wval;
                    e.aux[0] = static_cast<uint8_t>(hval);
                    e.aux[1] = static_cast<uint8_t>(hval);
                    e.aux[2] = static_cast<uint8_t>(hval);
                    e.aux[3] = static_cast<uint8_t>(hval);
                }
            }
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            for (int t = 4; t < NUM_PATTERN_TYPES; ++t) {
                const int32_t off  = pt.type_nonkey_off[t];
                const int32_t maxv = pt.type_raw_size[t] - 1;
                w.nonkey_w[ply][off]        = wval;
                w.nonkey_w[ply][off + maxv] = wval;
            }
            for (int g = 0; g < 3; ++g) {
                for (int k = 0; k < 4; ++k) {
                    w.screlu[ply].mu[g][k]   = muval;
                    w.screlu[ply].beta[g][k] = bval;
                }
            }
            for (int k = 0; k < 512; ++k) {
                w.macro[ply].t_cw[k] = mval;
                w.macro[ply].t_ea[k] = mval;
                w.macro[ply].t_cd[k] = mval;
            }
        }
    };

    for (int sgn : {1, -1}) {
        const int16_t wv = static_cast<int16_t>(sgn * 32767);
        const int8_t  hv = 127; // Saturates all 7 regions x 4 ch to u = 127 -> u^2 = 16129!
        const int16_t bv = static_cast<int16_t>(sgn * SCRELU_BETA_MAX);
        const int16_t mv = static_cast<int16_t>(sgn * 32767);

        set_extreme(wv, hv, 127, bv, mv);
        const int64_t ref_lin = 64LL * wv;
        const int64_t ref_acc_a = 7LL * 4LL * static_cast<int64_t>(bv) * (127LL * 127LL);
        const int64_t ref_res_a = (ref_acc_a >= 0) ? (ref_acc_a / (1LL << SCRELU_SHIFT))
                                                   : -((-ref_acc_a) / (1LL << SCRELU_SHIFT));
        const int64_t ref_a = ref_lin + ref_res_a;

        for (int stm : {1, -1}) {
            const uint16_t* idx = (stm > 0) ? idx_min : idx_max;
            int32_t got_a = evaluate_p2c64<EvalMode::CAND_A_PTM_SCRELU4>(w, pt.meta, idx, stm, 30);
            if (static_cast<int64_t>(got_a) != ref_a) {
                std::fprintf(stderr, "FAIL: CAND_A extreme overflow/mismatch stm=%d (got=%d vs ref64=%lld)\n",
                             stm, got_a, static_cast<long long>(ref_a));
                std::exit(1);
            }
            int32_t got_b = evaluate_p2c64<EvalMode::CAND_B_PTM_3W3B>(w, pt.meta, idx, stm, 30);
            int64_t ref_b = ref_lin + 10LL * mv;
            if (static_cast<int64_t>(got_b) != ref_b) {
                std::fprintf(stderr, "FAIL: CAND_B extreme overflow/mismatch stm=%d (got=%d vs ref64=%lld)\n",
                             stm, got_b, static_cast<long long>(ref_b));
                std::exit(1);
            }
            int32_t got_c = evaluate_p2c64<EvalMode::CAND_C_PTM_HYBRID>(w, pt.meta, idx, stm, 30);
            const int64_t ref_acc_c = 7LL * 2LL * static_cast<int64_t>(bv) * (127LL * 127LL);
            const int64_t ref_res_c = (ref_acc_c >= 0) ? (ref_acc_c / (1LL << SCRELU_SHIFT))
                                                       : -((-ref_acc_c) / (1LL << SCRELU_SHIFT));
            int64_t ref_c = ref_lin + 10LL * mv + ref_res_c;
            if (static_cast<int64_t>(got_c) != ref_c) {
                std::fprintf(stderr, "FAIL: CAND_C extreme overflow/mismatch stm=%d (got=%d vs ref64=%lld)\n",
                             stm, got_c, static_cast<long long>(ref_c));
                std::exit(1);
            }
            int32_t got_l = evaluate_p2c64<EvalMode::LINEAR_64>(w, pt.meta, idx, stm, 30);
            if (static_cast<int64_t>(got_l) != ref_lin) {
                std::fprintf(stderr, "FAIL: LINEAR_64 extreme mismatch stm=%d (%d != %lld)\n",
                             stm, got_l, static_cast<long long>(ref_lin));
                std::exit(1);
            }
        }
    }
    std::printf("[PASS] Test 4: Extreme Boundary int32_t Overflow Safety (both stm=+1/-1, exact int64_t match across all 4 modes)\n");
}

int main() {
    std::printf("=== P2C-64 v2.0 (Full PTM): Symmetry, RAW==LINEAR Identity & Overflow Suite ===\n");
    auto positions = test_incremental_and_collect_positions();

    verify_raw_and_linear_identity(positions);
    verify_mode_16sym<EvalMode::LINEAR_64>            ("LINEAR_64",          positions, 1001ULL);
    verify_mode_16sym<EvalMode::CAND_A_PTM_SCRELU4>   ("CAND_A_PTM_SCRELU4", positions, 2002ULL);
    verify_mode_16sym<EvalMode::CAND_B_PTM_3W3B>      ("CAND_B_PTM_3W3B",    positions, 3003ULL);
    verify_mode_16sym<EvalMode::CAND_B_PTM_3W3B_NAIVE>("CAND_B_NAIVE",       positions, 3003ULL);
    verify_mode_16sym<EvalMode::CAND_C_PTM_HYBRID>    ("CAND_C_PTM_HYBRID",  positions, 4004ULL);

    test_extreme_overflow_safety();
    std::printf("=== ALL TESTS PASSED (0 ERRORS) ===\n");
    return 0;
}
