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

// Test 5: Kogge-Stone Paired Move Generator Bit-Exact Equivalence
static void test_ks_move_pair_equivalence(const std::vector<TestPos>& positions) {
    for (const auto& tp : positions) {
        const uint64_t P = (tp.stm > 0) ? tp.black : tp.white;
        const uint64_t O = (tp.stm > 0) ? tp.white : tp.black;
        uint64_t mvP = 0, mvO = 0;
        generate_moves_pair(P, O, mvP, mvO);
        const uint64_t refP = generate_moves(P, O);
        const uint64_t refO = generate_moves(O, P);
        if (mvP != refP || mvO != refO) {
            std::fprintf(stderr, "FAIL: generate_moves_pair mismatch! got=(%016llx,%016llx) ref=(%016llx,%016llx)\n",
                         static_cast<unsigned long long>(mvP), static_cast<unsigned long long>(mvO),
                         static_cast<unsigned long long>(refP), static_cast<unsigned long long>(refO));
            std::exit(1);
        }
    }
    std::printf("[PASS] Test 5: Kogge-Stone generate_moves_pair Bit-Exact Equivalence (%zu positions)\n", positions.size());
}

// Test 6: WACA-G2 1,000+ positions x 16 Symmetries (D4 x Z2) across Stage 1, Stage 2.5, Stage 3
static void test_waca_g2_16sym(const std::vector<TestPos>& positions, int hidden_dim, uint64_t seed) {
    const auto& pt = get_pattern_tables();
    Rng rng(seed);
    WacaG2Weights w;
    w.mlp_hidden = hidden_dim;

    for (int ply = 0; ply < NUM_PLIES; ++ply) {
        for (int16_t& v : w.tier1.w_ply[ply]) v = static_cast<int16_t>(rng.range(-2500, 2500));
        w.s25[ply].b = rng.range(-50000, 50000);
        for (int i = 0; i < WACA_F25; ++i) w.s25[ply].w[i] = static_cast<int16_t>(rng.range(-4000, 4000));
        for (int g = 0; g < 3; ++g) {
            for (int k = 0; k < 4; ++k) {
                w.screlu[ply].mu[g][k]   = static_cast<int16_t>(rng.range(8, 64));
                w.screlu[ply].beta[g][k] = static_cast<int16_t>(rng.range(-500, 500));
            }
        }
        w.mlp[ply].b2 = rng.range(-50000, 50000);
        for (int j = 0; j < hidden_dim; ++j) {
            w.mlp[ply].b1[j] = rng.range(-80000, 80000);
            w.mlp[ply].w2[j] = static_cast<int16_t>(rng.range(-1200, 1200));
            for (int i = 0; i < WACA_F3; ++i) {
                w.mlp[ply].w1[j][i] = static_cast<int8_t>(rng.range(-120, 120));
            }
        }
    }
    for (int b = 0; b < WACA_EO_BLOCKS; ++b) {
        for (auto& e : w.h_emb[b]) {
            for (int k = 0; k < 4; ++k) e.h[k] = static_cast<int8_t>(rng.range(-100, 100));
        }
    }
    w.symmetrize();

    int32_t max_e1 = 0, max_e25 = 0, max_e3 = 0, max_raw_err = 0;
    for (size_t i = 0; i < positions.size(); ++i) {
        const auto& tp = positions[i];
        uint16_t idx0[NUM_PATTERNS];
        pt.extract_indices(tp.black, tp.white, idx0);

        int32_t b1 = 0, b25 = 0, b3 = 0;
        eval_waca_g2_stages(w, pt.meta, idx0, tp.black, tp.white, tp.stm, tp.ply, b1, b25, b3);
        const int32_t raw_v = eval_raw_ntuple64(w.tier1, pt.meta, idx0, tp.stm, tp.ply);
        max_raw_err = std::max(max_raw_err, abs_i32(b1 - raw_v));

        for (int sym = 0; sym < 16; ++sym) {
            uint64_t tb = 0, tw = 0;
            int tstm = 0;
            transform_board_d4z2(tp.black, tp.white, tp.stm, sym, tb, tw, tstm);
            uint16_t tidx[NUM_PATTERNS];
            pt.extract_indices(tb, tw, tidx);
            int32_t t1 = 0, t25 = 0, t3 = 0;
            eval_waca_g2_stages(w, pt.meta, tidx, tb, tw, tstm, tp.ply, t1, t25, t3);
            max_e1  = std::max(max_e1,  abs_i32(t1  - b1));
            max_e25 = std::max(max_e25, abs_i32(t25 - b25));
            max_e3  = std::max(max_e3,  abs_i32(t3  - b3));
        }
    }

    if (max_e1 != 0 || max_e25 != 0 || max_e3 != 0 || max_raw_err != 0) {
        std::fprintf(stderr, "FAIL: WACA_G2(H=%d) symmetry/identity error! e1=%d e25=%d e3=%d raw=%d\n",
                     hidden_dim, max_e1, max_e25, max_e3, max_raw_err);
        std::exit(1);
    }
    std::printf("[PASS] Test 6: WACA_G2(H=%2d) 16-Symmetry (%zu pos x 16 = %zu evals) | max_err: S1=%d S2.5=%d S3=%d RAW==S1=%d\n",
                hidden_dim, positions.size(), positions.size() * 16, max_e1, max_e25, max_e3, max_raw_err);
}

// Test 7: WACA-G2 Extreme Boundary int32_t Overflow Safety (exact int64_t match at H=24 worst case!)
static void test_waca_g2_overflow_safety() {
    const auto& pt = get_pattern_tables();
    WacaG2Weights w;
    w.mlp_hidden = WACA_MAX_HID; // 24 (worst-case accumulator!)

    uint16_t idx[NUM_PATTERNS];
    const uint64_t B = 0xffffffff00000000ULL;
    const uint64_t W = 0x0000000000ffffffULL;
    pt.extract_indices(B, W, idx);

    for (int sgn : {1, -1}) {
        const int16_t w1_t1 = static_cast<int16_t>(sgn * 20000);
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            std::fill(w.tier1.w_ply[ply].begin(), w.tier1.w_ply[ply].end(), w1_t1);
            w.s25[ply].b = sgn * WACA_S25_B_MAX;
            for (int i = 0; i < WACA_F25; ++i) w.s25[ply].w[i] = static_cast<int16_t>(sgn * WACA_S25_W_MAX);
            for (int g = 0; g < 3; ++g) {
                for (int k = 0; k < 4; ++k) {
                    w.screlu[ply].mu[g][k]   = 127;
                    w.screlu[ply].beta[g][k] = static_cast<int16_t>(sgn * SCRELU_BETA_MAX);
                }
            }
            w.mlp[ply].b2 = sgn * WACA_MLP_B2_MAX;
            for (int j = 0; j < WACA_MAX_HID; ++j) {
                w.mlp[ply].b1[j] = WACA_MLP_B1_MAX; // Force max positive z_j -> u_j = 127!
                w.mlp[ply].w2[j] = static_cast<int16_t>(sgn * WACA_MLP_W2_MAX);
                for (int i = 0; i < WACA_F3; ++i) w.mlp[ply].w1[j][i] = static_cast<int8_t>(WACA_MLP_W1_MAX);
            }
        }
        for (int b = 0; b < WACA_EO_BLOCKS; ++b) {
            for (auto& e : w.h_emb[b]) e.h[0] = e.h[1] = e.h[2] = e.h[3] = 127;
        }

        for (int stm : {1, -1}) {
            const int ply = 30;
            const uint64_t P = (stm > 0) ? B : W;
            const uint64_t O = (stm > 0) ? W : B;
            int32_t c[4], e[4], f25[WACA_F25], fex[4];
            const int32_t v1 = (stm < 0) ? eval_waca_stage1_ptm<true>(w.tier1, pt.meta, idx, ply)
                                         : eval_waca_stage1_ptm<false>(w.tier1, pt.meta, idx, ply);
            if (stm < 0) extract_waca_ce_ptm<true>(w.tier1, pt.meta, idx, ply, c, e);
            else         extract_waca_ce_ptm<false>(w.tier1, pt.meta, idx, ply, c, e);
            const int32_t r25 = eval_waca_stage25_kernel(w, P, O, v1, c, e, ply, f25);
            const int32_t rt  = (stm < 0) ? eval_waca_stage3_ptm<true>(w, pt.meta, idx, P, O, c, e, f25, r25, ply)
                                          : eval_waca_stage3_ptm<false>(w, pt.meta, idx, P, O, c, e, f25, r25, ply);

            // Compute exact int64_t reference for Stage 2.5 and Stage 3!
            int64_t acc25 = w.s25[ply].b;
            for (int i = 0; i < WACA_F25; ++i) acc25 += static_cast<int64_t>(w.s25[ply].w[i]) * f25[i];
            int64_t r25_64 = (acc25 >= 0) ? (acc25 / (1LL << WACA_S25_SHIFT)) : -((-acc25) / (1LL << WACA_S25_SHIFT));
            r25_64 = std::clamp<int64_t>(r25_64, -WACA_MAX_RES, WACA_MAX_RES);

            extract_waca_stage3_geom_features(P, O, c, e, fex);
            int64_t f64[WACA_F3];
            for (int i = 0; i < WACA_F25; ++i) f64[i] = f25[i];
            for (int i = 0; i < 4; ++i) f64[WACA_F25 + i] = fex[i];
            // All 7 regions saturate to 127 -> 7 * 127^2 = 112,903 -> >> 8 = 441
            for (int k = 0; k < 4; ++k) f64[20 + k] = (7LL * 127LL * 127LL) >> 8;

            int64_t sc_acc64 = 7LL * 4LL * static_cast<int64_t>(w.screlu[ply].beta[0][0]) * (127LL * 127LL);
            int64_t r_sc64 = (sc_acc64 >= 0) ? (sc_acc64 / (1LL << SCRELU_SHIFT)) : -((-sc_acc64) / (1LL << SCRELU_SHIFT));

            int64_t acc2_64 = w.mlp[ply].b2;
            for (int j = 0; j < WACA_MAX_HID; ++j) {
                int64_t z64 = w.mlp[ply].b1[j];
                for (int i = 0; i < WACA_F3; ++i) z64 += static_cast<int64_t>(w.mlp[ply].w1[j][i]) * f64[i];
                int64_t u64 = (z64 <= 0) ? 0 : std::min<int64_t>(z64 >> WACA_MLP_S1, SCRELU_Q);
                acc2_64 += static_cast<int64_t>(w.mlp[ply].w2[j]) * (u64 * u64);
            }
            int64_t r_mlp64 = (acc2_64 >= 0) ? (acc2_64 / (1LL << WACA_MLP_S2)) : -((-acc2_64) / (1LL << WACA_MLP_S2));
            int64_t rt64 = std::clamp<int64_t>(r25_64 + r_mlp64 + r_sc64, -WACA_MAX_RES, WACA_MAX_RES);

            if (static_cast<int64_t>(r25) != r25_64 || static_cast<int64_t>(rt) != rt64) {
                std::fprintf(stderr, "FAIL: WACA_G2 overflow mismatch sgn=%d stm=%d r25=(%d vs %lld) rt=(%d vs %lld)\n",
                             sgn, stm, r25, static_cast<long long>(r25_64), rt, static_cast<long long>(rt64));
                std::exit(1);
            }
        }
    }
    std::printf("[PASS] Test 7: WACA_G2(H=24) Extreme Boundary int32_t Overflow Safety (exact int64_t match for Stage 2.5 & Stage 3)\n");
}

int main() {
    std::printf("=== P2C-64 v3.0 (Full PTM + WACA-G2): Symmetry, Identity & Overflow Suite ===\n");
    auto positions = test_incremental_and_collect_positions();

    verify_raw_and_linear_identity(positions);
    verify_mode_16sym<EvalMode::LINEAR_64>            ("LINEAR_64",          positions, 1001ULL);
    verify_mode_16sym<EvalMode::CAND_A_PTM_SCRELU4>   ("CAND_A_PTM_SCRELU4", positions, 2002ULL);
    verify_mode_16sym<EvalMode::CAND_B_PTM_3W3B>      ("CAND_B_PTM_3W3B",    positions, 3003ULL);
    verify_mode_16sym<EvalMode::CAND_B_PTM_3W3B_NAIVE>("CAND_B_NAIVE",       positions, 3003ULL);
    verify_mode_16sym<EvalMode::CAND_C_PTM_HYBRID>    ("CAND_C_PTM_HYBRID",  positions, 4004ULL);

    test_extreme_overflow_safety();
    test_ks_move_pair_equivalence(positions);
    test_waca_g2_16sym(positions, 16, 5005ULL);
    test_waca_g2_16sym(positions, 24, 6006ULL);
    test_waca_g2_overflow_safety();
    std::printf("=== ALL TESTS PASSED (0 ERRORS) ===\n");
    return 0;
}
