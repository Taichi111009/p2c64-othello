#include "p2c64_eval.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

using namespace p2c64;

static_assert(sizeof(BlockEntry16) == 16 && alignof(BlockEntry16) == 16, "BlockEntry16 must be 16B aligned");

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
        for (int8_t& t : w.corner_tempo[b]) {
            t = static_cast<int8_t>(rng.range(-64, 64));
        }
        for (int g = 0; g < 3; ++g) {
            w.screlu[b].mu[g][0]      = rng.range(16, 64);
            w.screlu[b].mu[g][1]      = rng.range(0, 48);
            w.screlu[b].beta_sq[g][0] = rng.range(-500, 500);
            w.screlu[b].beta_sq[g][1] = rng.range(-500, 500);
            w.screlu[b].beta_pr[g]    = rng.range(-500, 500);
        }
        for (int k = 0; k < 512; ++k) {
            w.macro[b].t_cw[k] = static_cast<int16_t>(rng.range(-800, 800));
            w.macro[b].t_ea[k] = static_cast<int16_t>(rng.range(-800, 800));
            w.macro[b].t_cd[k] = static_cast<int16_t>(rng.range(-800, 800));
        }
    }
    for (int ply = 0; ply < NUM_PLIES; ++ply) {
        for (int16_t& v : w.nonkey_w[ply]) {
            v = static_cast<int16_t>(rng.range(-4000, 4000));
        }
    }
}

// Test 1: Structural & Incremental Index Update Verification (500 full random games)
static std::vector<TestPos> test_incremental_and_collect_positions() {
    const auto& pt = get_pattern_tables();
    if (pt.total_canon_entries != 205019 || pt.key_canon_entries != 45929 || pt.nonkey_raw_entries != 318168) {
        std::fprintf(stderr, "FAIL: table entry counts mismatch (canon=%d, key_canon=%d, nonkey_raw=%d)\n",
                     pt.total_canon_entries, pt.key_canon_entries, pt.nonkey_raw_entries);
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

            // Pick a random legal move
            int cnt = __builtin_popcountll(mv);
            int pick = rng.range(0, cnt - 1);
            int sq = 0;
            uint64_t m = mv;
            for (int k = 0; k <= pick; ++k) {
                sq = __builtin_ctzll(m);
                m &= m - 1;
            }
            uint64_t flips = calc_flips(sq, P, O);

            uint16_t inc_idx[NUM_PATTERNS];
            pt.update_indices_move(inc_idx, cur_idx, sq, flips, stm);

            if (stm > 0) { B |= (1ULL << sq) | flips; W ^= flips; }
            else         { W |= (1ULL << sq) | flips; B ^= flips; }

            uint16_t ref_idx[NUM_PATTERNS];
            pt.extract_indices(B, W, ref_idx);
            for (int p = 0; p < NUM_PATTERNS; ++p) {
                if (inc_idx[p] != ref_idx[p]) {
                    std::fprintf(stderr, "FAIL: incremental index mismatch at game=%d ply=%d pat=%d (%u != %u)\n",
                                 g, ply, p, inc_idx[p], ref_idx[p]);
                    std::exit(1);
                }
            }
            std::memcpy(cur_idx, inc_idx, sizeof(cur_idx));
            total_moves_checked++;
            stm = -stm;
        }
    }

    // Assemble exactly 1,000 diverse positions (including pass positions!)
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
    std::printf("[PASS] Test 1: Structural (key_canon=45,929 | nonkey_raw=318,168) & Incremental Update (%llu moves / 500 games)\n",
                static_cast<unsigned long long>(total_moves_checked));
    std::printf("       Collected 1,000 test positions (pass=%zu, opening=%zu, midgame=%zu, endgame=%zu)\n",
                passes.size(), openings.size(), midgames.size(), endgames.size());
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
        if constexpr (MODE == EvalMode::CAND_B_SCONED_3W3B) {
            const int32_t naive_val = evaluate_p2c64<EvalMode::CAND_B_SCONED_3W3B_NAIVE>(w, pt.meta, base_idx, tp.stm, tp.ply);
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
        if (non_zero_res < 900) {
            std::fprintf(stderr, "FAIL: %s residual was unexpectedly zero (%lld/1000)\n",
                         name, static_cast<long long>(non_zero_res));
            std::exit(1);
        }
    }
    std::printf("[PASS] Test 2 (%-22s): 16,000 D4xZ2 evals | max_err=%lld | non_zero=%lld/1000 | active_res=%lld/1000\n",
                name, static_cast<long long>(max_err), static_cast<long long>(non_zero_evals),
                static_cast<long long>(non_zero_res));
}

static void verify_raw_ntuple_16sym(const std::vector<TestPos>& positions) {
    const auto& pt = get_pattern_tables();
    RawNTuple64Weights raw;
    Rng rng(99999ULL);
    for (int ply = 0; ply < NUM_PLIES; ++ply) {
        for (int16_t& v : raw.w_ply[ply])        v = static_cast<int16_t>(rng.range(-4000, 4000));
        for (int8_t&  v : raw.corner_tempo[ply]) v = static_cast<int8_t>(rng.range(-64, 64));
    }
    raw.symmetrize();
    int64_t non_zero = 0;
    for (size_t i = 0; i < positions.size(); ++i) {
        const TestPos& tp = positions[i];
        uint16_t base_idx[NUM_PATTERNS];
        pt.extract_indices(tp.black, tp.white, base_idx);
        const int32_t base_val = eval_raw_ntuple64(raw, pt.meta, base_idx, tp.stm, tp.ply);
        if (base_val != 0) non_zero++;
        for (int sym = 0; sym < 16; ++sym) {
            uint64_t tb = 0, tw = 0; int tstm = 0;
            transform_board_d4z2(tp.black, tp.white, tp.stm, sym, tb, tw, tstm);
            uint16_t tidx[NUM_PATTERNS];
            pt.extract_indices(tb, tw, tidx);
            const int32_t val = eval_raw_ntuple64(raw, pt.meta, tidx, tstm, tp.ply);
            if (val != base_val) {
                std::fprintf(stderr, "FAIL: RAW_NTUPLE_64 symmetry mismatch pos=%zu sym=%d (%d != %d)\n",
                             i, sym, base_val, val);
                std::exit(1);
            }
        }
    }
    std::printf("[PASS] Test 2 (%-22s): 16,000 D4xZ2 evals | max_err=0 | non_zero=%lld/1000\n",
                "RAW_NTUPLE_64", static_cast<long long>(non_zero));
}

// Test 3: Bit-Exact Verification of Algebraically Factored SCReLU (v1.1) vs Unfactored Reference (v1.0 formula)
static void test_factored_screlu_bit_exact(const std::vector<TestPos>& positions) {
    const auto& pt = get_pattern_tables();

    auto eval_cand_a_unfactored = [&](const P2C64Weights& w, const uint16_t* idx, int stm, int ply) -> int32_t {
        const uint8_t blk = PLY_TO_BLK[ply];
        const uint8_t sub = PLY_TO_SUB[ply];
        const BlockEntry16* ktbl = w.key_blocks[blk].data();
        const int16_t* nk_w      = w.nonkey_w[ply].data();
        const int8_t*  ct        = w.corner_tempo[blk].data();
        int64_t sum_A = 0, sum_T = 0;
        int64_t ko0[14], ko1[14], ke0[14], ke1[14];
        int64_t g_o0 = 0, g_o1 = 0, g_e0 = 0, g_e1 = 0;
        for (int p = 0; p < 14; ++p) {
            int32_t x = idx[p];
            int fl = (x > pt.meta[p].half) ? 1 : 0;
            int32_t cx = fl ? (pt.meta[p].maxv - x) : x;
            const BlockEntry16& e = ktbl[pt.meta[p].canon_off + cx];
            sum_A += fl ? -e.w[sub] : e.w[sub];
            if (p < 4) sum_T += ct[cx];
            int64_t o0 = fl ? -static_cast<int8_t>(e.aux[0]) : static_cast<int8_t>(e.aux[0]);
            int64_t o1 = fl ? -static_cast<int8_t>(e.aux[1]) : static_cast<int8_t>(e.aux[1]);
            int64_t e0 = static_cast<int8_t>(e.aux[2]);
            int64_t e1 = static_cast<int8_t>(e.aux[3]);
            ko0[p] = o0; ko1[p] = o1; ke0[p] = e0; ke1[p] = e1;
            g_o0 += o0; g_o1 += o1; g_e0 += e0; g_e1 += e1;
        }
        for (int p = 14; p < 64; ++p) sum_A += nk_w[pt.meta[p].nonkey_off + idx[p]];

        const SCReLUBlockParams& sp = w.screlu[blk];
        int64_t res_acc = 0;
        auto add_reg = [&](int grp, int64_t o0, int64_t e0, int64_t o1, int64_t e1) {
            auto cl = [](int64_t v) -> int64_t { return v < 0 ? 0 : (v > SCRELU_Q ? SCRELU_Q : v); };
            int64_t b0 = cl(e0 + sp.mu[grp][0] + o0);
            int64_t w0 = cl(e0 + sp.mu[grp][0] - o0);
            int64_t b1 = cl(e1 + sp.mu[grp][1] + o1);
            int64_t w1 = cl(e1 + sp.mu[grp][1] - o1);
            res_acc += static_cast<int64_t>(sp.beta_sq[grp][0]) * (b0 * b0 - w0 * w0)
                     + static_cast<int64_t>(sp.beta_sq[grp][1]) * (b1 * b1 - w1 * w1)
                     + static_cast<int64_t>(sp.beta_pr[grp])    * (b0 * b1 - w0 * w1);
        };
        for (int r = 0; r < 4; ++r) {
            int c1 = (r + 1) & 3;
            add_reg(0, ko0[4+r]+ko0[8+r]+ko0[r]+ko0[c1], ke0[4+r]+ke0[8+r]+ke0[r]+ke0[c1],
                       ko1[4+r]+ko1[8+r]+ko1[r]+ko1[c1], ke1[4+r]+ke1[8+r]+ke1[r]+ke1[c1]);
        }
        for (int d = 0; d < 2; ++d) {
            add_reg(1, ko0[12+d]+ko0[d]+ko0[d+2], ke0[12+d]+ke0[d]+ke0[d+2],
                       ko1[12+d]+ko1[d]+ko1[d+2], ke1[12+d]+ke1[d]+ke1[d+2]);
        }
        add_reg(2, g_o0, g_e0, g_o1, g_e1);
        int64_t a_res = (res_acc >= 0) ? (res_acc / (1LL << SCRELU_SHIFT)) : -((-res_acc) / (1LL << SCRELU_SHIFT));
        int64_t tot = ((stm < 0) ? -(sum_A + a_res) : (sum_A + a_res)) + sum_T;
        return static_cast<int32_t>(tot);
    };

    auto eval_cand_c_unfactored = [&](const P2C64Weights& w, const uint16_t* idx, int stm, int ply) -> int32_t {
        const uint8_t blk = PLY_TO_BLK[ply];
        const uint8_t sub = PLY_TO_SUB[ply];
        const BlockEntry16* ktbl = w.key_blocks[blk].data();
        const int16_t* nk_w      = w.nonkey_w[ply].data();
        const int8_t*  ct        = w.corner_tempo[blk].data();
        int64_t sum_A = 0, sum_T = 0;
        int64_t ko0[14], ke0[14], g_o0 = 0, g_e0 = 0;
        uint32_t zf[14], zr[4];
        for (int p = 0; p < 14; ++p) {
            int32_t x = idx[p];
            int fl = (x > pt.meta[p].half) ? 1 : 0;
            int32_t cx = fl ? (pt.meta[p].maxv - x) : x;
            const BlockEntry16& e = ktbl[pt.meta[p].canon_off + cx];
            sum_A += fl ? -e.w[sub] : e.w[sub];
            if (p < 4) sum_T += ct[cx];
            int64_t o0 = fl ? -static_cast<int8_t>(e.aux[0]) : static_cast<int8_t>(e.aux[0]);
            int64_t e0 = static_cast<int8_t>(e.aux[1]);
            ko0[p] = o0; ke0[p] = e0; g_o0 += o0; g_e0 += e0;
            uint32_t packed = e.aux[2];
            zf[p] = color_inv_z3(packed & 7U, fl);
            if (p >= 4 && p < 8) zr[p - 4] = color_inv_z3((packed >> 3) & 7U, fl);
        }
        for (int p = 14; p < 64; ++p) sum_A += nk_w[pt.meta[p].nonkey_off + idx[p]];

        const MacroLUTBlock& ml = w.macro[blk];
        int64_t a_lut = 0;
        for (int i = 0; i < 4; ++i) {
            a_lut += ml.t_cw[(zf[i] << 6) | (zf[4 + i] << 3) | zr[(i + 3) & 3]];
            a_lut += ml.t_ea[(zf[4 + i] << 6) | (zr[i] << 3) | zf[8 + i]];
        }
        a_lut += ml.t_cd[(zf[0] << 6) | (zf[12] << 3) | zf[2]];
        a_lut += ml.t_cd[(zf[1] << 6) | (zf[13] << 3) | zf[3]];

        const SCReLUBlockParams& sp = w.screlu[blk];
        int64_t res_acc = 0;
        auto add_reg = [&](int grp, int64_t o0, int64_t e0) {
            auto cl = [](int64_t v) -> int64_t { return v < 0 ? 0 : (v > SCRELU_Q ? SCRELU_Q : v); };
            int64_t b0 = cl(e0 + sp.mu[grp][0] + o0);
            int64_t w0 = cl(e0 + sp.mu[grp][0] - o0);
            res_acc += static_cast<int64_t>(sp.beta_sq[grp][0]) * (b0 * b0 - w0 * w0);
        };
        for (int r = 0; r < 4; ++r) {
            int c1 = (r + 1) & 3;
            add_reg(0, ko0[4+r]+ko0[8+r]+ko0[r]+ko0[c1], ke0[4+r]+ke0[8+r]+ke0[r]+ke0[c1]);
        }
        for (int d = 0; d < 2; ++d) {
            add_reg(1, ko0[12+d]+ko0[d]+ko0[d+2], ke0[12+d]+ke0[d]+ke0[d+2]);
        }
        add_reg(2, g_o0, g_e0);
        int64_t a_sc = (res_acc >= 0) ? (res_acc / (1LL << SCRELU_SHIFT)) : -((-res_acc) / (1LL << SCRELU_SHIFT));
        int64_t tot = ((stm < 0) ? -(sum_A + a_sc + a_lut) : (sum_A + a_sc + a_lut)) + sum_T;
        return static_cast<int32_t>(tot);
    };

    P2C64Weights wa, wc;
    Rng rng_a(7777ULL), rng_c(8888ULL);
    fill_random_weights(wa, rng_a); wa.symmetrize_for_mode(EvalMode::CAND_A_SCRELU_K2);
    fill_random_weights(wc, rng_c); wc.symmetrize_for_mode(EvalMode::CAND_C_HYBRID);

    for (size_t i = 0; i < positions.size(); ++i) {
        const TestPos& tp = positions[i];
        uint16_t idx[NUM_PATTERNS];
        pt.extract_indices(tp.black, tp.white, idx);

        int32_t fa = evaluate_p2c64<EvalMode::CAND_A_SCRELU_K2>(wa, pt.meta, idx, tp.stm, tp.ply);
        int32_t ra = eval_cand_a_unfactored(wa, idx, tp.stm, tp.ply);
        if (fa != ra) {
            std::fprintf(stderr, "FAIL: CAND_A factored != unfactored at pos=%zu (%d != %d)\n", i, fa, ra);
            std::exit(1);
        }

        int32_t fc = evaluate_p2c64<EvalMode::CAND_C_HYBRID>(wc, pt.meta, idx, tp.stm, tp.ply);
        int32_t rc = eval_cand_c_unfactored(wc, idx, tp.stm, tp.ply);
        if (fc != rc) {
            std::fprintf(stderr, "FAIL: CAND_C factored != unfactored at pos=%zu (%d != %d)\n", i, fc, rc);
            std::exit(1);
        }
    }
    std::printf("[PASS] Test 3: Factored SCReLU Bit-Exact Identity (CAND_A & CAND_C, 1,000 pos each | max_err=0)\n");
}

// Test 4: Extreme Boundary Overflow Safety (exact match with int64_t reference)
static void test_extreme_overflow_safety() {
    const auto& pt = get_pattern_tables();
    P2C64Weights w;

    // Worst-case extreme pattern indices: all 0 (flip=0) and all maxv (flip=1)
    uint16_t idx_min[NUM_PATTERNS], idx_max[NUM_PATTERNS];
    for (int p = 0; p < NUM_PATTERNS; ++p) {
        idx_min[p] = 0;
        idx_max[p] = static_cast<uint16_t>(pt.meta[p].maxv);
    }

    auto set_extreme = [&](int16_t wval, int8_t oval, int8_t eval_e, int8_t tval,
                           int32_t muval, int32_t bval, int16_t mval) {
        for (int b = 0; b < NUM_BLOCKS; ++b) {
            for (int t = 0; t < 4; ++t) {
                int32_t off = pt.type_canon_off[t];
                BlockEntry16& e = w.key_blocks[b][off]; // cx = 0
                for (int k = 0; k < PLIES_PER_BLOCK; ++k) e.w[k] = wval;
                e.aux[0] = static_cast<uint8_t>(oval);
                e.aux[1] = static_cast<uint8_t>(oval);
                e.aux[2] = static_cast<uint8_t>(eval_e);
                e.aux[3] = static_cast<uint8_t>(eval_e);
            }
            w.corner_tempo[b][0] = tval;
            for (int g = 0; g < 3; ++g) {
                w.screlu[b].mu[g][0]      = muval;
                w.screlu[b].mu[g][1]      = muval;
                w.screlu[b].beta_sq[g][0] = bval;
                w.screlu[b].beta_sq[g][1] = bval;
                w.screlu[b].beta_pr[g]    = bval;
            }
            for (int k = 0; k < 512; ++k) {
                w.macro[b].t_cw[k] = mval;
                w.macro[b].t_ea[k] = mval;
                w.macro[b].t_cd[k] = mval;
            }
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            for (int t = 4; t < NUM_PATTERN_TYPES; ++t) {
                const int32_t off  = pt.type_nonkey_off[t];
                const int32_t maxv = pt.type_raw_size[t] - 1;
                w.nonkey_w[ply][off]        = wval;
                w.nonkey_w[ply][off + maxv] = static_cast<int16_t>(-wval);
            }
        }
    };

    for (int sgn : {1, -1}) {
        int16_t wv = static_cast<int16_t>(sgn * 32767);
        int8_t  ov = static_cast<int8_t>(sgn * 127);
        int8_t  tv = static_cast<int8_t>(sgn * 127);
        int32_t bv = sgn * SCRELU_BETA_MAX;
        int16_t mv = static_cast<int16_t>(sgn * 32767);

        // o=±127, e=0, mu=0 => max possible |b^2 - w^2| = 16129 across all 7 regions!
        set_extreme(wv, ov, 0, tv, 0, bv, mv);

        int64_t ref_A = 64LL * wv;
        int64_t ref_T = 4LL  * tv; // Corner-4 tempo
        int64_t b0 = (ov > 0) ? 127 : 0;
        int64_t w0 = (ov > 0) ? 0   : 127;
        int64_t ref_res_acc = 7LL * (3LL * bv * (b0 * b0 - w0 * w0));
        int64_t ref_a_res   = (ref_res_acc >= 0) ? (ref_res_acc / (1LL << SCRELU_SHIFT))
                                                 : -((-ref_res_acc) / (1LL << SCRELU_SHIFT));
        int64_t ref_tot     = ref_A + ref_a_res + ref_T;

        int32_t got_a = evaluate_p2c64<EvalMode::CAND_A_SCRELU_K2>(w, pt.meta, idx_min, 1, 30);
        if (static_cast<int64_t>(got_a) != ref_tot) {
            std::fprintf(stderr, "FAIL: CAND_A extreme overflow/mismatch (got=%d vs ref64=%lld)\n",
                         got_a, static_cast<long long>(ref_tot));
            std::exit(1);
        }

        // CAND_B extreme check (all 10 macro lookups = mv)
        int32_t got_b = evaluate_p2c64<EvalMode::CAND_B_SCONED_3W3B>(w, pt.meta, idx_min, 1, 30);
        int64_t ref_b = ref_A + 10LL * mv + ref_T;
        if (static_cast<int64_t>(got_b) != ref_b) {
            std::fprintf(stderr, "FAIL: CAND_B extreme overflow/mismatch (got=%d vs ref64=%lld)\n",
                         got_b, static_cast<long long>(ref_b));
            std::exit(1);
        }

        // CAND_C extreme check (aux[0]=ov, aux[1]=e0=0, aux[2]=codes=0)
        for (int b = 0; b < NUM_BLOCKS; ++b) {
            for (int t = 0; t < 4; ++t) {
                BlockEntry16& e = w.key_blocks[b][pt.type_canon_off[t]];
                e.aux[0] = static_cast<uint8_t>(ov);
                e.aux[1] = 0;
                e.aux[2] = 0;
                e.aux[3] = 0;
            }
        }
        int32_t got_c = evaluate_p2c64<EvalMode::CAND_C_HYBRID>(w, pt.meta, idx_min, 1, 30);
        int64_t ref_c_acc = 7LL * (1LL * bv * (b0 * b0 - w0 * w0));
        int64_t ref_c_sc  = (ref_c_acc >= 0) ? (ref_c_acc / (1LL << SCRELU_SHIFT))
                                             : -((-ref_c_acc) / (1LL << SCRELU_SHIFT));
        int64_t ref_c = ref_A + ref_c_sc + 10LL * mv + ref_T;
        if (static_cast<int64_t>(got_c) != ref_c) {
            std::fprintf(stderr, "FAIL: CAND_C extreme overflow/mismatch (got=%d vs ref64=%lld)\n",
                         got_c, static_cast<long long>(ref_c));
            std::exit(1);
        }

        // Also test idx_max (all flip = 1 for key-14, maxv for nonkey-50, stm = -1)
        int32_t got_l_max = evaluate_p2c64<EvalMode::LINEAR_64>(w, pt.meta, idx_max, -1, 30);
        if (static_cast<int64_t>(got_l_max) != ref_A + ref_T) {
            std::fprintf(stderr, "FAIL: LINEAR_64 idx_max mismatch (%d != %lld)\n",
                         got_l_max, static_cast<long long>(ref_A + ref_T));
            std::exit(1);
        }
    }
    std::printf("[PASS] Test 4: Extreme Boundary int32_t Overflow Safety (exact match vs int64_t across all 4 modes)\n");
}

int main() {
    std::printf("=== P2C-64 v1.1 (Milestone 1.5): Symmetry, Incremental, Factored & Overflow Suite ===\n");
    auto positions = test_incremental_and_collect_positions();

    verify_raw_ntuple_16sym(positions);
    verify_mode_16sym<EvalMode::LINEAR_64>               ("LINEAR_64",        positions, 1001ULL);
    verify_mode_16sym<EvalMode::CAND_A_SCRELU_K2>        ("CAND_A_SCRELU_K2", positions, 2002ULL);
    verify_mode_16sym<EvalMode::CAND_B_SCONED_3W3B>      ("CAND_B_3W3B",      positions, 3003ULL);
    verify_mode_16sym<EvalMode::CAND_B_SCONED_3W3B_NAIVE>("CAND_B_NAIVE",     positions, 3003ULL);
    verify_mode_16sym<EvalMode::CAND_C_HYBRID>           ("CAND_C_HYBRID",    positions, 4004ULL);

    test_factored_screlu_bit_exact(positions);
    test_extreme_overflow_safety();
    std::printf("=== ALL TESTS PASSED (0 ERRORS) ===\n");
    return 0;
}
