#pragma once

#include "p2c64_patterns.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace p2c64 {

// ============================================================================
// Pure-Scalar Integer Primitives (Zero SIMD, Zero Branches in Hot Paths)
// ============================================================================
static inline constexpr int32_t clamp_i32(int32_t x, int32_t hi) noexcept {
    return (x < 0) ? 0 : ((x > hi) ? hi : x);
}

// Round-toward-zero symmetric right shift: sym_rshift(-x, S) == -sym_rshift(x, S)
static inline constexpr int32_t sym_rshift(int32_t x, int shift) noexcept {
    return (x + ((x >> 31) & ((1 << shift) - 1))) >> shift;
}

// Constexpr ply -> (block, sub) LUTs to avoid magic-multiply `imul` for ply/6
static constexpr std::array<uint8_t, NUM_PLIES> PLY_TO_BLK = {
    0,0,0,0,0,0, 1,1,1,1,1,1, 2,2,2,2,2,2, 3,3,3,3,3,3, 4,4,4,4,4,4,
    5,5,5,5,5,5, 6,6,6,6,6,6, 7,7,7,7,7,7, 8,8,8,8,8,8, 9,9,9,9,9,9
};
static constexpr std::array<uint8_t, NUM_PLIES> PLY_TO_SUB = {
    0,1,2,3,4,5, 0,1,2,3,4,5, 0,1,2,3,4,5, 0,1,2,3,4,5, 0,1,2,3,4,5,
    0,1,2,3,4,5, 0,1,2,3,4,5, 0,1,2,3,4,5, 0,1,2,3,4,5, 0,1,2,3,4,5
};

// ============================================================================
// Core 16-Byte Block-Packed Entry for Key 14 Nodes (PTM, 4 entries / 64B line)
// ============================================================================
struct alignas(16) BlockEntry16 {
    int16_t w[PLIES_PER_BLOCK]; // 12B: 6 consecutive plies of PTM linear weight W
    uint8_t aux[4];             //  4B: 100% dedicated to Key 14 nonlinear residual data
};
static_assert(sizeof(BlockEntry16) == 16 && alignof(BlockEntry16) == 16, "Must be 16B aligned");

enum class EvalMode : int {
    RAW_NTUPLE_64         = 0,
    LINEAR_64             = 1,
    CAND_A_PTM_SCRELU4    = 2,
    CAND_B_PTM_3W3B_NAIVE = 3,
    CAND_B_PTM_3W3B       = 4,
    CAND_C_PTM_HYBRID     = 5
};

// ============================================================================
// 60-Ply SCReLU Head Parameters (48B / ply -> 2.81 KiB total, L1-resident!)
//   Group 0: 4 Edge strips  (r = 0..3)
//   Group 1: 2 Diag strips  (r = 4..5)
//   Group 2: 1 Global sum   (r = 6)
//   Safety: Q = 127, |beta| <= 2047 => max |res_acc| <= 7*4*127^2*2047 = 924,449,764 < INT32_MAX
// ============================================================================
static constexpr int32_t SCRELU_Q        = 127;
static constexpr int32_t SCRELU_BETA_MAX = 2047;
static constexpr int     SCRELU_SHIFT    = 7;

struct SCReLUPlyParams {
    int16_t mu[3][4]{};   // 24B: 3 D4 groups x 4 channels (CAND_C uses k=0..1)
    int16_t beta[3][4]{}; // 24B: 3 D4 groups x 4 channels (|beta| <= 2047)
};
static_assert(sizeof(SCReLUPlyParams) == 48, "SCReLUPlyParams must be 48 bytes (3*16 -> 0 imul)");

// ============================================================================
// 60-Ply SCONE-D 3way-3bit Macro-LUT (3 tables x 512 int16 = 3.0 KiB / ply; 180.0 KiB total)
//   All 8 discrete codes {0..7} fully unlocked under PTM!
//   T_CW[512]: (C3x3[i], E2X[i].fwd, E2X[(i+3)&3].rev)   x4
//   T_EA[512]: (E2X[i].fwd, E2X[i].rev, L2[i])           x4
//   T_CD[512]: (C3x3[d], Diag8[d], C3x3[d+2])            x2
// ============================================================================
struct alignas(64) MacroLUTPly {
    int16_t t_cw[512]{};
    int16_t t_ea[512]{};
    int16_t t_cd[512]{};
};
static_assert(sizeof(MacroLUTPly) == 3072, "MacroLUTPly must be 3072 bytes (3*1024 -> 0 imul)");

// ============================================================================
// True Egaroucid-Compatible PTM Raw N-Tuple Baseline (46.92 MiB total)
//   All 64 patterns (410,022 PTM entries) x 60 plies x 2B int16_t
// ============================================================================
struct RawNTuple64Weights {
    std::array<std::vector<int16_t>, NUM_PLIES> w_ply;

    RawNTuple64Weights() {
        const auto& pt = get_pattern_tables();
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            w_ply[ply].assign(static_cast<size_t>(pt.total_raw_entries), 0);
        }
    }

    // D4 internal mirror symmetrization (Z2 color inversion is automatic under PTM!)
    void symmetrize() {
        const auto& pt = get_pattern_tables();
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            int16_t* wp = w_ply[ply].data();
            for (int t = 0; t < NUM_PATTERN_TYPES; ++t) {
                const int32_t off = pt.type_raw_off[t];
                const int32_t sz  = pt.type_raw_size[t];
                const int ng      = pt.type_sym[t].num_gens;
                if (ng == 0) continue;
                std::vector<uint8_t> vis(static_cast<size_t>(sz), 0);
                for (int32_t x = 0; x < sz; ++x) {
                    if (vis[x]) continue;
                    std::vector<int32_t> orb;
                    orb.push_back(x);
                    vis[x] = 1;
                    for (size_t qi = 0; qi < orb.size(); ++qi) {
                        int32_t cur = orb[qi];
                        for (int g = 0; g < ng; ++g) {
                            int32_t nxt = pt.permute_raw_index(t, g, cur);
                            if (!vis[nxt]) {
                                vis[nxt] = 1;
                                orb.push_back(nxt);
                            }
                        }
                    }
                    int16_t rep_w = wp[off + x];
                    for (int32_t idx : orb) wp[off + idx] = rep_w;
                }
            }
        }
    }
};

// ============================================================================
// P2C-64 v2.0 Full-PTM Asymmetric Memory Layout Weights Container (50.43 MiB)
//   - Key 14 nodes (types 0..3):     16B BlockEntry16 x 91,854 x 10 blk = 14.02 MiB
//   - Non-key 50 nodes (types 4..15): 2B int16_t x 318,168 x 60 plies   = 36.41 MiB
//   - 60-ply SCReLU Heads:           48B x 60 plies                     =  2.81 KiB
//   - 60-ply SCONE-D Macro-LUTs:     3.0 KiB x 60 plies                 = 180.0 KiB
// ============================================================================
struct P2C64Weights {
    std::array<std::vector<BlockEntry16>, NUM_BLOCKS> key_blocks;
    std::array<std::vector<int16_t>,      NUM_PLIES>  nonkey_w;
    std::array<SCReLUPlyParams,           NUM_PLIES>  screlu{};
    std::array<MacroLUTPly,               NUM_PLIES>  macro{};

    P2C64Weights() {
        const auto& pt = get_pattern_tables();
        for (int b = 0; b < NUM_BLOCKS; ++b) {
            key_blocks[b].resize(static_cast<size_t>(pt.key_raw_entries));
            std::memset(key_blocks[b].data(), 0, key_blocks[b].size() * sizeof(BlockEntry16));
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            nonkey_w[ply].assign(static_cast<size_t>(pt.nonkey_raw_entries), 0);
        }
    }

    // Copy identical 60-ply PTM linear weights from RawNTuple64Weights (strictly 0-handicap sharing!)
    void load_linear_from_raw(const RawNTuple64Weights& raw) {
        const auto& pt = get_pattern_tables();
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            const uint8_t blk = PLY_TO_BLK[ply];
            const uint8_t sub = PLY_TO_SUB[ply];
            const int16_t* rw = raw.w_ply[ply].data();
            BlockEntry16* kt  = key_blocks[blk].data();
            for (int32_t i = 0; i < pt.key_raw_entries; ++i) {
                kt[i].w[sub] = rw[i];
            }
            std::memcpy(nonkey_w[ply].data(),
                        rw + pt.key_raw_entries,
                        static_cast<size_t>(pt.nonkey_raw_entries) * sizeof(int16_t));
        }
    }

    // D4 mirror symmetrization for all 60-ply Macro-LUTs (no Z2 antisymmetry needed under PTM!)
    void symmetrize_macro_luts() {
        auto sym_table = [](int16_t tbl[512], bool swap_12, bool swap_01, bool swap_02) {
            for (int z0 = 0; z0 < 8; ++z0) {
                for (int z1 = 0; z1 < 8; ++z1) {
                    for (int z2 = 0; z2 < 8; ++z2) {
                        int k = (z0 << 6) | (z1 << 3) | z2;
                        int mk = k;
                        if (swap_12) mk = (z0 << 6) | (z2 << 3) | z1;
                        if (swap_01) mk = (z1 << 6) | (z0 << 3) | z2;
                        if (swap_02) mk = (z2 << 6) | (z1 << 3) | z0;
                        if (k < mk) tbl[mk] = tbl[k];
                    }
                }
            }
        };
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            sym_table(macro[ply].t_cw, true,  false, false);
            sym_table(macro[ply].t_ea, false, true,  false);
            sym_table(macro[ply].t_cd, false, false, true);
        }
    }

    // D4 internal mirror symmetrization for Key-14 BlockEntry16, NonKey-50 2B, SCReLU clamp, MacroLUT
    void symmetrize_for_mode(EvalMode mode) {
        const auto& pt = get_pattern_tables();
        auto clamp_i8_sym = [](uint8_t u) -> uint8_t {
            int8_t v = static_cast<int8_t>(u);
            if (v == INT8_MIN) v = -127;
            return static_cast<uint8_t>(v);
        };

        // 1) Key 4 pattern types (0..3) in BlockEntry16 (10 blocks)
        for (int b = 0; b < NUM_BLOCKS; ++b) {
            BlockEntry16* tbl = key_blocks[b].data();
            for (int t = 0; t < 4; ++t) {
                const int32_t off = pt.type_raw_off[t];
                const int32_t sz  = pt.type_raw_size[t];
                const int ng      = pt.type_sym[t].num_gens;
                std::vector<uint8_t> vis(static_cast<size_t>(sz), 0);

                for (int32_t x = 0; x < sz; ++x) {
                    if (vis[x]) continue;
                    std::vector<int32_t> orb;
                    orb.push_back(x);
                    vis[x] = 1;
                    for (size_t qi = 0; qi < orb.size(); ++qi) {
                        int32_t cur = orb[qi];
                        for (int g = 0; g < ng; ++g) {
                            int32_t nxt = pt.permute_raw_index(t, g, cur);
                            if (!vis[nxt]) {
                                vis[nxt] = 1;
                                orb.push_back(nxt);
                            }
                        }
                    }

                    BlockEntry16 rep = tbl[off + x];
                    for (int32_t idx : orb) {
                        for (int k = 0; k < PLIES_PER_BLOCK; ++k) tbl[off + idx].w[k] = rep.w[k];
                    }

                    if (mode == EvalMode::CAND_A_PTM_SCRELU4) {
                        uint8_t h0 = clamp_i8_sym(rep.aux[0]);
                        uint8_t h1 = clamp_i8_sym(rep.aux[1]);
                        uint8_t h2 = clamp_i8_sym(rep.aux[2]);
                        uint8_t h3 = clamp_i8_sym(rep.aux[3]);
                        for (int32_t idx : orb) {
                            tbl[off + idx].aux[0] = h0;
                            tbl[off + idx].aux[1] = h1;
                            tbl[off + idx].aux[2] = h2;
                            tbl[off + idx].aux[3] = h3;
                        }
                    } else if (mode == EvalMode::CAND_B_PTM_3W3B || mode == EvalMode::CAND_B_PTM_3W3B_NAIVE) {
                        if (t == 1) {
                            // Edge+2X directional codes: code_fwd(rev(x)) == code_rev(x)
                            uint8_t zf = rep.aux[0] & 7U;
                            uint8_t zr = rep.aux[1] & 7U;
                            if (orb.size() == 1) {
                                tbl[off + x].aux[0] = zf;
                                tbl[off + x].aux[1] = zf;
                                tbl[off + x].aux[2] = 0;
                                tbl[off + x].aux[3] = 0;
                            } else {
                                int32_t y = orb[1];
                                tbl[off + x].aux[0] = zf;
                                tbl[off + x].aux[1] = zr;
                                tbl[off + x].aux[2] = 0;
                                tbl[off + x].aux[3] = 0;
                                tbl[off + y].aux[0] = zr;
                                tbl[off + y].aux[1] = zf;
                                tbl[off + y].aux[2] = 0;
                                tbl[off + y].aux[3] = 0;
                            }
                        } else {
                            uint8_t z = rep.aux[0] & 7U;
                            for (int32_t idx : orb) {
                                tbl[off + idx].aux[0] = z;
                                tbl[off + idx].aux[1] = z;
                                tbl[off + idx].aux[2] = 0;
                                tbl[off + idx].aux[3] = 0;
                            }
                        }
                    } else if (mode == EvalMode::CAND_C_PTM_HYBRID) {
                        uint8_t h0 = clamp_i8_sym(rep.aux[0]);
                        uint8_t h1 = clamp_i8_sym(rep.aux[1]);
                        uint8_t zf = rep.aux[2] & 7U;
                        uint8_t zr = (rep.aux[2] >> 3) & 7U;
                        if (t == 1) {
                            if (orb.size() == 1) {
                                tbl[off + x].aux[0] = h0;
                                tbl[off + x].aux[1] = h1;
                                tbl[off + x].aux[2] = static_cast<uint8_t>(zf | (zf << 3));
                                tbl[off + x].aux[3] = 0;
                            } else {
                                int32_t y = orb[1];
                                tbl[off + x].aux[0] = h0;
                                tbl[off + x].aux[1] = h1;
                                tbl[off + x].aux[2] = static_cast<uint8_t>(zf | (zr << 3));
                                tbl[off + x].aux[3] = 0;
                                tbl[off + y].aux[0] = h0;
                                tbl[off + y].aux[1] = h1;
                                tbl[off + y].aux[2] = static_cast<uint8_t>(zr | (zf << 3));
                                tbl[off + y].aux[3] = 0;
                            }
                        } else {
                            uint8_t pk = static_cast<uint8_t>(zf | (zf << 3));
                            for (int32_t idx : orb) {
                                tbl[off + idx].aux[0] = h0;
                                tbl[off + idx].aux[1] = h1;
                                tbl[off + idx].aux[2] = pk;
                                tbl[off + idx].aux[3] = 0;
                            }
                        }
                    }
                }
            }
        }

        // 2) Non-key 12 pattern types (4..15) in 2B per-ply arrays (D4 mirror invariance only)
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            int16_t* wp = nonkey_w[ply].data();
            for (int t = 4; t < NUM_PATTERN_TYPES; ++t) {
                const int32_t off = pt.type_nonkey_off[t];
                const int32_t sz  = pt.type_raw_size[t];
                const int ng      = pt.type_sym[t].num_gens;
                if (ng == 0) continue;
                std::vector<uint8_t> vis(static_cast<size_t>(sz), 0);
                for (int32_t x = 0; x < sz; ++x) {
                    if (vis[x]) continue;
                    std::vector<int32_t> orb;
                    orb.push_back(x);
                    vis[x] = 1;
                    for (size_t qi = 0; qi < orb.size(); ++qi) {
                        int32_t cur = orb[qi];
                        for (int g = 0; g < ng; ++g) {
                            int32_t nxt = pt.permute_raw_index(t, g, cur);
                            if (!vis[nxt]) {
                                vis[nxt] = 1;
                                orb.push_back(nxt);
                            }
                        }
                    }
                    int16_t rep_w = wp[off + x];
                    for (int32_t idx : orb) wp[off + idx] = rep_w;
                }
            }
            for (int g = 0; g < 3; ++g) {
                for (int k = 0; k < 4; ++k) {
                    screlu[ply].beta[g][k] = static_cast<int16_t>(
                        std::clamp<int32_t>(screlu[ply].beta[g][k], -SCRELU_BETA_MAX, SCRELU_BETA_MAX));
                }
            }
        }
        symmetrize_macro_luts();
    }
};

// ============================================================================
// Pure-Scalar Evaluation Kernels (v2.0: Full PTM 1-Branch Template + 60-Ply Heads)
// ============================================================================

template<bool IS_WHITE>
__attribute__((noinline))
inline int32_t eval_raw_ntuple64_ptm(const RawNTuple64Weights& raw,
                                     const PatternMeta* __restrict meta,
                                     const uint16_t* __restrict idx,
                                     int ply) noexcept {
    const int16_t* __restrict wp = raw.w_ply[ply].data();
    int32_t sum = 0;
    for (int p = 0; p < NUM_PATTERNS; ++p) {
        const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        sum += wp[flat];
    }
    return sum;
}

inline int32_t eval_raw_ntuple64(const RawNTuple64Weights& raw,
                                 const PatternMeta* __restrict meta,
                                 const uint16_t* __restrict idx,
                                 int stm, int ply) noexcept {
    if (stm < 0) return eval_raw_ntuple64_ptm<true>(raw, meta, idx, ply);
    return eval_raw_ntuple64_ptm<false>(raw, meta, idx, ply);
}

template<EvalMode MODE, bool IS_WHITE>
__attribute__((noinline))
int32_t evaluate_p2c64_ptm(const P2C64Weights& weights,
                           const PatternMeta* __restrict meta,
                           const uint16_t* __restrict idx,
                           int ply) noexcept {
    const uint8_t blk = PLY_TO_BLK[ply];
    const uint8_t sub = PLY_TO_SUB[ply];
    const BlockEntry16* __restrict ktbl = weights.key_blocks[blk].data();
    const int16_t*      __restrict nk_w = weights.nonkey_w[ply].data();

    int32_t sum = 0;

    if constexpr (MODE == EvalMode::LINEAR_64) {
        // Key 14 nodes (PTM direct/1-sub into 16B BlockEntry16)
        for (int p = 0; p < NUM_KEY_PATTERNS; ++p) {
            const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
            sum += ktbl[flat].w[sub];
        }
        // Non-key 50 nodes (PTM direct/1-sub into 2B array)
        for (int p = NUM_KEY_PATTERNS; p < NUM_PATTERNS; ++p) {
            const int32_t flat = IS_WHITE ? (meta[p].nonkey_off_end - idx[p]) : (meta[p].nonkey_off + idx[p]);
            sum += nk_w[flat];
        }
        return sum;
    }
    else if constexpr (MODE == EvalMode::CAND_A_PTM_SCRELU4) {
        int32_t h0[NUM_KEY_PATTERNS], h1[NUM_KEY_PATTERNS];
        int32_t h2[NUM_KEY_PATTERNS], h3[NUM_KEY_PATTERNS];
        int32_t g0 = 0, g1 = 0, g2 = 0, g3 = 0;

        for (int p = 0; p < NUM_KEY_PATTERNS; ++p) {
            const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
            const BlockEntry16& e = ktbl[flat];
            sum += e.w[sub];
            const int32_t v0 = static_cast<int8_t>(e.aux[0]);
            const int32_t v1 = static_cast<int8_t>(e.aux[1]);
            const int32_t v2 = static_cast<int8_t>(e.aux[2]);
            const int32_t v3 = static_cast<int8_t>(e.aux[3]);
            h0[p] = v0; h1[p] = v1; h2[p] = v2; h3[p] = v3;
            g0 += v0;   g1 += v1;   g2 += v2;   g3 += v3;
        }
        for (int p = NUM_KEY_PATTERNS; p < NUM_PATTERNS; ++p) {
            const int32_t flat = IS_WHITE ? (meta[p].nonkey_off_end - idx[p]) : (meta[p].nonkey_off + idx[p]);
            sum += nk_w[flat];
        }

        // 60-ply smooth head (48B/ply, L1-resident) with distributive group factoring:
        //   28 squares (7 regions x 4 ch) + 12 group betas (3 groups x 4 ch) = 40 imuls!
        const SCReLUPlyParams& sp = weights.screlu[ply];
        int32_t res_acc = 0;

        // Group 0: 4 Edge strips r = 0..3
        {
            const int32_t m0 = sp.mu[0][0], m1 = sp.mu[0][1], m2 = sp.mu[0][2], m3 = sp.mu[0][3];
            int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (int r = 0; r < 4; ++r) {
                const int c1 = (r + 1) & 3;
                const int32_t u0 = clamp_i32(h0[4 + r] + h0[8 + r] + h0[r] + h0[c1] + m0, SCRELU_Q);
                const int32_t u1 = clamp_i32(h1[4 + r] + h1[8 + r] + h1[r] + h1[c1] + m1, SCRELU_Q);
                const int32_t u2 = clamp_i32(h2[4 + r] + h2[8 + r] + h2[r] + h2[c1] + m2, SCRELU_Q);
                const int32_t u3 = clamp_i32(h3[4 + r] + h3[8 + r] + h3[r] + h3[c1] + m3, SCRELU_Q);
                s0 += u0 * u0;
                s1 += u1 * u1;
                s2 += u2 * u2;
                s3 += u3 * u3;
            }
            res_acc += static_cast<int32_t>(sp.beta[0][0]) * s0
                     + static_cast<int32_t>(sp.beta[0][1]) * s1
                     + static_cast<int32_t>(sp.beta[0][2]) * s2
                     + static_cast<int32_t>(sp.beta[0][3]) * s3;
        }
        // Group 1: 2 Diag strips d = 0..1
        {
            const int32_t m0 = sp.mu[1][0], m1 = sp.mu[1][1], m2 = sp.mu[1][2], m3 = sp.mu[1][3];
            int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (int d = 0; d < 2; ++d) {
                const int32_t u0 = clamp_i32(h0[12 + d] + h0[d] + h0[d + 2] + m0, SCRELU_Q);
                const int32_t u1 = clamp_i32(h1[12 + d] + h1[d] + h1[d + 2] + m1, SCRELU_Q);
                const int32_t u2 = clamp_i32(h2[12 + d] + h2[d] + h2[d + 2] + m2, SCRELU_Q);
                const int32_t u3 = clamp_i32(h3[12 + d] + h3[d] + h3[d + 2] + m3, SCRELU_Q);
                s0 += u0 * u0;
                s1 += u1 * u1;
                s2 += u2 * u2;
                s3 += u3 * u3;
            }
            res_acc += static_cast<int32_t>(sp.beta[1][0]) * s0
                     + static_cast<int32_t>(sp.beta[1][1]) * s1
                     + static_cast<int32_t>(sp.beta[1][2]) * s2
                     + static_cast<int32_t>(sp.beta[1][3]) * s3;
        }
        // Group 2: 1 Global sum
        {
            const int32_t u0 = clamp_i32(g0 + sp.mu[2][0], SCRELU_Q);
            const int32_t u1 = clamp_i32(g1 + sp.mu[2][1], SCRELU_Q);
            const int32_t u2 = clamp_i32(g2 + sp.mu[2][2], SCRELU_Q);
            const int32_t u3 = clamp_i32(g3 + sp.mu[2][3], SCRELU_Q);
            res_acc += static_cast<int32_t>(sp.beta[2][0]) * (u0 * u0)
                     + static_cast<int32_t>(sp.beta[2][1]) * (u1 * u1)
                     + static_cast<int32_t>(sp.beta[2][2]) * (u2 * u2)
                     + static_cast<int32_t>(sp.beta[2][3]) * (u3 * u3);
        }

        return sum + sym_rshift(res_acc, SCRELU_SHIFT);
    }
    else if constexpr (MODE == EvalMode::CAND_B_PTM_3W3B || MODE == EvalMode::CAND_B_PTM_3W3B_NAIVE) {
        if constexpr (MODE == EvalMode::CAND_B_PTM_3W3B_NAIVE) {
            for (int p = NUM_KEY_PATTERNS; p < NUM_PATTERNS; ++p) {
                const int32_t flat = IS_WHITE ? (meta[p].nonkey_off_end - idx[p]) : (meta[p].nonkey_off + idx[p]);
                sum += nk_w[flat];
            }
        }

        // Key 14 nodes (ZERO multiplications, ZERO branches, ZERO color-inv bit ops!)
        uint32_t zf[NUM_KEY_PATTERNS];
        uint32_t zr[4];
        for (int p = 0; p < NUM_KEY_PATTERNS; ++p) {
            const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
            const BlockEntry16& e = ktbl[flat];
            sum += e.w[sub];
            zf[p] = e.aux[0] & 7U;
            if (p >= 4 && p < 8) zr[p - 4] = e.aux[1] & 7U;
        }

        // 10 L1-resident 60-ply Macro-LUT lookups (3.0 KiB/ply, strict 0 imul!)
        const MacroLUTPly& ml = weights.macro[ply];
        int32_t a_res = 0;
        for (int i = 0; i < 4; ++i) {
            a_res += ml.t_cw[(zf[i] << 6) | (zf[4 + i] << 3) | zr[(i + 3) & 3]];
            a_res += ml.t_ea[(zf[4 + i] << 6) | (zr[i] << 3) | zf[8 + i]];
        }
        a_res += ml.t_cd[(zf[0] << 6) | (zf[12] << 3) | zf[2]];
        a_res += ml.t_cd[(zf[1] << 6) | (zf[13] << 3) | zf[3]];

        if constexpr (MODE == EvalMode::CAND_B_PTM_3W3B) {
            // Key-14 First: overlap remaining 50 direct 2B loads with the 10 Macro-LUT loads!
            for (int p = NUM_KEY_PATTERNS; p < NUM_PATTERNS; ++p) {
                const int32_t flat = IS_WHITE ? (meta[p].nonkey_off_end - idx[p]) : (meta[p].nonkey_off + idx[p]);
                sum += nk_w[flat];
            }
        }
        return sum + a_res;
    }
    else { // EvalMode::CAND_C_PTM_HYBRID (2-ch PTM SCReLU [20 imul] + 60-ply 3way-3bit Macro-LUT, Key-14 First)
        int32_t h0[NUM_KEY_PATTERNS], h1[NUM_KEY_PATTERNS];
        int32_t g0 = 0, g1 = 0;
        uint32_t zf[NUM_KEY_PATTERNS];
        uint32_t zr[4];

        for (int p = 0; p < NUM_KEY_PATTERNS; ++p) {
            const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
            const BlockEntry16& e = ktbl[flat];
            sum += e.w[sub];
            const int32_t v0 = static_cast<int8_t>(e.aux[0]);
            const int32_t v1 = static_cast<int8_t>(e.aux[1]);
            h0[p] = v0; h1[p] = v1;
            g0 += v0;   g1 += v1;
            const uint32_t packed = e.aux[2];
            zf[p] = packed & 7U;
            if (p >= 4 && p < 8) zr[p - 4] = (packed >> 3) & 7U;
        }

        // Issue 10 Macro-LUT lookups immediately (Key-14 First!)
        const MacroLUTPly& ml = weights.macro[ply];
        int32_t a_lut = 0;
        for (int i = 0; i < 4; ++i) {
            a_lut += ml.t_cw[(zf[i] << 6) | (zf[4 + i] << 3) | zr[(i + 3) & 3]];
            a_lut += ml.t_ea[(zf[4 + i] << 6) | (zr[i] << 3) | zf[8 + i]];
        }
        a_lut += ml.t_cd[(zf[0] << 6) | (zf[12] << 3) | zf[2]];
        a_lut += ml.t_cd[(zf[1] << 6) | (zf[13] << 3) | zf[3]];

        // Overlap remaining 50 2B loads
        for (int p = NUM_KEY_PATTERNS; p < NUM_PATTERNS; ++p) {
            const int32_t flat = IS_WHITE ? (meta[p].nonkey_off_end - idx[p]) : (meta[p].nonkey_off + idx[p]);
            sum += nk_w[flat];
        }

        // 2-channel 60-ply PTM SCReLU (14 + 6 = 20 imuls!)
        const SCReLUPlyParams& sp = weights.screlu[ply];
        int32_t res_acc = 0;
        {
            const int32_t m0 = sp.mu[0][0], m1 = sp.mu[0][1];
            int32_t s0 = 0, s1 = 0;
            for (int r = 0; r < 4; ++r) {
                const int c1 = (r + 1) & 3;
                const int32_t u0 = clamp_i32(h0[4 + r] + h0[8 + r] + h0[r] + h0[c1] + m0, SCRELU_Q);
                const int32_t u1 = clamp_i32(h1[4 + r] + h1[8 + r] + h1[r] + h1[c1] + m1, SCRELU_Q);
                s0 += u0 * u0;
                s1 += u1 * u1;
            }
            res_acc += static_cast<int32_t>(sp.beta[0][0]) * s0
                     + static_cast<int32_t>(sp.beta[0][1]) * s1;
        }
        {
            const int32_t m0 = sp.mu[1][0], m1 = sp.mu[1][1];
            int32_t s0 = 0, s1 = 0;
            for (int d = 0; d < 2; ++d) {
                const int32_t u0 = clamp_i32(h0[12 + d] + h0[d] + h0[d + 2] + m0, SCRELU_Q);
                const int32_t u1 = clamp_i32(h1[12 + d] + h1[d] + h1[d + 2] + m1, SCRELU_Q);
                s0 += u0 * u0;
                s1 += u1 * u1;
            }
            res_acc += static_cast<int32_t>(sp.beta[1][0]) * s0
                     + static_cast<int32_t>(sp.beta[1][1]) * s1;
        }
        {
            const int32_t u0 = clamp_i32(g0 + sp.mu[2][0], SCRELU_Q);
            const int32_t u1 = clamp_i32(g1 + sp.mu[2][1], SCRELU_Q);
            res_acc += static_cast<int32_t>(sp.beta[2][0]) * (u0 * u0)
                     + static_cast<int32_t>(sp.beta[2][1]) * (u1 * u1);
        }

        return sum + a_lut + sym_rshift(res_acc, SCRELU_SHIFT);
    }
}

template<EvalMode MODE>
inline int32_t evaluate_p2c64(const P2C64Weights& weights,
                              const PatternMeta* __restrict meta,
                              const uint16_t* __restrict idx,
                              int stm, int ply) noexcept {
    if (stm < 0) return evaluate_p2c64_ptm<MODE, true>(weights, meta, idx, ply);
    return evaluate_p2c64_ptm<MODE, false>(weights, meta, idx, ply);
}

} // namespace p2c64
