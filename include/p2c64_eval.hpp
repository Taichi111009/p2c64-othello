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
    CAND_C_PTM_HYBRID     = 5,
    WACA_G2               = 6,
    WACA_G2_V31           = 7,
    WACA_G2_V32           = 8,
    WACA_G2_V33           = 9
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

// ============================================================================
// P2C-64 v3.0: WACA-G2 (MPC-PureV64 + SAFE_BND 2-Stage Cascade Global Eval)
//   - Stage 1 (Tier-1): Unmodified 64-pat x 60-ply PTM int16_t (46.92 MiB, 0 overhead!)
//   - Stage 2.5       : 16-dim D4xZ2 global mobility + geom linear gate (2.16 KiB, L1D)
//   - Stage 3         : Physically separated 20-block EO Key-14 int8[4] (7.01 MiB)
//                       + 60-ply Key-14 SCReLU head + 60-ply 24->H->1 SCReLU MLP (L1D)
// ============================================================================
static constexpr int     WACA_EO_BLOCKS    = 20;   // 10 blocks x 2 (even/odd ply parity!)
static constexpr int     WACA_F25          = 16;   // Stage 2.5 input dims
static constexpr int     WACA_F3           = 24;   // Stage 3 input dims (16 reused + 8 extra)
static constexpr int     WACA_MAX_HID      = 24;   // Max MLP hidden width (16 or 24)
static constexpr int32_t WACA_FEAT_CLAMP   = 512;  // Every feature clamped to [-512, +512]
static constexpr int32_t WACA_S25_W_MAX    = 16384;
static constexpr int32_t WACA_S25_B_MAX    = 1048576;
static constexpr int     WACA_S25_SHIFT    = 10;
static constexpr int32_t WACA_MLP_W1_MAX   = 127;
static constexpr int32_t WACA_MLP_B1_MAX   = 1048576;
static constexpr int     WACA_MLP_S1       = 9;
static constexpr int32_t WACA_MLP_W2_MAX   = 4095; // 24 * 127^2 * 4095 = 1.585e9 < INT32_MAX
static constexpr int32_t WACA_MLP_B2_MAX   = 1048576;
static constexpr int     WACA_MLP_S2       = 10;
static constexpr int32_t WACA_MAX_RES      = 1024; // +/- 8.0 discs wide guard (128 units/disc)

static constexpr std::array<uint8_t, NUM_PLIES> PLY_TO_EO_BLK = {
     0, 1, 0, 1, 0, 1,   2, 3, 2, 3, 2, 3,   4, 5, 4, 5, 4, 5,   6, 7, 6, 7, 6, 7,
     8, 9, 8, 9, 8, 9,  10,11,10,11,10,11,  12,13,12,13,12,13,  14,15,14,15,14,15,
    16,17,16,17,16,17,  18,19,18,19,18,19
};

static inline constexpr int32_t clamp_feat_i32(int32_t x) noexcept {
    return (x < -WACA_FEAT_CLAMP) ? -WACA_FEAT_CLAMP : ((x > WACA_FEAT_CLAMP) ? WACA_FEAT_CLAMP : x);
}

static inline constexpr int32_t abs_i32(int32_t x) noexcept {
    return (x < 0) ? -x : x;
}

enum class GateMode : int {
    SAFE_BND      = 0, // Return shrunk-margin sound bound (V+M / V-M)
    FAIL_HARD     = 1, // Return alpha / beta directly
    FULL          = 2, // Disable leaf gating (100% Stage 3 on non-MPC leaves)
    WIDE_SAFE_BND = 3  // Separate gate margin (M) from wide TT safety guard (std::min(alpha, V+G))
};

struct alignas(64) WacaStage25Ply {
    int16_t w[WACA_F25]{}; // 32B
    int32_t b = 0;         //  4B
};

struct alignas(64) WacaMlpPly {
    int8_t  w1[WACA_MAX_HID][WACA_F3]{}; // 576B
    int32_t b1[WACA_MAX_HID]{};          //  96B
    int16_t w2[WACA_MAX_HID]{};          //  48B
    int32_t b2 = 0;                      //   4B
};

struct WacaKeyEmb4 {
    int8_t h[4]{};
};
static_assert(sizeof(WacaKeyEmb4) == 4, "WacaKeyEmb4 must be 4 bytes");

struct WacaG2Weights {
    int32_t                                      mlp_hidden = 16; // 16 or 24
    RawNTuple64Weights                           tier1;           // 46.92 MiB untouched RAW_64!
    std::array<WacaStage25Ply, NUM_PLIES>        s25{};           // 2.16 KiB (L1D)
    std::array<std::vector<WacaKeyEmb4>, WACA_EO_BLOCKS> h_emb;   // 7.01 MiB (Stage 3 only!)
    std::array<SCReLUPlyParams, NUM_PLIES>       screlu{};        // 2.81 KiB (L1D)
    std::array<WacaMlpPly, NUM_PLIES>            mlp{};           // <= 44 KiB (L1D)

    WacaG2Weights() {
        const auto& pt = get_pattern_tables();
        for (int b = 0; b < WACA_EO_BLOCKS; ++b) {
            h_emb[b].assign(static_cast<size_t>(pt.key_raw_entries), WacaKeyEmb4{});
        }
    }

    // Strict D4 internal mirror symmetrization & overflow-safe parameter clamping
    void symmetrize() {
        tier1.symmetrize();
        const auto& pt = get_pattern_tables();
        auto clamp_i8 = [](int8_t v) -> int8_t { return (v == INT8_MIN) ? -127 : v; };

        for (int b = 0; b < WACA_EO_BLOCKS; ++b) {
            WacaKeyEmb4* tbl = h_emb[b].data();
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
                    WacaKeyEmb4 rep = tbl[off + x];
                    for (int k = 0; k < 4; ++k) rep.h[k] = clamp_i8(rep.h[k]);
                    for (int32_t idx : orb) tbl[off + idx] = rep;
                }
            }
        }

        mlp_hidden = std::clamp<int32_t>(mlp_hidden, 1, WACA_MAX_HID);
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            s25[ply].b = std::clamp<int32_t>(s25[ply].b, -WACA_S25_B_MAX, WACA_S25_B_MAX);
            for (int i = 0; i < WACA_F25; ++i) {
                s25[ply].w[i] = static_cast<int16_t>(
                    std::clamp<int32_t>(s25[ply].w[i], -WACA_S25_W_MAX, WACA_S25_W_MAX));
            }
            for (int g = 0; g < 3; ++g) {
                for (int k = 0; k < 4; ++k) {
                    screlu[ply].beta[g][k] = static_cast<int16_t>(
                        std::clamp<int32_t>(screlu[ply].beta[g][k], -SCRELU_BETA_MAX, SCRELU_BETA_MAX));
                }
            }
            mlp[ply].b2 = std::clamp<int32_t>(mlp[ply].b2, -WACA_MLP_B2_MAX, WACA_MLP_B2_MAX);
            for (int j = 0; j < WACA_MAX_HID; ++j) {
                mlp[ply].b1[j] = std::clamp<int32_t>(mlp[ply].b1[j], -WACA_MLP_B1_MAX, WACA_MLP_B1_MAX);
                mlp[ply].w2[j] = static_cast<int16_t>(
                    std::clamp<int32_t>(mlp[ply].w2[j], -WACA_MLP_W2_MAX, WACA_MLP_W2_MAX));
                for (int i = 0; i < WACA_F3; ++i) {
                    mlp[ply].w1[j][i] = clamp_i8(mlp[ply].w1[j][i]);
                }
            }
        }
    }
};

// Stage 1: Identical 20-insn 64-pattern PTM loop to RAW_NTUPLE_64 (0.0% overhead on gated leaves & MPC!)
template<bool IS_WHITE>
__attribute__((noinline, noclone))
int32_t eval_waca_stage1_ptm(const RawNTuple64Weights& tier1,
                             const PatternMeta* __restrict meta,
                             const uint16_t* __restrict idx,
                             int ply) noexcept {
    const int16_t* __restrict wp = tier1.w_ply[ply].data();
    int32_t sum = 0;
    for (int p = 0; p < NUM_PATTERNS; ++p) {
        const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        sum += wp[flat];
    }
    return sum;
}

// Deferred L1D-hot capture of 4 corner (p=0..3) and 4 edge (p=4..7) weights ONLY when Gate 1 passes (~15% leaves!)
template<bool IS_WHITE>
static inline void extract_waca_ce_ptm(const RawNTuple64Weights& tier1,
                                       const PatternMeta* __restrict meta,
                                       const uint16_t* __restrict idx,
                                       int ply,
                                       int32_t c_out[4],
                                       int32_t e_out[4]) noexcept {
    const int16_t* __restrict wp = tier1.w_ply[ply].data();
    for (int p = 0; p < 4; ++p) {
        const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        c_out[p] = wp[flat];
    }
    for (int p = 4; p < 8; ++p) {
        const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        e_out[p - 4] = wp[flat];
    }
}

// Stage 2.5 feature extraction helper (shared verbatim by inference & C++ training bridge!)
static inline void extract_waca_stage25_features(uint64_t P, uint64_t O,
                                                 int32_t v1,
                                                 const int32_t c[4],
                                                 const int32_t e[4],
                                                 int32_t f_out[WACA_F25]) noexcept {
    uint64_t mvP = 0, mvO = 0;
    generate_moves_pair(P, O, mvP, mvO);
    const int32_t mobP  = popcount64_swar(mvP);
    const int32_t mobO  = popcount64_swar(mvO);

    constexpr uint64_t EDGES = 0xff818181818181ffULL;
    const int32_t cmobP = popcount4_corners(mvP);
    const int32_t cmobO = popcount4_corners(mvO);
    const int32_t xmobP = popcount4_xsqs(mvP);
    const int32_t xmobO = popcount4_xsqs(mvO);
    const int32_t emobP = popcount64_swar(mvP & EDGES);
    const int32_t emobO = popcount64_swar(mvO & EDGES);

    const uint64_t E    = ~(P | O);
    const int32_t potP  = popcount64_swar(adj_mask_bb(O) & E);
    const int32_t potO  = popcount64_swar(adj_mask_bb(P) & E);
    const uint64_t adjE = adj_mask_bb(E);
    const int32_t frP   = popcount64_swar(P & adjE);
    const int32_t frO   = popcount64_swar(O & adjE);
    const int32_t dP    = popcount64_swar(P);
    const int32_t dO    = popcount64_swar(O);

    const int32_t c_sum = c[0] + c[1] + c[2] + c[3];
    const int32_t e_sum = e[0] + e[1] + e[2] + e[3];
    int32_t c_min = c[0], c_max = c[0], e_min = e[0], e_max = e[0];
    for (int i = 1; i < 4; ++i) {
        if (c[i] < c_min) c_min = c[i];
        if (c[i] > c_max) c_max = c[i];
        if (e[i] < e_min) e_min = e[i];
        if (e[i] > e_max) e_max = e[i];
    }

    int32_t c_sq = 0;
    for (int i = 0; i < 4; ++i) {
        const int32_t cs = std::clamp<int32_t>(sym_rshift(c[i], 4), -256, 256);
        c_sq += cs * cs;
    }

    f_out[0]  = clamp_feat_i32((mobP - mobO) << 4);
    f_out[1]  = clamp_feat_i32(((mobP + mobO) - 16) << 3);
    f_out[2]  = ((mobP == 0) ? 128 : 0) - ((mobO == 0) ? 128 : 0);
    f_out[3]  = clamp_feat_i32((cmobP - cmobO) << 6);
    f_out[4]  = clamp_feat_i32((xmobP - xmobO) << 6);
    f_out[5]  = clamp_feat_i32((potP - potO) << 4);
    f_out[6]  = clamp_feat_i32((frP - frO) << 4);
    f_out[7]  = clamp_feat_i32((dP - dO) << 3);
    f_out[8]  = clamp_feat_i32(sym_rshift(v1, 4));
    f_out[9]  = clamp_feat_i32(sym_rshift(v1, 2));
    f_out[10] = clamp_feat_i32(sym_rshift(c_sum, 3));
    f_out[11] = clamp_feat_i32(sym_rshift(e_sum, 3));
    f_out[12] = clamp_feat_i32(sym_rshift(c_max - c_min, 3));
    f_out[13] = clamp_feat_i32(sym_rshift(e_max - e_min, 3));
    f_out[14] = clamp_feat_i32((emobP - emobO) << 4);
    f_out[15] = clamp_feat_i32(sym_rshift(c_sq, 7));
}

// Stage 3 extra 4 bitboard/geometric features f[16..19] (shared by inference & C++ bridge!)
__attribute__((always_inline))
static inline void extract_waca_stage3_geom_features(uint64_t P, uint64_t O,
                                                     const int32_t c[4],
                                                     const int32_t e[4],
                                                     int32_t f_extra[4]) noexcept {
    const int32_t stabP = count_edge_stable_bb(P);
    const int32_t stabO = count_edge_stable_bb(O);

    int32_t cs[4], es[4];
    for (int i = 0; i < 4; ++i) {
        cs[i] = std::clamp<int32_t>(sym_rshift(c[i], 4), -256, 256);
        es[i] = std::clamp<int32_t>(sym_rshift(e[i], 4), -256, 256);
    }
    const int32_t diag_prod = cs[0] * cs[2] + cs[1] * cs[3];
    const int32_t adj_diff  = abs_i32(cs[0] - cs[1]) + abs_i32(cs[1] - cs[2])
                            + abs_i32(cs[2] - cs[3]) + abs_i32(cs[3] - cs[0]);
    const int32_t diag_diff = abs_i32(cs[0] - cs[2]) + abs_i32(cs[1] - cs[3]);
    const int32_t polar     = adj_diff - 2 * diag_diff;
    const int32_t ce_coup   = cs[0] * (es[0] + es[3]) + cs[1] * (es[1] + es[0])
                            + cs[2] * (es[2] + es[1]) + cs[3] * (es[3] + es[2]);

    f_extra[0] = clamp_feat_i32((stabP - stabO) << 4);
    f_extra[1] = clamp_feat_i32(sym_rshift(diag_prod, 6));
    f_extra[2] = clamp_feat_i32(sym_rshift(polar, 1));
    f_extra[3] = clamp_feat_i32(sym_rshift(ce_coup, 7));
}

// Stage 2.5 Kernel: Fast Global Mobility + Geom Linear Residual (~50 ns, 0 SIMD, 0 PLT calls!)
__attribute__((noinline, noclone))
inline int32_t eval_waca_stage25_kernel(const WacaG2Weights& w,
                                        uint64_t P, uint64_t O,
                                        int32_t v1,
                                        const int32_t c[4], const int32_t e[4],
                                        int ply,
                                        int32_t f_out[WACA_F25]) noexcept {
    extract_waca_stage25_features(P, O, v1, c, e, f_out);
    const WacaStage25Ply& sp = w.s25[ply];
    int32_t a0 = sp.b, a1 = 0, a2 = 0, a3 = 0;
    for (int i = 0; i < WACA_F25; i += 4) {
        a0 += static_cast<int32_t>(sp.w[i + 0]) * f_out[i + 0];
        a1 += static_cast<int32_t>(sp.w[i + 1]) * f_out[i + 1];
        a2 += static_cast<int32_t>(sp.w[i + 2]) * f_out[i + 2];
        a3 += static_cast<int32_t>(sp.w[i + 3]) * f_out[i + 3];
    }
    const int32_t acc = (a0 + a1) + (a2 + a3);
    return std::clamp<int32_t>(sym_rshift(acc, WACA_S25_SHIFT), -WACA_MAX_RES, WACA_MAX_RES);
}

// Stage 3 Kernel: 20-Block EO Key-14 4ch SCReLU + 60-Ply 24->H->1 Integer SCReLU MLP (0 SIMD!)
template<bool IS_WHITE>
__attribute__((noinline, noclone))
int32_t eval_waca_stage3_ptm(const WacaG2Weights& w,
                             const PatternMeta* __restrict meta,
                             const uint16_t* __restrict idx,
                             uint64_t P, uint64_t O,
                             const int32_t c[4], const int32_t e[4],
                             const int32_t* __restrict f25,
                             int32_t r25,
                             int ply) noexcept {
    int32_t f_ext[8];
    extract_waca_stage3_geom_features(P, O, c, e, f_ext);

    // 1) Physically separated 20-block Even/Odd Key-14 4ch embeddings (ONLY touched on ~5% leaves!)
    const uint8_t eo_blk = PLY_TO_EO_BLK[ply];
    const WacaKeyEmb4* __restrict htbl = w.h_emb[eo_blk].data();
    int32_t h0[NUM_KEY_PATTERNS], h1[NUM_KEY_PATTERNS];
    int32_t h2[NUM_KEY_PATTERNS], h3[NUM_KEY_PATTERNS];
    int32_t g0 = 0, g1 = 0, g2 = 0, g3 = 0;
    for (int p = 0; p < NUM_KEY_PATTERNS; ++p) {
        const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        const WacaKeyEmb4 emb = htbl[flat];
        const int32_t v0 = emb.h[0], v1 = emb.h[1], v2 = emb.h[2], v3 = emb.h[3];
        h0[p] = v0; h1[p] = v1; h2[p] = v2; h3[p] = v3;
        g0 += v0;   g1 += v1;   g2 += v2;   g3 += v3;
    }

    const SCReLUPlyParams& sc = w.screlu[ply];
    int32_t s0_0 = 0, s0_1 = 0, s0_2 = 0, s0_3 = 0;
    {
        const int32_t m0 = sc.mu[0][0], m1 = sc.mu[0][1], m2 = sc.mu[0][2], m3 = sc.mu[0][3];
        for (int r = 0; r < 4; ++r) {
            const int c1 = (r + 1) & 3;
            const int32_t u0 = clamp_i32(h0[4 + r] + h0[8 + r] + h0[r] + h0[c1] + m0, SCRELU_Q);
            const int32_t u1 = clamp_i32(h1[4 + r] + h1[8 + r] + h1[r] + h1[c1] + m1, SCRELU_Q);
            const int32_t u2 = clamp_i32(h2[4 + r] + h2[8 + r] + h2[r] + h2[c1] + m2, SCRELU_Q);
            const int32_t u3 = clamp_i32(h3[4 + r] + h3[8 + r] + h3[r] + h3[c1] + m3, SCRELU_Q);
            s0_0 += u0 * u0; s0_1 += u1 * u1; s0_2 += u2 * u2; s0_3 += u3 * u3;
        }
    }
    int32_t s1_0 = 0, s1_1 = 0, s1_2 = 0, s1_3 = 0;
    {
        const int32_t m0 = sc.mu[1][0], m1 = sc.mu[1][1], m2 = sc.mu[1][2], m3 = sc.mu[1][3];
        for (int d = 0; d < 2; ++d) {
            const int32_t u0 = clamp_i32(h0[12 + d] + h0[d] + h0[d + 2] + m0, SCRELU_Q);
            const int32_t u1 = clamp_i32(h1[12 + d] + h1[d] + h1[d + 2] + m1, SCRELU_Q);
            const int32_t u2 = clamp_i32(h2[12 + d] + h2[d] + h2[d + 2] + m2, SCRELU_Q);
            const int32_t u3 = clamp_i32(h3[12 + d] + h3[d] + h3[d + 2] + m3, SCRELU_Q);
            s1_0 += u0 * u0; s1_1 += u1 * u1; s1_2 += u2 * u2; s1_3 += u3 * u3;
        }
    }
    const int32_t ug0 = clamp_i32(g0 + sc.mu[2][0], SCRELU_Q);
    const int32_t ug1 = clamp_i32(g1 + sc.mu[2][1], SCRELU_Q);
    const int32_t ug2 = clamp_i32(g2 + sc.mu[2][2], SCRELU_Q);
    const int32_t ug3 = clamp_i32(g3 + sc.mu[2][3], SCRELU_Q);
    const int32_t s2_0 = ug0 * ug0, s2_1 = ug1 * ug1, s2_2 = ug2 * ug2, s2_3 = ug3 * ug3;

    // Direct v2.0-compatible Key-14 SCReLU residual
    const int32_t sc_acc =
          static_cast<int32_t>(sc.beta[0][0]) * s0_0 + static_cast<int32_t>(sc.beta[0][1]) * s0_1
        + static_cast<int32_t>(sc.beta[0][2]) * s0_2 + static_cast<int32_t>(sc.beta[0][3]) * s0_3
        + static_cast<int32_t>(sc.beta[1][0]) * s1_0 + static_cast<int32_t>(sc.beta[1][1]) * s1_1
        + static_cast<int32_t>(sc.beta[1][2]) * s1_2 + static_cast<int32_t>(sc.beta[1][3]) * s1_3
        + static_cast<int32_t>(sc.beta[2][0]) * s2_0 + static_cast<int32_t>(sc.beta[2][1]) * s2_1
        + static_cast<int32_t>(sc.beta[2][2]) * s2_2 + static_cast<int32_t>(sc.beta[2][3]) * s2_3;
    const int32_t r_key_screlu = sym_rshift(sc_acc, SCRELU_SHIFT);

    // 4-channel Key-14 nonlinear summary features fed into the MLP [f20..f23]
    f_ext[4] = clamp_feat_i32((s0_0 + s1_0 + s2_0) >> 8);
    f_ext[5] = clamp_feat_i32((s0_1 + s1_1 + s2_1) >> 8);
    f_ext[6] = clamp_feat_i32((s0_2 + s1_2 + s2_2) >> 8);
    f_ext[7] = clamp_feat_i32((s0_3 + s1_3 + s2_3) >> 8);

    // 2) 60-ply Pure-Integer SCReLU MLP (24 -> H -> 1, 4-way unrolled scalar ILP, overflow-proof in int32_t!)
    const WacaMlpPly& mp = w.mlp[ply];
    const int H = w.mlp_hidden;
    int32_t acc2 = mp.b2;
    for (int j = 0; j < H; ++j) {
        const int8_t* __restrict w1j = mp.w1[j];
        int32_t z0 = mp.b1[j], z1 = 0, z2 = 0, z3 = 0;
        for (int i = 0; i < WACA_F25; i += 4) {
            z0 += static_cast<int32_t>(w1j[i + 0]) * f25[i + 0];
            z1 += static_cast<int32_t>(w1j[i + 1]) * f25[i + 1];
            z2 += static_cast<int32_t>(w1j[i + 2]) * f25[i + 2];
            z3 += static_cast<int32_t>(w1j[i + 3]) * f25[i + 3];
        }
        z0 += static_cast<int32_t>(w1j[16]) * f_ext[0] + static_cast<int32_t>(w1j[20]) * f_ext[4];
        z1 += static_cast<int32_t>(w1j[17]) * f_ext[1] + static_cast<int32_t>(w1j[21]) * f_ext[5];
        z2 += static_cast<int32_t>(w1j[18]) * f_ext[2] + static_cast<int32_t>(w1j[22]) * f_ext[6];
        z3 += static_cast<int32_t>(w1j[19]) * f_ext[3] + static_cast<int32_t>(w1j[23]) * f_ext[7];
        const int32_t z = (z0 + z1) + (z2 + z3);
        const int32_t u = (z <= 0) ? 0 : (((z >> WACA_MLP_S1) > SCRELU_Q) ? SCRELU_Q : (z >> WACA_MLP_S1));
        acc2 += static_cast<int32_t>(mp.w2[j]) * (u * u);
    }
    const int32_t r_mlp = sym_rshift(acc2, WACA_MLP_S2);

    return std::clamp<int32_t>(r25 + r_mlp + r_key_screlu, -WACA_MAX_RES, WACA_MAX_RES);
}

// Evaluate all 3 stages unconditionally (used for bit-exact testing, Val RMSE, and GateMode::FULL)
inline void eval_waca_g2_stages(const WacaG2Weights& w,
                                const PatternMeta* __restrict meta,
                                const uint16_t* __restrict idx,
                                uint64_t B, uint64_t W, int stm, int ply,
                                int32_t& out_v1, int32_t& out_v25, int32_t& out_v3) noexcept {
    const bool is_w = (stm < 0);
    const uint64_t P = is_w ? W : B;
    const uint64_t O = is_w ? B : W;
    int32_t c[4], e[4], f25[WACA_F25];
    const int32_t v1 = is_w ? eval_waca_stage1_ptm<true>(w.tier1, meta, idx, ply)
                            : eval_waca_stage1_ptm<false>(w.tier1, meta, idx, ply);
    if (is_w) extract_waca_ce_ptm<true>(w.tier1, meta, idx, ply, c, e);
    else      extract_waca_ce_ptm<false>(w.tier1, meta, idx, ply, c, e);
    const int32_t r25 = eval_waca_stage25_kernel(w, P, O, v1, c, e, ply, f25);
    const int32_t rt  = is_w ? eval_waca_stage3_ptm<true>(w, meta, idx, P, O, c, e, f25, r25, ply)
                             : eval_waca_stage3_ptm<false>(w, meta, idx, P, O, c, e, f25, r25, ply);
    out_v1  = v1;
    out_v25 = v1 + r25;
    out_v3  = v1 + rt;
}

inline int32_t eval_waca_g2_full(const WacaG2Weights& w,
                                 const PatternMeta* __restrict meta,
                                 const uint16_t* __restrict idx,
                                 uint64_t B, uint64_t W, int stm, int ply) noexcept {
    int32_t v1 = 0, v25 = 0, v3 = 0;
    eval_waca_g2_stages(w, meta, idx, B, W, stm, ply, v1, v25, v3);
    return v3;
}

static inline constexpr int32_t gate_return_low(GateMode gm, int32_t v, int32_t m_units,
                                                int32_t tt_guard_units, int32_t alpha) noexcept {
    if (gm == GateMode::SAFE_BND)      return v + m_units;
    if (gm == GateMode::WIDE_SAFE_BND) return std::min(alpha, v + tt_guard_units);
    return alpha;
}

static inline constexpr int32_t gate_return_high(GateMode gm, int32_t v, int32_t m_units,
                                                 int32_t tt_guard_units, int32_t beta) noexcept {
    if (gm == GateMode::SAFE_BND)      return v - m_units;
    if (gm == GateMode::WIDE_SAFE_BND) return std::max(beta, v - tt_guard_units);
    return beta;
}

// Search-integrated 2-stage SAFE_BND / FAIL_HARD / FULL / WIDE_SAFE_BND gated evaluator (v3.0)
inline int32_t eval_waca_g2_gated(const WacaG2Weights& w,
                                  const PatternMeta* __restrict meta,
                                  const uint16_t* __restrict idx,
                                  uint64_t B, uint64_t W, int stm, int ply,
                                  int32_t alpha, int32_t beta,
                                  bool in_mpc_probe,
                                  GateMode gate_mode,
                                  int32_t m25_units,
                                  int32_t m3_units,
                                  uint64_t& stat_s25,
                                  uint64_t& stat_s3,
                                  int32_t tt_guard_units = 576) noexcept {
    const bool is_w = (stm < 0);
    const int32_t v1 = is_w ? eval_waca_stage1_ptm<true>(w.tier1, meta, idx, ply)
                            : eval_waca_stage1_ptm<false>(w.tier1, meta, idx, ply);

    // MPC-PureV64: unconditionally return exact Stage-1 V1 during shallow MPC probe when enabled!
    if (in_mpc_probe) return v1;

    const bool is_pv = (beta > alpha + 1);
    if (!is_pv && gate_mode != GateMode::FULL) {
        if (v1 + m25_units <= alpha) return gate_return_low(gate_mode, v1, m25_units, tt_guard_units, alpha);
        if (v1 - m25_units >= beta)  return gate_return_high(gate_mode, v1, m25_units, tt_guard_units, beta);
    }

    // Stage 2.5 (~14-16% of main-search leaves + PV)
    stat_s25++;
    int32_t c[4], e[4], f25[WACA_F25];
    if (is_w) extract_waca_ce_ptm<true>(w.tier1, meta, idx, ply, c, e);
    else      extract_waca_ce_ptm<false>(w.tier1, meta, idx, ply, c, e);
    const uint64_t P = is_w ? W : B;
    const uint64_t O = is_w ? B : W;
    const int32_t r25 = eval_waca_stage25_kernel(w, P, O, v1, c, e, ply, f25);
    const int32_t v25 = v1 + r25;

    if (!is_pv && gate_mode != GateMode::FULL) {
        if (v25 + m3_units <= alpha) return gate_return_low(gate_mode, v25, m3_units, tt_guard_units, alpha);
        if (v25 - m3_units >= beta)  return gate_return_high(gate_mode, v25, m3_units, tt_guard_units, beta);
    }

    // Stage 3 (~4.5-6.0% of main-search leaves + PV)
    stat_s3++;
    const int32_t rt = is_w ? eval_waca_stage3_ptm<true>(w, meta, idx, P, O, c, e, f25, r25, ply)
                            : eval_waca_stage3_ptm<false>(w, meta, idx, P, O, c, e, f25, r25, ply);
    return v1 + rt;
}

// ============================================================================
// P2C-64 v3.1: WACA_G2_V31 (Slim 8-Dim Stage 2.5 + 32-Dim Long-Term Stage 3
//   with 4-Neighbor Empty Parity & 5-Zone 64-Pat Decomposition + 30-Blk EO Key-14)
// ============================================================================
static constexpr int WACA_V31_EO_BLOCKS = 30; // 15 x 2 (even/odd ply parity, max +/-1 ply drift!)
static constexpr int WACA_V31_F25       = 8;  // Ultra-light Stage 2.5 input dims (3 SWAR popcounts, 8 imul!)
static constexpr int WACA_V31_F3_EXTRA  = 20; // Stage 3 extra bitboard + parity + 5-zone features f[8..27]
static constexpr int WACA_V31_F3        = 32; // Stage 3 total MLP input dims (8 + 20 + 4 Key-14 SCReLU)
static constexpr int WACA_V31_MAX_HID   = 24; // 16 (fast default) or 24

static constexpr std::array<uint8_t, NUM_PLIES> PLY_TO_EO30_BLK = {
     0, 1, 0, 1,   2, 3, 2, 3,   4, 5, 4, 5,   6, 7, 6, 7,   8, 9, 8, 9,
    10,11,10,11,  12,13,12,13,  14,15,14,15,  16,17,16,17,  18,19,18,19,
    20,21,20,21,  22,23,22,23,  24,25,24,25,  26,27,26,27,  28,29,28,29
};

struct alignas(32) WacaV31Stage25Ply {
    int16_t w[WACA_V31_F25]{}; // 16B
    int32_t b = 0;             //  4B
};

struct alignas(64) WacaV31MlpPly {
    int8_t  w1[WACA_V31_MAX_HID][WACA_V31_F3]{}; // 768B
    int32_t b1[WACA_V31_MAX_HID]{};              //  96B
    int16_t w2[WACA_V31_MAX_HID]{};              //  48B
    int32_t b2 = 0;                              //   4B
};

struct WacaG2V31Weights {
    int32_t                                          mlp_hidden = 16; // 16 or 24
    RawNTuple64Weights                               tier1;           // 46.92 MiB untouched RAW_64!
    std::array<WacaV31Stage25Ply, NUM_PLIES>         s25{};           //  1.92 KiB (L1D)
    std::array<std::vector<WacaKeyEmb4>, WACA_V31_EO_BLOCKS> h_emb;   // 10.51 MiB (Stage 3 only!)
    std::array<SCReLUPlyParams, NUM_PLIES>           screlu{};        //  2.81 KiB (L1D)
    std::array<WacaV31MlpPly, NUM_PLIES>             mlp{};           // <= 57 KiB (L1D/L2)

    WacaG2V31Weights() {
        const auto& pt = get_pattern_tables();
        for (int b = 0; b < WACA_V31_EO_BLOCKS; ++b) {
            h_emb[b].assign(static_cast<size_t>(pt.key_raw_entries), WacaKeyEmb4{});
        }
    }

    // Strict D4 internal mirror symmetrization & overflow-safe parameter clamping
    void symmetrize() {
        tier1.symmetrize();
        const auto& pt = get_pattern_tables();
        auto clamp_i8 = [](int8_t v) -> int8_t { return (v == INT8_MIN) ? -127 : v; };

        for (int b = 0; b < WACA_V31_EO_BLOCKS; ++b) {
            WacaKeyEmb4* tbl = h_emb[b].data();
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
                    WacaKeyEmb4 rep = tbl[off + x];
                    for (int k = 0; k < 4; ++k) rep.h[k] = clamp_i8(rep.h[k]);
                    for (int32_t idx : orb) tbl[off + idx] = rep;
                }
            }
        }

        mlp_hidden = std::clamp<int32_t>(mlp_hidden, 1, WACA_V31_MAX_HID);
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            s25[ply].b = std::clamp<int32_t>(s25[ply].b, -WACA_S25_B_MAX, WACA_S25_B_MAX);
            for (int i = 0; i < WACA_V31_F25; ++i) {
                s25[ply].w[i] = static_cast<int16_t>(
                    std::clamp<int32_t>(s25[ply].w[i], -WACA_S25_W_MAX, WACA_S25_W_MAX));
            }
            for (int g = 0; g < 3; ++g) {
                for (int k = 0; k < 4; ++k) {
                    screlu[ply].beta[g][k] = static_cast<int16_t>(
                        std::clamp<int32_t>(screlu[ply].beta[g][k], -SCRELU_BETA_MAX, SCRELU_BETA_MAX));
                }
            }
            mlp[ply].b2 = std::clamp<int32_t>(mlp[ply].b2, -WACA_MLP_B2_MAX, WACA_MLP_B2_MAX);
            for (int j = 0; j < WACA_V31_MAX_HID; ++j) {
                mlp[ply].b1[j] = std::clamp<int32_t>(mlp[ply].b1[j], -WACA_MLP_B1_MAX, WACA_MLP_B1_MAX);
                mlp[ply].w2[j] = static_cast<int16_t>(
                    std::clamp<int32_t>(mlp[ply].w2[j], -WACA_MLP_W2_MAX, WACA_MLP_W2_MAX));
                for (int i = 0; i < WACA_V31_F3; ++i) {
                    mlp[ply].w1[j][i] = clamp_i8(mlp[ply].w1[j][i]);
                }
            }
        }
    }
};

// v3.1 Stage 1: Identical 20-insn 64-pattern PTM loop (0.0% overhead on gated leaves!)
template<bool IS_WHITE>
__attribute__((noinline, noclone))
int32_t eval_waca_v31_stage1_ptm(const RawNTuple64Weights& tier1,
                                 const PatternMeta* __restrict meta,
                                 const uint16_t* __restrict idx,
                                 int ply) noexcept {
    const int16_t* __restrict wp = tier1.w_ply[ply].data();
    int32_t sum = 0;
    for (int p = 0; p < NUM_PATTERNS; ++p) {
        const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        sum += wp[flat];
    }
    return sum;
}

// v3.1 Stage 2.5 feature extraction: 8 dims, 0 adj_mask_bb, 0 Tier-1 reloads, ONLY 3 popcount64_swar!
// Outputs mvP, mvO for zero-recomputation reuse in Stage 3.
__attribute__((always_inline))
static inline void extract_waca_v31_stage25_features(uint64_t P, uint64_t O,
                                                     int32_t v1, int ply,
                                                     int32_t f_out[WACA_V31_F25],
                                                     uint64_t& out_mvP,
                                                     uint64_t& out_mvO) noexcept {
    uint64_t mvP = 0, mvO = 0;
    generate_moves_pair(P, O, mvP, mvO);
    out_mvP = mvP;
    out_mvO = mvO;

    const int32_t mobP  = popcount64_swar(mvP);
    const int32_t mobO  = popcount64_swar(mvO);
    const int32_t cmobP = popcount4_corners(mvP);
    const int32_t cmobO = popcount4_corners(mvO);
    const int32_t xmobP = popcount4_xsqs(mvP);
    const int32_t xmobO = popcount4_xsqs(mvO);
    const int32_t dP    = popcount64_swar(P);

    f_out[0] = clamp_feat_i32((mobP - mobO) << 4);
    f_out[1] = clamp_feat_i32(((mobP + mobO) - 16) << 3);
    f_out[2] = ((mobP == 0) ? 128 : 0) - ((mobO == 0) ? 128 : 0);
    f_out[3] = clamp_feat_i32((cmobP - cmobO) << 6);
    f_out[4] = clamp_feat_i32((xmobP - xmobO) << 6);
    f_out[5] = clamp_feat_i32((2 * dP - (ply + 4)) << 3);
    f_out[6] = clamp_feat_i32(sym_rshift(v1, 4));
    f_out[7] = clamp_feat_i32(sym_rshift(v1, 2));
}

// v3.1 Stage 2.5 Kernel: Ultra-Light 8-Dim Linear Gate (~55-65 ns, 0 SIMD, 0 PLT calls!)
__attribute__((noinline, noclone))
inline int32_t eval_waca_v31_stage25_kernel(const WacaG2V31Weights& w,
                                            uint64_t P, uint64_t O,
                                            int32_t v1, int ply,
                                            int32_t f_out[WACA_V31_F25],
                                            uint64_t& out_mvP,
                                            uint64_t& out_mvO) noexcept {
    extract_waca_v31_stage25_features(P, O, v1, ply, f_out, out_mvP, out_mvO);
    const WacaV31Stage25Ply& sp = w.s25[ply];
    const int32_t a0 = sp.b
                     + static_cast<int32_t>(sp.w[0]) * f_out[0]
                     + static_cast<int32_t>(sp.w[4]) * f_out[4];
    const int32_t a1 = static_cast<int32_t>(sp.w[1]) * f_out[1]
                     + static_cast<int32_t>(sp.w[5]) * f_out[5];
    const int32_t a2 = static_cast<int32_t>(sp.w[2]) * f_out[2]
                     + static_cast<int32_t>(sp.w[6]) * f_out[6];
    const int32_t a3 = static_cast<int32_t>(sp.w[3]) * f_out[3]
                     + static_cast<int32_t>(sp.w[7]) * f_out[7];
    const int32_t acc = (a0 + a1) + (a2 + a3);
    return std::clamp<int32_t>(sym_rshift(acc, WACA_S25_SHIFT), -WACA_MAX_RES, WACA_MAX_RES);
}

// v3.1 Stage 3: Single-pass L1D-hot 5-zone geometric decomposition of all 64 Tier-1 patterns
//   Zone 1: c[0..3]     (p = 0..3:  Corner 3x3)
//   Zone 2: e[0..3]     (p = 4..7:  Edge+2X)
//   Zone 3: inner_sum   (p = 8..21: Line2, Diag8, Line3, Line4 + p = 62..63: Center 2x4)
//   Zone 4: subdiag_sum (p = 22..37: Diag 7, 6, 5, 4)
//   Zone 5: outer10_sum = v1 - (c_sum + e_sum + inner_sum + subdiag_sum) [p = 38..61, 0 extra loads!]
template<bool IS_WHITE>
__attribute__((always_inline))
static inline void extract_waca_v31_zones_ptm(const RawNTuple64Weights& tier1,
                                              const PatternMeta* __restrict meta,
                                              const uint16_t* __restrict idx,
                                              int ply, int32_t v1,
                                              int32_t c_out[4], int32_t e_out[4],
                                              int32_t& out_inner_sum,
                                              int32_t& out_subdiag_sum,
                                              int32_t& out_outer10_sum) noexcept {
    const int16_t* __restrict wp = tier1.w_ply[ply].data();
    int32_t c_sum = 0, e_sum = 0, inner_sum = 0, subdiag_sum = 0;
    for (int p = 0; p < 4; ++p) {
        const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        const int32_t wv = wp[flat];
        c_out[p] = wv;
        c_sum += wv;
    }
    for (int p = 4; p < 8; ++p) {
        const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        const int32_t wv = wp[flat];
        e_out[p - 4] = wv;
        e_sum += wv;
    }
    for (int p = 8; p < 22; ++p) {
        const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        inner_sum += wp[flat];
    }
    for (int p = 22; p < 38; ++p) {
        const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        subdiag_sum += wp[flat];
    }
    for (int p = 62; p < 64; ++p) {
        const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        inner_sum += wp[flat];
    }
    out_inner_sum   = inner_sum;
    out_subdiag_sum = subdiag_sum;
    out_outer10_sum = v1 - (c_sum + e_sum + inner_sum + subdiag_sum);
}

// v3.1 Stage 3 extra 20 features f[8..27] (shared verbatim by C++ inference & OpenMP training bridge!)
//   f_ext[0..3]  = f[8..11]  : edge mobility, potential mobility, frontier, corner-edge+triangle stability
//   f_ext[4..7]  = f[12..15] : 4-neighbor connected empty-region parity & board fragmentation (ply >= 24)
//   f_ext[8..19] = f[16..27] : 5-zone 64-pattern geometric decomposition + 4-quadrant symmetric polynomials
__attribute__((always_inline))
static inline void extract_waca_v31_stage3_extra_features(uint64_t P, uint64_t O,
                                                          uint64_t mvP, uint64_t mvO,
                                                          int ply,
                                                          const int32_t c[4], const int32_t e[4],
                                                          int32_t inner_sum,
                                                          int32_t subdiag_sum,
                                                          int32_t outer10_sum,
                                                          int32_t f_ext[WACA_V31_F3_EXTRA]) noexcept {
    // f[8..11]: Bitboard features reusing mvP/mvO from Stage 2.5
    constexpr uint64_t EDGES = 0xff818181818181ffULL;
    const int32_t emobP = popcount64_swar(mvP & EDGES);
    const int32_t emobO = popcount64_swar(mvO & EDGES);
    const uint64_t E    = ~(P | O);
    const int32_t potP  = popcount64_swar(adj_mask_bb(O) & E);
    const int32_t potO  = popcount64_swar(adj_mask_bb(P) & E);
    const uint64_t adjE = adj_mask_bb(E);
    const int32_t frP   = popcount64_swar(P & adjE);
    const int32_t frO   = popcount64_swar(O & adjE);
    const int32_t stabP = count_edge_stable_bb(P);
    const int32_t stabO = count_edge_stable_bb(O);

    f_ext[0] = clamp_feat_i32((emobP - emobO) << 4);
    f_ext[1] = clamp_feat_i32((potP - potO) << 4);
    f_ext[2] = clamp_feat_i32((frP - frO) << 4);
    f_ext[3] = clamp_feat_i32((stabP - stabO) << 4);

    // f[12..15]: 4-neighbor connected empty-region parity & board fragmentation (0 cost when ply < 24!)
    int32_t odd_cnt = 0, single_cnt = 0, odd_mv_diff = 0, reg_cnt = 0;
    if (ply < 24) {
        odd_cnt     = (60 - ply) & 1;
        single_cnt  = 0;
        odd_mv_diff = 0;
        reg_cnt     = 1;
    } else {
        // Fast-path: extract all isolated 1-empty regions in O(1) without Flood-Fill iterations
        const uint64_t adjE4 = ((E << 1) & 0xfefefefefefefefeULL)
                             | ((E >> 1) & 0x7f7f7f7f7f7f7f7fULL)
                             | (E << 8)
                             | (E >> 8);
        const uint64_t singles = E & ~adjE4;
        if (singles) {
            single_cnt  = popcount64_swar(singles);
            odd_cnt     = single_cnt;
            reg_cnt     = single_cnt;
            odd_mv_diff = popcount64_swar(singles & mvP) - popcount64_swar(singles & mvO);
        }
        uint64_t rem = E ^ singles;
        while (rem) {
            uint64_t comp = rem & (~rem + 1ULL);
            uint64_t prev;
            do {
                prev = comp;
                comp |= (((comp << 1) & 0xfefefefefefefefeULL)
                       | ((comp >> 1) & 0x7f7f7f7f7f7f7f7fULL)
                       | (comp << 8)
                       | (comp >> 8)) & rem;
                comp |= (((comp << 1) & 0xfefefefefefefefeULL)
                       | ((comp >> 1) & 0x7f7f7f7f7f7f7f7fULL)
                       | (comp << 8)
                       | (comp >> 8)) & rem;
            } while (comp != prev);
            rem ^= comp;
            reg_cnt++;
            const int32_t sz = (reg_cnt == 1 && rem == 0ULL) ? (60 - ply) : popcount64_swar(comp);
            if (sz & 1) {
                odd_cnt++;
                odd_mv_diff += ((comp & mvP) != 0ULL ? 1 : 0) - ((comp & mvO) != 0ULL ? 1 : 0);
            }
        }
    }
    f_ext[4] = clamp_feat_i32((odd_cnt - 1) << 5);
    f_ext[5] = clamp_feat_i32(single_cnt << 6);
    f_ext[6] = clamp_feat_i32(odd_mv_diff << 6);
    f_ext[7] = clamp_feat_i32((reg_cnt - 2) << 5);

    // f[16..27]: Full 64-pattern 5-zone geometric decomposition + 4-quadrant symmetric polynomials
    const int32_t c_sum = c[0] + c[1] + c[2] + c[3];
    const int32_t e_sum = e[0] + e[1] + e[2] + e[3];
    int32_t c_min = c[0], c_max = c[0], e_min = e[0], e_max = e[0];
    for (int i = 1; i < 4; ++i) {
        if (c[i] < c_min) c_min = c[i];
        if (c[i] > c_max) c_max = c[i];
        if (e[i] < e_min) e_min = e[i];
        if (e[i] > e_max) e_max = e[i];
    }

    int32_t cs[4], es[4];
    int32_t c_sq = 0;
    for (int i = 0; i < 4; ++i) {
        cs[i] = std::clamp<int32_t>(sym_rshift(c[i], 4), -256, 256);
        es[i] = std::clamp<int32_t>(sym_rshift(e[i], 4), -256, 256);
        c_sq += cs[i] * cs[i];
    }
    const int32_t diag_prod = cs[0] * cs[2] + cs[1] * cs[3];
    const int32_t adj_diff  = abs_i32(cs[0] - cs[1]) + abs_i32(cs[1] - cs[2])
                            + abs_i32(cs[2] - cs[3]) + abs_i32(cs[3] - cs[0]);
    const int32_t diag_diff = abs_i32(cs[0] - cs[2]) + abs_i32(cs[1] - cs[3]);
    const int32_t polar     = adj_diff - 2 * diag_diff;
    const int32_t ce_coup   = cs[0] * (es[0] + es[3]) + cs[1] * (es[1] + es[0])
                            + cs[2] * (es[2] + es[1]) + cs[3] * (es[3] + es[2]);

    f_ext[8]  = clamp_feat_i32(sym_rshift(c_sum, 3));
    f_ext[9]  = clamp_feat_i32(sym_rshift(e_sum, 3));
    f_ext[10] = clamp_feat_i32(sym_rshift(c_max - c_min, 3));
    f_ext[11] = clamp_feat_i32(sym_rshift(e_max - e_min, 3));
    f_ext[12] = clamp_feat_i32(sym_rshift(c_sq, 7));
    f_ext[13] = clamp_feat_i32(sym_rshift(diag_prod, 6));
    f_ext[14] = clamp_feat_i32(sym_rshift(polar, 1));
    f_ext[15] = clamp_feat_i32(sym_rshift(ce_coup, 7));
    f_ext[16] = clamp_feat_i32(sym_rshift(inner_sum, 3));
    f_ext[17] = clamp_feat_i32(sym_rshift(subdiag_sum, 3));
    f_ext[18] = clamp_feat_i32(sym_rshift(outer10_sum, 3));
    f_ext[19] = clamp_feat_i32(sym_rshift(inner_sum - (c_sum + e_sum + outer10_sum), 4));
}

// v3.1 Stage 3 Kernel: 30-Block EO Key-14 4ch SCReLU + 60-Ply 32->H->1 Integer SCReLU MLP (0 SIMD!)
template<bool IS_WHITE>
__attribute__((noinline, noclone))
int32_t eval_waca_v31_stage3_ptm(const WacaG2V31Weights& w,
                                 const PatternMeta* __restrict meta,
                                 const uint16_t* __restrict idx,
                                 uint64_t P, uint64_t O,
                                 uint64_t mvP, uint64_t mvO,
                                 int32_t v1,
                                 const int32_t* __restrict f25,
                                 int32_t r25,
                                 int ply) noexcept {
    int32_t c[4], e[4];
    int32_t inner_sum = 0, subdiag_sum = 0, outer10_sum = 0;
    extract_waca_v31_zones_ptm<IS_WHITE>(w.tier1, meta, idx, ply, v1, c, e,
                                         inner_sum, subdiag_sum, outer10_sum);

    int32_t f_ext[24];
    extract_waca_v31_stage3_extra_features(P, O, mvP, mvO, ply, c, e,
                                           inner_sum, subdiag_sum, outer10_sum, f_ext);

    // 1) Physically separated 30-block Even/Odd Key-14 4ch embeddings (ONLY touched on ~5-7% leaves!)
    const uint8_t eo_blk = PLY_TO_EO30_BLK[ply];
    const WacaKeyEmb4* __restrict htbl = w.h_emb[eo_blk].data();
    int32_t h0[NUM_KEY_PATTERNS], h1[NUM_KEY_PATTERNS];
    int32_t h2[NUM_KEY_PATTERNS], h3[NUM_KEY_PATTERNS];
    int32_t g0 = 0, g1 = 0, g2 = 0, g3 = 0;
    for (int p = 0; p < NUM_KEY_PATTERNS; ++p) {
        const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        const WacaKeyEmb4 emb = htbl[flat];
        const int32_t e0 = emb.h[0], e1 = emb.h[1], e2 = emb.h[2], e3 = emb.h[3];
        h0[p] = e0; h1[p] = e1; h2[p] = e2; h3[p] = e3;
        g0 += e0;   g1 += e1;   g2 += e2;   g3 += e3;
    }

    const SCReLUPlyParams& sc = w.screlu[ply];
    int32_t s0_0 = 0, s0_1 = 0, s0_2 = 0, s0_3 = 0;
    {
        const int32_t m0 = sc.mu[0][0], m1 = sc.mu[0][1], m2 = sc.mu[0][2], m3 = sc.mu[0][3];
        for (int r = 0; r < 4; ++r) {
            const int c1 = (r + 1) & 3;
            const int32_t u0 = clamp_i32(h0[4 + r] + h0[8 + r] + h0[r] + h0[c1] + m0, SCRELU_Q);
            const int32_t u1 = clamp_i32(h1[4 + r] + h1[8 + r] + h1[r] + h1[c1] + m1, SCRELU_Q);
            const int32_t u2 = clamp_i32(h2[4 + r] + h2[8 + r] + h2[r] + h2[c1] + m2, SCRELU_Q);
            const int32_t u3 = clamp_i32(h3[4 + r] + h3[8 + r] + h3[r] + h3[c1] + m3, SCRELU_Q);
            s0_0 += u0 * u0; s0_1 += u1 * u1; s0_2 += u2 * u2; s0_3 += u3 * u3;
        }
    }
    int32_t s1_0 = 0, s1_1 = 0, s1_2 = 0, s1_3 = 0;
    {
        const int32_t m0 = sc.mu[1][0], m1 = sc.mu[1][1], m2 = sc.mu[1][2], m3 = sc.mu[1][3];
        for (int d = 0; d < 2; ++d) {
            const int32_t u0 = clamp_i32(h0[12 + d] + h0[d] + h0[d + 2] + m0, SCRELU_Q);
            const int32_t u1 = clamp_i32(h1[12 + d] + h1[d] + h1[d + 2] + m1, SCRELU_Q);
            const int32_t u2 = clamp_i32(h2[12 + d] + h2[d] + h2[d + 2] + m2, SCRELU_Q);
            const int32_t u3 = clamp_i32(h3[12 + d] + h3[d] + h3[d + 2] + m3, SCRELU_Q);
            s1_0 += u0 * u0; s1_1 += u1 * u1; s1_2 += u2 * u2; s1_3 += u3 * u3;
        }
    }
    const int32_t ug0 = clamp_i32(g0 + sc.mu[2][0], SCRELU_Q);
    const int32_t ug1 = clamp_i32(g1 + sc.mu[2][1], SCRELU_Q);
    const int32_t ug2 = clamp_i32(g2 + sc.mu[2][2], SCRELU_Q);
    const int32_t ug3 = clamp_i32(g3 + sc.mu[2][3], SCRELU_Q);
    const int32_t s2_0 = ug0 * ug0, s2_1 = ug1 * ug1, s2_2 = ug2 * ug2, s2_3 = ug3 * ug3;

    const int32_t sc_acc =
          static_cast<int32_t>(sc.beta[0][0]) * s0_0 + static_cast<int32_t>(sc.beta[0][1]) * s0_1
        + static_cast<int32_t>(sc.beta[0][2]) * s0_2 + static_cast<int32_t>(sc.beta[0][3]) * s0_3
        + static_cast<int32_t>(sc.beta[1][0]) * s1_0 + static_cast<int32_t>(sc.beta[1][1]) * s1_1
        + static_cast<int32_t>(sc.beta[1][2]) * s1_2 + static_cast<int32_t>(sc.beta[1][3]) * s1_3
        + static_cast<int32_t>(sc.beta[2][0]) * s2_0 + static_cast<int32_t>(sc.beta[2][1]) * s2_1
        + static_cast<int32_t>(sc.beta[2][2]) * s2_2 + static_cast<int32_t>(sc.beta[2][3]) * s2_3;
    const int32_t r_key_screlu = sym_rshift(sc_acc, SCRELU_SHIFT);

    // f[28..31]: 4-channel Key-14 nonlinear summary features
    f_ext[20] = clamp_feat_i32((s0_0 + s1_0 + s2_0) >> 8);
    f_ext[21] = clamp_feat_i32((s0_1 + s1_1 + s2_1) >> 8);
    f_ext[22] = clamp_feat_i32((s0_2 + s1_2 + s2_2) >> 8);
    f_ext[23] = clamp_feat_i32((s0_3 + s1_3 + s2_3) >> 8);

    // 2) 60-ply Pure-Integer SCReLU MLP (32 -> H -> 1, 4-way unrolled scalar ILP, overflow-proof in int32_t!)
    const WacaV31MlpPly& mp = w.mlp[ply];
    const int H = w.mlp_hidden;
    int32_t acc2 = mp.b2;
    for (int j = 0; j < H; ++j) {
        const int8_t* __restrict w1j = mp.w1[j];
        int32_t z0 = mp.b1[j], z1 = 0, z2 = 0, z3 = 0;
        for (int i = 0; i < WACA_V31_F25; i += 4) {
            z0 += static_cast<int32_t>(w1j[i + 0]) * f25[i + 0];
            z1 += static_cast<int32_t>(w1j[i + 1]) * f25[i + 1];
            z2 += static_cast<int32_t>(w1j[i + 2]) * f25[i + 2];
            z3 += static_cast<int32_t>(w1j[i + 3]) * f25[i + 3];
        }
        for (int i = 0; i < 24; i += 4) {
            z0 += static_cast<int32_t>(w1j[WACA_V31_F25 + i + 0]) * f_ext[i + 0];
            z1 += static_cast<int32_t>(w1j[WACA_V31_F25 + i + 1]) * f_ext[i + 1];
            z2 += static_cast<int32_t>(w1j[WACA_V31_F25 + i + 2]) * f_ext[i + 2];
            z3 += static_cast<int32_t>(w1j[WACA_V31_F25 + i + 3]) * f_ext[i + 3];
        }
        const int32_t z = (z0 + z1) + (z2 + z3);
        const int32_t u = (z <= 0) ? 0 : (((z >> WACA_MLP_S1) > SCRELU_Q) ? SCRELU_Q : (z >> WACA_MLP_S1));
        acc2 += static_cast<int32_t>(mp.w2[j]) * (u * u);
    }
    const int32_t r_mlp = sym_rshift(acc2, WACA_MLP_S2);

    return std::clamp<int32_t>(r25 + r_mlp + r_key_screlu, -WACA_MAX_RES, WACA_MAX_RES);
}

// Evaluate all 3 stages unconditionally (used for bit-exact testing, Val RMSE, and GateMode::FULL)
inline void eval_waca_g2_v31_stages(const WacaG2V31Weights& w,
                                    const PatternMeta* __restrict meta,
                                    const uint16_t* __restrict idx,
                                    uint64_t B, uint64_t W, int stm, int ply,
                                    int32_t& out_v1, int32_t& out_v25, int32_t& out_v3) noexcept {
    const bool is_w = (stm < 0);
    const uint64_t P = is_w ? W : B;
    const uint64_t O = is_w ? B : W;
    int32_t f25[WACA_V31_F25];
    uint64_t mvP = 0, mvO = 0;
    const int32_t v1  = is_w ? eval_waca_v31_stage1_ptm<true>(w.tier1, meta, idx, ply)
                             : eval_waca_v31_stage1_ptm<false>(w.tier1, meta, idx, ply);
    const int32_t r25 = eval_waca_v31_stage25_kernel(w, P, O, v1, ply, f25, mvP, mvO);
    const int32_t rt  = is_w ? eval_waca_v31_stage3_ptm<true>(w, meta, idx, P, O, mvP, mvO, v1, f25, r25, ply)
                             : eval_waca_v31_stage3_ptm<false>(w, meta, idx, P, O, mvP, mvO, v1, f25, r25, ply);
    out_v1  = v1;
    out_v25 = v1 + r25;
    out_v3  = v1 + rt;
}

inline int32_t eval_waca_g2_v31_full(const WacaG2V31Weights& w,
                                     const PatternMeta* __restrict meta,
                                     const uint16_t* __restrict idx,
                                     uint64_t B, uint64_t W, int stm, int ply) noexcept {
    int32_t v1 = 0, v25 = 0, v3 = 0;
    eval_waca_g2_v31_stages(w, meta, idx, B, W, stm, ply, v1, v25, v3);
    return v3;
}

// Search-integrated 2-stage gated evaluator for WACA_G2_V31 (SAFE_BND / FAIL_HARD / FULL / WIDE_SAFE_BND)
inline int32_t eval_waca_g2_v31_gated(const WacaG2V31Weights& w,
                                      const PatternMeta* __restrict meta,
                                      const uint16_t* __restrict idx,
                                      uint64_t B, uint64_t W, int stm, int ply,
                                      int32_t alpha, int32_t beta,
                                      bool in_mpc_probe,
                                      GateMode gate_mode,
                                      int32_t m25_units,
                                      int32_t m3_units,
                                      uint64_t& stat_s25,
                                      uint64_t& stat_s3,
                                      int32_t tt_guard_units = 576) noexcept {
    const bool is_w = (stm < 0);
    const int32_t v1 = is_w ? eval_waca_v31_stage1_ptm<true>(w.tier1, meta, idx, ply)
                            : eval_waca_v31_stage1_ptm<false>(w.tier1, meta, idx, ply);

    if (in_mpc_probe) return v1;

    const bool is_pv = (beta > alpha + 1);
    if (!is_pv && gate_mode != GateMode::FULL) {
        if (v1 + m25_units <= alpha) return gate_return_low(gate_mode, v1, m25_units, tt_guard_units, alpha);
        if (v1 - m25_units >= beta)  return gate_return_high(gate_mode, v1, m25_units, tt_guard_units, beta);
    }

    // Ultra-light Stage 2.5 (~55-65 ns, 0 Tier-1 reloads, 3 SWAR popcounts!)
    stat_s25++;
    const uint64_t P = is_w ? W : B;
    const uint64_t O = is_w ? B : W;
    int32_t f25[WACA_V31_F25];
    uint64_t mvP = 0, mvO = 0;
    const int32_t r25 = eval_waca_v31_stage25_kernel(w, P, O, v1, ply, f25, mvP, mvO);
    const int32_t v25 = v1 + r25;

    if (!is_pv && gate_mode != GateMode::FULL) {
        if (v25 + m3_units <= alpha) return gate_return_low(gate_mode, v25, m3_units, tt_guard_units, alpha);
        if (v25 - m3_units >= beta)  return gate_return_high(gate_mode, v25, m3_units, tt_guard_units, beta);
    }

    // Long-term Stage 3 (~5-7% leaves: 4-neighbor parity + 5-zone 64-pat + 30-blk EO Key14 + 32->H->1 MLP)
    stat_s3++;
    const int32_t rt = is_w ? eval_waca_v31_stage3_ptm<true>(w, meta, idx, P, O, mvP, mvO, v1, f25, r25, ply)
                            : eval_waca_v31_stage3_ptm<false>(w, meta, idx, P, O, mvP, mvO, v1, f25, r25, ply);
    return v1 + rt;
}

// ============================================================================
// P2C-64 v3.2: WACA_G2_V32 (Slim 8-Dim Stage 2.5 + Fast Fused 16-Load 28-Dim
//   4-Way j-Tile Stage 3 with Smooth ply 19..25 Flood-Fill Ramp + 30-Blk EO Key-14)
// ============================================================================
static constexpr int WACA_V32_EO_BLOCKS = 30; // 15 x 2 (even/odd ply parity, max +/-1 ply drift!)
static constexpr int WACA_V32_F25       = 8;  // Ultra-light Stage 2.5 input dims (3 SWAR popcounts, 8 imul!)
static constexpr int WACA_V32_F3_EXTRA  = 16; // Stage 3 extra bitboard + smooth-ramp parity + fused geom f[8..23]
static constexpr int WACA_V32_F3        = 28; // Stage 3 total MLP input dims (8 + 16 + 4 Key-14 SCReLU)
static constexpr int WACA_V32_MAX_HID   = 24; // 16 (fast default) or 24

struct alignas(64) WacaV32MlpPly {
    int8_t  w1[WACA_V32_MAX_HID][WACA_V32_F3]{}; // 672B
    int32_t b1[WACA_V32_MAX_HID]{};              //  96B
    int16_t w2[WACA_V32_MAX_HID]{};              //  48B
    int32_t b2 = 0;                              //   4B
};

struct WacaG2V32Weights {
    int32_t                                          mlp_hidden = 16; // 16 or 24
    RawNTuple64Weights                               tier1;           // 46.92 MiB untouched RAW_64!
    std::array<WacaV31Stage25Ply, NUM_PLIES>         s25{};           //  1.92 KiB (L1D)
    std::array<std::vector<WacaKeyEmb4>, WACA_V32_EO_BLOCKS> h_emb;   // 10.51 MiB (Stage 3 only!)
    std::array<SCReLUPlyParams, NUM_PLIES>           screlu{};        //  2.81 KiB (L1D)
    std::array<WacaV32MlpPly, NUM_PLIES>             mlp{};           // <= 50 KiB (L1D/L2)

    WacaG2V32Weights() {
        const auto& pt = get_pattern_tables();
        for (int b = 0; b < WACA_V32_EO_BLOCKS; ++b) {
            h_emb[b].assign(static_cast<size_t>(pt.key_raw_entries), WacaKeyEmb4{});
        }
    }

    // Strict D4 internal mirror symmetrization & overflow-safe parameter clamping
    void symmetrize() {
        tier1.symmetrize();
        const auto& pt = get_pattern_tables();
        auto clamp_i8 = [](int8_t v) -> int8_t { return (v == INT8_MIN) ? -127 : v; };

        for (int b = 0; b < WACA_V32_EO_BLOCKS; ++b) {
            WacaKeyEmb4* tbl = h_emb[b].data();
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
                    WacaKeyEmb4 rep = tbl[off + x];
                    for (int k = 0; k < 4; ++k) rep.h[k] = clamp_i8(rep.h[k]);
                    for (int32_t idx : orb) tbl[off + idx] = rep;
                }
            }
        }

        mlp_hidden = std::clamp<int32_t>(mlp_hidden, 1, WACA_V32_MAX_HID);
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            s25[ply].b = std::clamp<int32_t>(s25[ply].b, -WACA_S25_B_MAX, WACA_S25_B_MAX);
            for (int i = 0; i < WACA_V32_F25; ++i) {
                s25[ply].w[i] = static_cast<int16_t>(
                    std::clamp<int32_t>(s25[ply].w[i], -WACA_S25_W_MAX, WACA_S25_W_MAX));
            }
            for (int g = 0; g < 3; ++g) {
                for (int k = 0; k < 4; ++k) {
                    screlu[ply].beta[g][k] = static_cast<int16_t>(
                        std::clamp<int32_t>(screlu[ply].beta[g][k], -SCRELU_BETA_MAX, SCRELU_BETA_MAX));
                }
            }
            mlp[ply].b2 = std::clamp<int32_t>(mlp[ply].b2, -WACA_MLP_B2_MAX, WACA_MLP_B2_MAX);
            for (int j = 0; j < WACA_V32_MAX_HID; ++j) {
                mlp[ply].b1[j] = std::clamp<int32_t>(mlp[ply].b1[j], -WACA_MLP_B1_MAX, WACA_MLP_B1_MAX);
                mlp[ply].w2[j] = static_cast<int16_t>(
                    std::clamp<int32_t>(mlp[ply].w2[j], -WACA_MLP_W2_MAX, WACA_MLP_W2_MAX));
                for (int i = 0; i < WACA_V32_F3; ++i) {
                    mlp[ply].w1[j][i] = clamp_i8(mlp[ply].w1[j][i]);
                }
            }
        }
    }
};

// v3.2 Stage 1: Identical 20-insn 64-pattern PTM loop (0.0% overhead on gated leaves!)
template<bool IS_WHITE>
__attribute__((noinline, noclone))
int32_t eval_waca_v32_stage1_ptm(const RawNTuple64Weights& tier1,
                                 const PatternMeta* __restrict meta,
                                 const uint16_t* __restrict idx,
                                 int ply) noexcept {
    const int16_t* __restrict wp = tier1.w_ply[ply].data();
    int32_t sum = 0;
    for (int p = 0; p < NUM_PATTERNS; ++p) {
        const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        sum += wp[flat];
    }
    return sum;
}

// v3.2 Stage 2.5 Kernel: Same ultra-light 8-dim linear gate (~70 ns, 0 SIMD, 0 PLT calls!)
__attribute__((noinline, noclone))
inline int32_t eval_waca_v32_stage25_kernel(const WacaG2V32Weights& w,
                                            uint64_t P, uint64_t O,
                                            int32_t v1, int ply,
                                            int32_t f_out[WACA_V32_F25],
                                            uint64_t& out_mvP,
                                            uint64_t& out_mvO) noexcept {
    extract_waca_v31_stage25_features(P, O, v1, ply, f_out, out_mvP, out_mvO);
    const WacaV31Stage25Ply& sp = w.s25[ply];
    const int32_t a0 = sp.b
                     + static_cast<int32_t>(sp.w[0]) * f_out[0]
                     + static_cast<int32_t>(sp.w[4]) * f_out[4];
    const int32_t a1 = static_cast<int32_t>(sp.w[1]) * f_out[1]
                     + static_cast<int32_t>(sp.w[5]) * f_out[5];
    const int32_t a2 = static_cast<int32_t>(sp.w[2]) * f_out[2]
                     + static_cast<int32_t>(sp.w[6]) * f_out[6];
    const int32_t a3 = static_cast<int32_t>(sp.w[3]) * f_out[3]
                     + static_cast<int32_t>(sp.w[7]) * f_out[7];
    const int32_t acc = (a0 + a1) + (a2 + a3);
    return std::clamp<int32_t>(sym_rshift(acc, WACA_S25_SHIFT), -WACA_MAX_RES, WACA_MAX_RES);
}

// v3.2 Stage 3 extra 16 features f[8..23] (shared verbatim by C++ inference & OpenMP training bridge!)
//   f_ext[0..3]  = f[8..11]  : edge mobility, potential mobility, frontier, corner-edge+triangle stability
//   f_ext[4..7]  = f[12..15] : smooth ply 19..25 ramp 4-neighbor empty-region parity (0 cost when ply <= 18!)
//   f_ext[8..15] = f[16..23] : fused 16-load geometric stats (c_sum, e_sum, c_spread, c_sq, diag_prod, polar, ce_coup, inner16 - (c+e))
__attribute__((always_inline))
static inline void extract_waca_v32_stage3_extra_features(uint64_t P, uint64_t O,
                                                          uint64_t mvP, uint64_t mvO,
                                                          int ply,
                                                          const int32_t c[4], const int32_t e[4],
                                                          int32_t c_sum, int32_t e_sum,
                                                          int32_t inner16_sum,
                                                          int32_t f_ext[WACA_V32_F3_EXTRA]) noexcept {
    // f[8..11]: Bitboard features reusing mvP/mvO from Stage 2.5
    constexpr uint64_t EDGES = 0xff818181818181ffULL;
    const int32_t emobP = popcount64_swar(mvP & EDGES);
    const int32_t emobO = popcount64_swar(mvO & EDGES);
    const uint64_t E    = ~(P | O);
    const int32_t potP  = popcount64_swar(adj_mask_bb(O) & E);
    const int32_t potO  = popcount64_swar(adj_mask_bb(P) & E);
    const uint64_t adjE = adj_mask_bb(E);
    const int32_t frP   = popcount64_swar(P & adjE);
    const int32_t frO   = popcount64_swar(O & adjE);
    const int32_t stabP = count_edge_stable_bb(P);
    const int32_t stabO = count_edge_stable_bb(O);

    f_ext[0] = clamp_feat_i32((emobP - emobO) << 4);
    f_ext[1] = clamp_feat_i32((potP - potO) << 4);
    f_ext[2] = clamp_feat_i32((frP - frO) << 4);
    f_ext[3] = clamp_feat_i32((stabP - stabO) << 4);

    // f[12..15]: Smooth linear ramp over ply 19..25 (ramp = clamp(ply - 18, 0, 8), 0 cost when ply <= 18!)
    const int32_t ramp = std::clamp<int32_t>(ply - 18, 0, 8);
    if (ramp == 0) {
        f_ext[4] = 0;
        f_ext[5] = 0;
        f_ext[6] = 0;
        f_ext[7] = 0;
    } else {
        int32_t odd_cnt = 0, single_cnt = 0, odd_mv_diff = 0, reg_cnt = 0;
        const uint64_t adjE4 = ((E << 1) & 0xfefefefefefefefeULL)
                             | ((E >> 1) & 0x7f7f7f7f7f7f7f7fULL)
                             | (E << 8)
                             | (E >> 8);
        const uint64_t singles = E & ~adjE4;
        if (singles) {
            single_cnt  = popcount64_swar(singles);
            odd_cnt     = single_cnt;
            reg_cnt     = single_cnt;
            odd_mv_diff = popcount64_swar(singles & mvP) - popcount64_swar(singles & mvO);
        }
        uint64_t rem = E ^ singles;
        while (rem) {
            uint64_t comp = rem & (~rem + 1ULL);
            uint64_t prev;
            do {
                prev = comp;
                comp |= (((comp << 1) & 0xfefefefefefefefeULL)
                       | ((comp >> 1) & 0x7f7f7f7f7f7f7f7fULL)
                       | (comp << 8)
                       | (comp >> 8)) & rem;
                comp |= (((comp << 1) & 0xfefefefefefefefeULL)
                       | ((comp >> 1) & 0x7f7f7f7f7f7f7f7fULL)
                       | (comp << 8)
                       | (comp >> 8)) & rem;
            } while (comp != prev);
            rem ^= comp;
            reg_cnt++;
            const int32_t sz = (reg_cnt == 1 && rem == 0ULL) ? (60 - ply) : popcount64_swar(comp);
            if (sz & 1) {
                odd_cnt++;
                odd_mv_diff += ((comp & mvP) != 0ULL ? 1 : 0) - ((comp & mvO) != 0ULL ? 1 : 0);
            }
        }
        const int32_t raw0 = (odd_cnt - 1) << 5;
        const int32_t raw1 = single_cnt << 6;
        const int32_t raw2 = odd_mv_diff << 6;
        const int32_t raw3 = (reg_cnt - 2) << 5;
        if (ramp < 8) {
            f_ext[4] = clamp_feat_i32(sym_rshift(raw0 * ramp, 3));
            f_ext[5] = clamp_feat_i32(sym_rshift(raw1 * ramp, 3));
            f_ext[6] = clamp_feat_i32(sym_rshift(raw2 * ramp, 3));
            f_ext[7] = clamp_feat_i32(sym_rshift(raw3 * ramp, 3));
        } else {
            f_ext[4] = clamp_feat_i32(raw0);
            f_ext[5] = clamp_feat_i32(raw1);
            f_ext[6] = clamp_feat_i32(raw2);
            f_ext[7] = clamp_feat_i32(raw3);
        }
    }

    // f[16..23]: Fused 16-load geometric stats (8 dims, no e_min/e_max!)
    int32_t c_min = c[0], c_max = c[0];
    for (int i = 1; i < 4; ++i) {
        if (c[i] < c_min) c_min = c[i];
        if (c[i] > c_max) c_max = c[i];
    }

    int32_t cs[4], es[4];
    int32_t c_sq = 0;
    for (int i = 0; i < 4; ++i) {
        cs[i] = std::clamp<int32_t>(sym_rshift(c[i], 4), -256, 256);
        es[i] = std::clamp<int32_t>(sym_rshift(e[i], 4), -256, 256);
        c_sq += cs[i] * cs[i];
    }
    const int32_t diag_prod = cs[0] * cs[2] + cs[1] * cs[3];
    const int32_t adj_diff  = abs_i32(cs[0] - cs[1]) + abs_i32(cs[1] - cs[2])
                            + abs_i32(cs[2] - cs[3]) + abs_i32(cs[3] - cs[0]);
    const int32_t diag_diff = abs_i32(cs[0] - cs[2]) + abs_i32(cs[1] - cs[3]);
    const int32_t polar     = adj_diff - 2 * diag_diff;
    const int32_t ce_coup   = cs[0] * (es[0] + es[3]) + cs[1] * (es[1] + es[0])
                            + cs[2] * (es[2] + es[1]) + cs[3] * (es[3] + es[2]);

    f_ext[8]  = clamp_feat_i32(sym_rshift(c_sum, 3));
    f_ext[9]  = clamp_feat_i32(sym_rshift(e_sum, 3));
    f_ext[10] = clamp_feat_i32(sym_rshift(c_max - c_min, 3));
    f_ext[11] = clamp_feat_i32(sym_rshift(c_sq, 7));
    f_ext[12] = clamp_feat_i32(sym_rshift(diag_prod, 6));
    f_ext[13] = clamp_feat_i32(sym_rshift(polar, 1));
    f_ext[14] = clamp_feat_i32(sym_rshift(ce_coup, 7));
    f_ext[15] = clamp_feat_i32(sym_rshift(inner16_sum - (c_sum + e_sum), 4));
}

// v3.2 Stage 3 Kernel: Fused 16-load Key-14 + Center2x4, 30-Blk EO SCReLU + 4-way j-Tile 28->H->1 MLP (0 SIMD!)
// Expects contiguous f[WACA_V32_F3] with f[0..7] already populated by Stage 2.5 (0 copy!).
template<bool IS_WHITE>
__attribute__((noinline, noclone))
int32_t eval_waca_v32_stage3_ptm(const WacaG2V32Weights& w,
                                 const PatternMeta* __restrict meta,
                                 const uint16_t* __restrict idx,
                                 uint64_t P, uint64_t O,
                                 uint64_t mvP, uint64_t mvO,
                                 int32_t* __restrict f,
                                 int32_t r25,
                                 int ply) noexcept {
    // 1) Fused p=0..13 loop: single flat index calculation loads BOTH L1D-hot wp[flat] & htbl[flat]!
    const int16_t* __restrict wp = w.tier1.w_ply[ply].data();
    const uint8_t eo_blk = PLY_TO_EO30_BLK[ply];
    const WacaKeyEmb4* __restrict htbl = w.h_emb[eo_blk].data();

    int32_t c[4], e[4];
    int32_t c_sum = 0, e_sum = 0, l2d8_sum = 0;
    int32_t h0[NUM_KEY_PATTERNS], h1[NUM_KEY_PATTERNS];
    int32_t h2[NUM_KEY_PATTERNS], h3[NUM_KEY_PATTERNS];
    int32_t g0 = 0, g1 = 0, g2 = 0, g3 = 0;

    for (int p = 0; p < 4; ++p) {
        const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        const int32_t wv = wp[flat];
        c[p] = wv;
        c_sum += wv;
        const WacaKeyEmb4 emb = htbl[flat];
        const int32_t e0 = emb.h[0], e1 = emb.h[1], e2 = emb.h[2], e3 = emb.h[3];
        h0[p] = e0; h1[p] = e1; h2[p] = e2; h3[p] = e3;
        g0 += e0;   g1 += e1;   g2 += e2;   g3 += e3;
    }
    for (int p = 4; p < 8; ++p) {
        const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        const int32_t wv = wp[flat];
        e[p - 4] = wv;
        e_sum += wv;
        const WacaKeyEmb4 emb = htbl[flat];
        const int32_t e0 = emb.h[0], e1 = emb.h[1], e2 = emb.h[2], e3 = emb.h[3];
        h0[p] = e0; h1[p] = e1; h2[p] = e2; h3[p] = e3;
        g0 += e0;   g1 += e1;   g2 += e2;   g3 += e3;
    }
    for (int p = 8; p < NUM_KEY_PATTERNS; ++p) {
        const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        l2d8_sum += wp[flat];
        const WacaKeyEmb4 emb = htbl[flat];
        const int32_t e0 = emb.h[0], e1 = emb.h[1], e2 = emb.h[2], e3 = emb.h[3];
        h0[p] = e0; h1[p] = e1; h2[p] = e2; h3[p] = e3;
        g0 += e0;   g1 += e1;   g2 += e2;   g3 += e3;
    }
    // Only 2 additional Tier-1 loads: p=62..63 (Center 2x4 x 2) -> total 16 Tier-1 loads!
    const int32_t f62 = IS_WHITE ? (meta[62].raw_off_end - idx[62]) : (meta[62].raw_off + idx[62]);
    const int32_t f63 = IS_WHITE ? (meta[63].raw_off_end - idx[63]) : (meta[63].raw_off + idx[63]);
    const int32_t inner16_sum = l2d8_sum + wp[f62] + wp[f63];

    // 2) Populate f[8..23] in-place (f[0..7] already written by Stage 2.5 -> 0 copy, 0 xmm!)
    extract_waca_v32_stage3_extra_features(P, O, mvP, mvO, ply, c, e, c_sum, e_sum, inner16_sum, f + WACA_V32_F25);

    // 3) 30-block EO Key-14 4ch SCReLU head + f[24..27]
    const SCReLUPlyParams& sc = w.screlu[ply];
    int32_t s0_0 = 0, s0_1 = 0, s0_2 = 0, s0_3 = 0;
    {
        const int32_t m0 = sc.mu[0][0], m1 = sc.mu[0][1], m2 = sc.mu[0][2], m3 = sc.mu[0][3];
        for (int r = 0; r < 4; ++r) {
            const int c1 = (r + 1) & 3;
            const int32_t u0 = clamp_i32(h0[4 + r] + h0[8 + r] + h0[r] + h0[c1] + m0, SCRELU_Q);
            const int32_t u1 = clamp_i32(h1[4 + r] + h1[8 + r] + h1[r] + h1[c1] + m1, SCRELU_Q);
            const int32_t u2 = clamp_i32(h2[4 + r] + h2[8 + r] + h2[r] + h2[c1] + m2, SCRELU_Q);
            const int32_t u3 = clamp_i32(h3[4 + r] + h3[8 + r] + h3[r] + h3[c1] + m3, SCRELU_Q);
            s0_0 += u0 * u0; s0_1 += u1 * u1; s0_2 += u2 * u2; s0_3 += u3 * u3;
        }
    }
    int32_t s1_0 = 0, s1_1 = 0, s1_2 = 0, s1_3 = 0;
    {
        const int32_t m0 = sc.mu[1][0], m1 = sc.mu[1][1], m2 = sc.mu[1][2], m3 = sc.mu[1][3];
        for (int d = 0; d < 2; ++d) {
            const int32_t u0 = clamp_i32(h0[12 + d] + h0[d] + h0[d + 2] + m0, SCRELU_Q);
            const int32_t u1 = clamp_i32(h1[12 + d] + h1[d] + h1[d + 2] + m1, SCRELU_Q);
            const int32_t u2 = clamp_i32(h2[12 + d] + h2[d] + h2[d + 2] + m2, SCRELU_Q);
            const int32_t u3 = clamp_i32(h3[12 + d] + h3[d] + h3[d + 2] + m3, SCRELU_Q);
            s1_0 += u0 * u0; s1_1 += u1 * u1; s1_2 += u2 * u2; s1_3 += u3 * u3;
        }
    }
    const int32_t ug0 = clamp_i32(g0 + sc.mu[2][0], SCRELU_Q);
    const int32_t ug1 = clamp_i32(g1 + sc.mu[2][1], SCRELU_Q);
    const int32_t ug2 = clamp_i32(g2 + sc.mu[2][2], SCRELU_Q);
    const int32_t ug3 = clamp_i32(g3 + sc.mu[2][3], SCRELU_Q);
    const int32_t s2_0 = ug0 * ug0, s2_1 = ug1 * ug1, s2_2 = ug2 * ug2, s2_3 = ug3 * ug3;

    const int32_t sc_acc =
          static_cast<int32_t>(sc.beta[0][0]) * s0_0 + static_cast<int32_t>(sc.beta[0][1]) * s0_1
        + static_cast<int32_t>(sc.beta[0][2]) * s0_2 + static_cast<int32_t>(sc.beta[0][3]) * s0_3
        + static_cast<int32_t>(sc.beta[1][0]) * s1_0 + static_cast<int32_t>(sc.beta[1][1]) * s1_1
        + static_cast<int32_t>(sc.beta[1][2]) * s1_2 + static_cast<int32_t>(sc.beta[1][3]) * s1_3
        + static_cast<int32_t>(sc.beta[2][0]) * s2_0 + static_cast<int32_t>(sc.beta[2][1]) * s2_1
        + static_cast<int32_t>(sc.beta[2][2]) * s2_2 + static_cast<int32_t>(sc.beta[2][3]) * s2_3;
    const int32_t r_key_screlu = sym_rshift(sc_acc, SCRELU_SHIFT);

    f[24] = clamp_feat_i32((s0_0 + s1_0 + s2_0) >> 8);
    f[25] = clamp_feat_i32((s0_1 + s1_1 + s2_1) >> 8);
    f[26] = clamp_feat_i32((s0_2 + s1_2 + s2_2) >> 8);
    f[27] = clamp_feat_i32((s0_3 + s1_3 + s2_3) >> 8);

    // 4) 60-ply Pure-Integer SCReLU MLP (28 -> H -> 1, 4-way j-tile scalar ILP, 4x fewer f[i] loads!)
    const WacaV32MlpPly& mp = w.mlp[ply];
    const int H = w.mlp_hidden;
    int32_t acc2 = mp.b2;
    int j = 0;
    for (; j + 3 < H; j += 4) {
        const int8_t* __restrict w0 = mp.w1[j + 0];
        const int8_t* __restrict w1 = mp.w1[j + 1];
        const int8_t* __restrict w2 = mp.w1[j + 2];
        const int8_t* __restrict w3 = mp.w1[j + 3];
        int32_t z0 = mp.b1[j + 0];
        int32_t z1 = mp.b1[j + 1];
        int32_t z2 = mp.b1[j + 2];
        int32_t z3 = mp.b1[j + 3];
        for (int i = 0; i < WACA_V32_F3; ++i) {
            const int32_t fv = f[i];
            z0 += static_cast<int32_t>(w0[i]) * fv;
            z1 += static_cast<int32_t>(w1[i]) * fv;
            z2 += static_cast<int32_t>(w2[i]) * fv;
            z3 += static_cast<int32_t>(w3[i]) * fv;
        }
        const int32_t u0 = (z0 <= 0) ? 0 : (((z0 >> WACA_MLP_S1) > SCRELU_Q) ? SCRELU_Q : (z0 >> WACA_MLP_S1));
        const int32_t u1 = (z1 <= 0) ? 0 : (((z1 >> WACA_MLP_S1) > SCRELU_Q) ? SCRELU_Q : (z1 >> WACA_MLP_S1));
        const int32_t u2 = (z2 <= 0) ? 0 : (((z2 >> WACA_MLP_S1) > SCRELU_Q) ? SCRELU_Q : (z2 >> WACA_MLP_S1));
        const int32_t u3 = (z3 <= 0) ? 0 : (((z3 >> WACA_MLP_S1) > SCRELU_Q) ? SCRELU_Q : (z3 >> WACA_MLP_S1));
        acc2 += static_cast<int32_t>(mp.w2[j + 0]) * (u0 * u0)
              + static_cast<int32_t>(mp.w2[j + 1]) * (u1 * u1)
              + static_cast<int32_t>(mp.w2[j + 2]) * (u2 * u2)
              + static_cast<int32_t>(mp.w2[j + 3]) * (u3 * u3);
    }
    for (; j < H; ++j) {
        const int8_t* __restrict w0 = mp.w1[j];
        int32_t z = mp.b1[j];
        for (int i = 0; i < WACA_V32_F3; ++i) {
            z += static_cast<int32_t>(w0[i]) * f[i];
        }
        const int32_t u = (z <= 0) ? 0 : (((z >> WACA_MLP_S1) > SCRELU_Q) ? SCRELU_Q : (z >> WACA_MLP_S1));
        acc2 += static_cast<int32_t>(mp.w2[j]) * (u * u);
    }
    const int32_t r_mlp = sym_rshift(acc2, WACA_MLP_S2);

    return std::clamp<int32_t>(r25 + r_mlp + r_key_screlu, -WACA_MAX_RES, WACA_MAX_RES);
}

// Evaluate all 3 stages unconditionally (used for bit-exact testing, Val RMSE, and GateMode::FULL)
inline void eval_waca_g2_v32_stages(const WacaG2V32Weights& w,
                                    const PatternMeta* __restrict meta,
                                    const uint16_t* __restrict idx,
                                    uint64_t B, uint64_t W, int stm, int ply,
                                    int32_t& out_v1, int32_t& out_v25, int32_t& out_v3) noexcept {
    const bool is_w = (stm < 0);
    const uint64_t P = is_w ? W : B;
    const uint64_t O = is_w ? B : W;
    alignas(64) int32_t f[WACA_V32_F3];
    uint64_t mvP = 0, mvO = 0;
    const int32_t v1  = is_w ? eval_waca_v32_stage1_ptm<true>(w.tier1, meta, idx, ply)
                             : eval_waca_v32_stage1_ptm<false>(w.tier1, meta, idx, ply);
    const int32_t r25 = eval_waca_v32_stage25_kernel(w, P, O, v1, ply, f, mvP, mvO);
    const int32_t rt  = is_w ? eval_waca_v32_stage3_ptm<true>(w, meta, idx, P, O, mvP, mvO, f, r25, ply)
                             : eval_waca_v32_stage3_ptm<false>(w, meta, idx, P, O, mvP, mvO, f, r25, ply);
    out_v1  = v1;
    out_v25 = v1 + r25;
    out_v3  = v1 + rt;
}

inline int32_t eval_waca_g2_v32_full(const WacaG2V32Weights& w,
                                     const PatternMeta* __restrict meta,
                                     const uint16_t* __restrict idx,
                                     uint64_t B, uint64_t W, int stm, int ply) noexcept {
    int32_t v1 = 0, v25 = 0, v3 = 0;
    eval_waca_g2_v32_stages(w, meta, idx, B, W, stm, ply, v1, v25, v3);
    return v3;
}

// Search-integrated 2-stage gated evaluator for WACA_G2_V32 (SAFE_BND / FAIL_HARD / FULL / WIDE_SAFE_BND)
inline int32_t eval_waca_g2_v32_gated(const WacaG2V32Weights& w,
                                      const PatternMeta* __restrict meta,
                                      const uint16_t* __restrict idx,
                                      uint64_t B, uint64_t W, int stm, int ply,
                                      int32_t alpha, int32_t beta,
                                      bool in_mpc_probe,
                                      GateMode gate_mode,
                                      int32_t m25_units,
                                      int32_t m3_units,
                                      uint64_t& stat_s25,
                                      uint64_t& stat_s3,
                                      int32_t tt_guard_units = 576) noexcept {
    const bool is_w = (stm < 0);
    const int32_t v1 = is_w ? eval_waca_v32_stage1_ptm<true>(w.tier1, meta, idx, ply)
                            : eval_waca_v32_stage1_ptm<false>(w.tier1, meta, idx, ply);

    if (in_mpc_probe) return v1;

    const bool is_pv = (beta > alpha + 1);
    if (!is_pv && gate_mode != GateMode::FULL) {
        if (v1 + m25_units <= alpha) return gate_return_low(gate_mode, v1, m25_units, tt_guard_units, alpha);
        if (v1 - m25_units >= beta)  return gate_return_high(gate_mode, v1, m25_units, tt_guard_units, beta);
    }

    // Ultra-light Stage 2.5 (~70 ns, 0 Tier-1 reloads, 3 SWAR popcounts!)
    stat_s25++;
    const uint64_t P = is_w ? W : B;
    const uint64_t O = is_w ? B : W;
    alignas(64) int32_t f[WACA_V32_F3];
    uint64_t mvP = 0, mvO = 0;
    const int32_t r25 = eval_waca_v32_stage25_kernel(w, P, O, v1, ply, f, mvP, mvO);
    const int32_t v25 = v1 + r25;

    if (!is_pv && gate_mode != GateMode::FULL) {
        if (v25 + m3_units <= alpha) return gate_return_low(gate_mode, v25, m3_units, tt_guard_units, alpha);
        if (v25 - m3_units >= beta)  return gate_return_high(gate_mode, v25, m3_units, tt_guard_units, beta);
    }

    // Fast Stage 3 (~5-7% leaves: fused 16-load + smooth ply 19..25 ramp + 30-blk EO Key14 + 4-way j-tile 28->H->1 MLP)
    stat_s3++;
    const int32_t rt = is_w ? eval_waca_v32_stage3_ptm<true>(w, meta, idx, P, O, mvP, mvO, f, r25, ply)
                            : eval_waca_v32_stage3_ptm<false>(w, meta, idx, P, O, mvP, mvO, f, r25, ply);
    return v1 + rt;
}

// ============================================================================
// P2C-64 v3.3: WACA_G2_V33
//   - Stage 2.5 : 12-dim polynomial gate (8 base + 4 cross terms, 0 extra loads/popcounts)
//   - Stage 3   : 30-block EO Key-14 8ch (21.02 MiB) + Center-10 4ch (2.25 MiB)
//                 + 44-term SCReLU/Pairwise/Center-Outer head + ply 24..31 ramp
//                 + 60-ply 32 -> H(=16) -> 1 4-way j-tile pure-scalar MLP (0 SIMD!)
// ============================================================================
static constexpr int     WACA_V33_EO_BLOCKS       = 30;    // 15 x 2 (even/odd ply parity, max +/-1 ply drift!)
static constexpr int     WACA_V33_F25             = 12;    // 8 base + 4 polynomial cross terms
static constexpr int     WACA_V33_F3_EXTRA        = 16;    // f[8..23]: 4 bitboard + 4 ramp parity + 8 full-64 geom
static constexpr int     WACA_V33_F3              = 32;    // 8 base + 16 extra + 8 sparse summary
static constexpr int     WACA_V33_MAX_HID         = 24;    // 16 (optimal default) or up to 24
static constexpr int     WACA_V33_CTR_PATTERNS    = 10;    // Line3(4) + Line4(4) + Center2x4(2)
static constexpr int32_t WACA_V33_CTR_RAW_ENTRIES = 19683; // 3 * 3^8 (type4: 0..6560, type5: 6561..13121, type15: 13122..19682)
static constexpr int     WACA_V33_SCRELU_TERMS    = 44;    // 12(g0) + 12(g1) + 12(g2) + 4(g3 ctr) + 4(g4 cross)
static constexpr int32_t WACA_V33_SCRELU_BETA_MAX = 1023;  // 92 * 127^2 * 1023 = 1,517,996,964 < INT32_MAX!

struct alignas(8) WacaKeyEmb8 {
    int8_t h[8]{};
};
static_assert(sizeof(WacaKeyEmb8) == 8, "WacaKeyEmb8 must be 8 bytes (single 64-bit scalar load)");

struct alignas(32) WacaV33Stage25Ply {
    int16_t w[WACA_V33_F25]{}; // 24B
    int32_t b = 0;             //  4B
};

struct alignas(16) WacaV33SCReLUPly {
    int16_t mu_key[3][8]{};               // 48B: Group 0/1/2 x 8ch
    int16_t mu_ctr[4]{};                  //  8B: Group 3 (Center-10) x 4ch
    int16_t beta[WACA_V33_SCRELU_TERMS]{}; // 88B: 44-term linear combination (|beta| <= 1023)
};
static_assert(sizeof(WacaV33SCReLUPly) == 144, "WacaV33SCReLUPly must be 144 bytes");

struct alignas(64) WacaV33MlpPly {
    int8_t  w1[WACA_V33_MAX_HID][WACA_V33_F3]{}; // 768B (active H=16 uses 512B!)
    int32_t b1[WACA_V33_MAX_HID]{};              //  96B
    int16_t w2[WACA_V33_MAX_HID]{};              //  48B
    int32_t b2 = 0;                              //   4B
};

struct WacaG2V33Weights {
    int32_t                                          mlp_hidden = 16; // 16 (optimal default)
    RawNTuple64Weights                               tier1;           // 46.92 MiB untouched RAW_64!
    std::array<WacaV33Stage25Ply, NUM_PLIES>         s25{};           //  1.92 KiB (L1D)
    std::array<std::vector<WacaKeyEmb8>, WACA_V33_EO_BLOCKS> h_emb;   // 21.02 MiB (Key-14 8ch, Stage 3 only!)
    std::array<std::vector<WacaKeyEmb4>, WACA_V33_EO_BLOCKS> ctr_emb; //  2.25 MiB (Center-10 4ch, Stage 3 only!)
    std::array<WacaV33SCReLUPly, NUM_PLIES>          screlu{};        //  8.64 KiB (L1D)
    std::array<WacaV33MlpPly, NUM_PLIES>             mlp{};           // 36.7 KiB active (H=16)

    WacaG2V33Weights() {
        const auto& pt = get_pattern_tables();
        for (int b = 0; b < WACA_V33_EO_BLOCKS; ++b) {
            h_emb[b].assign(static_cast<size_t>(pt.key_raw_entries), WacaKeyEmb8{});
            ctr_emb[b].assign(static_cast<size_t>(WACA_V33_CTR_RAW_ENTRIES), WacaKeyEmb4{});
        }
    }

    // Strict D4 internal mirror symmetrization & overflow-safe parameter clamping
    void symmetrize() {
        tier1.symmetrize();
        const auto& pt = get_pattern_tables();
        auto clamp_i8 = [](int8_t v) -> int8_t { return (v == INT8_MIN) ? -127 : v; };

        // 1) Key-14 8ch embeddings (types 0..3)
        for (int b = 0; b < WACA_V33_EO_BLOCKS; ++b) {
            WacaKeyEmb8* tbl = h_emb[b].data();
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
                    WacaKeyEmb8 rep = tbl[off + x];
                    for (int k = 0; k < 8; ++k) rep.h[k] = clamp_i8(rep.h[k]);
                    for (int32_t idx : orb) tbl[off + idx] = rep;
                }
            }
        }

        // 2) Center-10 4ch embeddings (types 4, 5, 15 -> offsets 0, 6561, 13122)
        static constexpr int CTR_TYPES[3] = {4, 5, 15};
        for (int b = 0; b < WACA_V33_EO_BLOCKS; ++b) {
            WacaKeyEmb4* tbl = ctr_emb[b].data();
            for (int ci = 0; ci < 3; ++ci) {
                const int t       = CTR_TYPES[ci];
                const int32_t off = ci * 6561;
                const int32_t sz  = pt.type_raw_size[t]; // 6561
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
                    WacaKeyEmb4 rep = tbl[off + x];
                    for (int k = 0; k < 4; ++k) rep.h[k] = clamp_i8(rep.h[k]);
                    for (int32_t idx : orb) tbl[off + idx] = rep;
                }
            }
        }

        mlp_hidden = std::clamp<int32_t>(mlp_hidden, 1, WACA_V33_MAX_HID);
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            s25[ply].b = std::clamp<int32_t>(s25[ply].b, -WACA_S25_B_MAX, WACA_S25_B_MAX);
            for (int i = 0; i < WACA_V33_F25; ++i) {
                s25[ply].w[i] = static_cast<int16_t>(
                    std::clamp<int32_t>(s25[ply].w[i], -WACA_S25_W_MAX, WACA_S25_W_MAX));
            }
            for (int m = 0; m < WACA_V33_SCRELU_TERMS; ++m) {
                screlu[ply].beta[m] = static_cast<int16_t>(
                    std::clamp<int32_t>(screlu[ply].beta[m], -WACA_V33_SCRELU_BETA_MAX, WACA_V33_SCRELU_BETA_MAX));
            }
            mlp[ply].b2 = std::clamp<int32_t>(mlp[ply].b2, -WACA_MLP_B2_MAX, WACA_MLP_B2_MAX);
            for (int j = 0; j < WACA_V33_MAX_HID; ++j) {
                mlp[ply].b1[j] = std::clamp<int32_t>(mlp[ply].b1[j], -WACA_MLP_B1_MAX, WACA_MLP_B1_MAX);
                mlp[ply].w2[j] = static_cast<int16_t>(
                    std::clamp<int32_t>(mlp[ply].w2[j], -WACA_MLP_W2_MAX, WACA_MLP_W2_MAX));
                for (int i = 0; i < WACA_V33_F3; ++i) {
                    mlp[ply].w1[j][i] = clamp_i8(mlp[ply].w1[j][i]);
                }
            }
        }
    }
};

// v3.3 Stage 1: Identical 20-insn 64-pattern PTM loop (0.0% overhead on gated leaves!)
template<bool IS_WHITE>
__attribute__((noinline, noclone))
int32_t eval_waca_v33_stage1_ptm(const RawNTuple64Weights& tier1,
                                 const PatternMeta* __restrict meta,
                                 const uint16_t* __restrict idx,
                                 int ply) noexcept {
    const int16_t* __restrict wp = tier1.w_ply[ply].data();
    int32_t sum = 0;
    for (int p = 0; p < NUM_PATTERNS; ++p) {
        const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        sum += wp[flat];
    }
    return sum;
}

// v3.3 Stage 2.5 feature extraction: 12-dim polynomial (8 base + 4 cross terms, 0 extra loads/popcounts!)
__attribute__((always_inline))
static inline void extract_waca_v33_stage25_features(uint64_t P, uint64_t O,
                                                     int32_t v1, int ply,
                                                     int32_t f_out[WACA_V33_F25],
                                                     uint64_t& out_mvP,
                                                     uint64_t& out_mvO) noexcept {
    extract_waca_v31_stage25_features(P, O, v1, ply, f_out, out_mvP, out_mvO);
    const int32_t f0 = f_out[0];
    const int32_t f6 = f_out[6];
    f_out[8]  = clamp_feat_i32(sym_rshift(f0 * f_out[1], 8));
    f_out[9]  = clamp_feat_i32(sym_rshift(f0 * f6, 8));
    f_out[10] = clamp_feat_i32(sym_rshift(f0 * f_out[3], 8));
    f_out[11] = clamp_feat_i32(sym_rshift(f6 * abs_i32(f6), 8));
}

// v3.3 Stage 2.5 Kernel: 12-Dim Polynomial Linear Gate (~62 ns, 0 SIMD, 0 PLT calls!)
__attribute__((noinline, noclone))
inline int32_t eval_waca_v33_stage25_kernel(const WacaG2V33Weights& w,
                                            uint64_t P, uint64_t O,
                                            int32_t v1, int ply,
                                            int32_t f_out[WACA_V33_F25],
                                            uint64_t& out_mvP,
                                            uint64_t& out_mvO) noexcept {
    extract_waca_v33_stage25_features(P, O, v1, ply, f_out, out_mvP, out_mvO);
    const WacaV33Stage25Ply& sp = w.s25[ply];
    const int32_t a0 = sp.b
                     + static_cast<int32_t>(sp.w[0]) * f_out[0]
                     + static_cast<int32_t>(sp.w[4]) * f_out[4]
                     + static_cast<int32_t>(sp.w[8]) * f_out[8];
    const int32_t a1 = static_cast<int32_t>(sp.w[1]) * f_out[1]
                     + static_cast<int32_t>(sp.w[5]) * f_out[5]
                     + static_cast<int32_t>(sp.w[9]) * f_out[9];
    const int32_t a2 = static_cast<int32_t>(sp.w[2]) * f_out[2]
                     + static_cast<int32_t>(sp.w[6]) * f_out[6]
                     + static_cast<int32_t>(sp.w[10]) * f_out[10];
    const int32_t a3 = static_cast<int32_t>(sp.w[3]) * f_out[3]
                     + static_cast<int32_t>(sp.w[7]) * f_out[7]
                     + static_cast<int32_t>(sp.w[11]) * f_out[11];
    const int32_t acc = (a0 + a1) + (a2 + a3);
    return std::clamp<int32_t>(sym_rshift(acc, WACA_S25_SHIFT), -WACA_MAX_RES, WACA_MAX_RES);
}

// v3.3 Stage 3 extra 16 features f[8..23] (shared verbatim by C++ inference & OpenMP training bridge!)
//   f_ext[0..3]  = f[8..11]  : edge mobility, potential mobility, frontier, corner-edge+triangle stability
//   f_ext[4..7]  = f[12..15] : smooth ply 24..31 ramp 4-neighbor empty-region parity (0 cost when ply <= 23!)
//   f_ext[8..15] = f[16..23] : fused 24-load + 0-load outer_rest_sum full-64 geometry
__attribute__((always_inline))
static inline void extract_waca_v33_stage3_extra_features(uint64_t P, uint64_t O,
                                                          uint64_t mvP, uint64_t mvO,
                                                          int ply,
                                                          const int32_t c[4], const int32_t e[4],
                                                          int32_t c_sum, int32_t e_sum,
                                                          int32_t ctr10_tier1_sum,
                                                          int32_t outer_rest_sum,
                                                          int32_t f_ext[WACA_V33_F3_EXTRA]) noexcept {
    // f[8..11]: Bitboard features reusing mvP/mvO from Stage 2.5
    constexpr uint64_t EDGES = 0xff818181818181ffULL;
    const int32_t emobP = popcount64_swar(mvP & EDGES);
    const int32_t emobO = popcount64_swar(mvO & EDGES);
    const uint64_t E    = ~(P | O);
    const int32_t potP  = popcount64_swar(adj_mask_bb(O) & E);
    const int32_t potO  = popcount64_swar(adj_mask_bb(P) & E);
    const uint64_t adjE = adj_mask_bb(E);
    const int32_t frP   = popcount64_swar(P & adjE);
    const int32_t frO   = popcount64_swar(O & adjE);
    const int32_t stabP = count_edge_stable_bb(P);
    const int32_t stabO = count_edge_stable_bb(O);

    f_ext[0] = clamp_feat_i32((emobP - emobO) << 4);
    f_ext[1] = clamp_feat_i32((potP - potO) << 4);
    f_ext[2] = clamp_feat_i32((frP - frO) << 4);
    f_ext[3] = clamp_feat_i32((stabP - stabO) << 4);

    // f[12..15]: Smooth linear ramp over ply 24..31 (ramp = clamp(ply - 23, 0, 8), 0 cost when ply <= 23!)
    const int32_t ramp = std::clamp<int32_t>(ply - 23, 0, 8);
    if (ramp == 0) {
        f_ext[4] = 0;
        f_ext[5] = 0;
        f_ext[6] = 0;
        f_ext[7] = 0;
    } else {
        int32_t odd_cnt = 0, single_cnt = 0, odd_mv_diff = 0, reg_cnt = 0;
        const uint64_t adjE4 = ((E << 1) & 0xfefefefefefefefeULL)
                             | ((E >> 1) & 0x7f7f7f7f7f7f7f7fULL)
                             | (E << 8)
                             | (E >> 8);
        const uint64_t singles = E & ~adjE4;
        if (singles) {
            single_cnt  = popcount64_swar(singles);
            odd_cnt     = single_cnt;
            reg_cnt     = single_cnt;
            odd_mv_diff = popcount64_swar(singles & mvP) - popcount64_swar(singles & mvO);
        }
        uint64_t rem = E ^ singles;
        while (rem) {
            uint64_t comp = rem & (~rem + 1ULL);
            uint64_t prev;
            do {
                prev = comp;
                comp |= (((comp << 1) & 0xfefefefefefefefeULL)
                       | ((comp >> 1) & 0x7f7f7f7f7f7f7f7fULL)
                       | (comp << 8)
                       | (comp >> 8)) & rem;
                comp |= (((comp << 1) & 0xfefefefefefefefeULL)
                       | ((comp >> 1) & 0x7f7f7f7f7f7f7f7fULL)
                       | (comp << 8)
                       | (comp >> 8)) & rem;
            } while (comp != prev);
            rem ^= comp;
            reg_cnt++;
            const int32_t sz = (reg_cnt == 1 && rem == 0ULL) ? (60 - ply) : popcount64_swar(comp);
            if (sz & 1) {
                odd_cnt++;
                odd_mv_diff += ((comp & mvP) != 0ULL ? 1 : 0) - ((comp & mvO) != 0ULL ? 1 : 0);
            }
        }
        const int32_t raw0 = (odd_cnt - 1) << 5;
        const int32_t raw1 = single_cnt << 6;
        const int32_t raw2 = odd_mv_diff << 6;
        const int32_t raw3 = (reg_cnt - 2) << 5;
        if (ramp < 8) {
            f_ext[4] = clamp_feat_i32(sym_rshift(raw0 * ramp, 3));
            f_ext[5] = clamp_feat_i32(sym_rshift(raw1 * ramp, 3));
            f_ext[6] = clamp_feat_i32(sym_rshift(raw2 * ramp, 3));
            f_ext[7] = clamp_feat_i32(sym_rshift(raw3 * ramp, 3));
        } else {
            f_ext[4] = clamp_feat_i32(raw0);
            f_ext[5] = clamp_feat_i32(raw1);
            f_ext[6] = clamp_feat_i32(raw2);
            f_ext[7] = clamp_feat_i32(raw3);
        }
    }

    // f[16..23]: Full 64-pattern geometric stats (fused 24-load + 0-load outer_rest_sum)
    int32_t c_min = c[0], c_max = c[0];
    for (int i = 1; i < 4; ++i) {
        if (c[i] < c_min) c_min = c[i];
        if (c[i] > c_max) c_max = c[i];
    }

    int32_t cs[4], es[4];
    for (int i = 0; i < 4; ++i) {
        cs[i] = std::clamp<int32_t>(sym_rshift(c[i], 4), -256, 256);
        es[i] = std::clamp<int32_t>(sym_rshift(e[i], 4), -256, 256);
    }
    const int32_t diag_prod = cs[0] * cs[2] + cs[1] * cs[3];
    const int32_t adj_diff  = abs_i32(cs[0] - cs[1]) + abs_i32(cs[1] - cs[2])
                            + abs_i32(cs[2] - cs[3]) + abs_i32(cs[3] - cs[0]);
    const int32_t diag_diff = abs_i32(cs[0] - cs[2]) + abs_i32(cs[1] - cs[3]);
    const int32_t polar     = adj_diff - 2 * diag_diff;
    const int32_t ce_coup   = cs[0] * (es[0] + es[3]) + cs[1] * (es[1] + es[0])
                            + cs[2] * (es[2] + es[1]) + cs[3] * (es[3] + es[2]);

    f_ext[8]  = clamp_feat_i32(sym_rshift(c_sum, 3));
    f_ext[9]  = clamp_feat_i32(sym_rshift(e_sum, 3));
    f_ext[10] = clamp_feat_i32(sym_rshift(c_max - c_min, 3));
    f_ext[11] = clamp_feat_i32(sym_rshift(diag_prod, 6));
    f_ext[12] = clamp_feat_i32(sym_rshift(polar, 1));
    f_ext[13] = clamp_feat_i32(sym_rshift(ce_coup, 7));
    f_ext[14] = clamp_feat_i32(sym_rshift(ctr10_tier1_sum, 3));
    f_ext[15] = clamp_feat_i32(sym_rshift(outer_rest_sum, 3));
}

// v3.3 Stage 3 Kernel: Key-14 8ch + Center-10 4ch + 44-term Pairwise SCReLU + 4-way j-Tile 32->H->1 MLP (0 SIMD!)
// Expects contiguous f[WACA_V33_F3] with f[0..7] already populated by Stage 2.5 (0 copy!).
template<bool IS_WHITE>
__attribute__((noinline, noclone))
int32_t eval_waca_v33_stage3_ptm(const WacaG2V33Weights& w,
                                 const PatternMeta* __restrict meta,
                                 const uint16_t* __restrict idx,
                                 uint64_t P, uint64_t O,
                                 uint64_t mvP, uint64_t mvO,
                                 int32_t v1,
                                 int32_t* __restrict f,
                                 int32_t r25,
                                 int ply) noexcept {
    const int16_t* __restrict wp = w.tier1.w_ply[ply].data();
    const uint8_t eo_blk = PLY_TO_EO30_BLK[ply];
    const WacaKeyEmb8* __restrict htbl = w.h_emb[eo_blk].data();
    const WacaKeyEmb4* __restrict ctbl = w.ctr_emb[eo_blk].data();

    // 1) Fused p=0..13 loop: single flat index loads BOTH L1D-hot wp[flat] & 8-byte htbl[flat]!
    int32_t c[4], e[4];
    int32_t c_sum = 0, e_sum = 0, l2d8_sum = 0;
    int32_t h[NUM_KEY_PATTERNS][8];
    int32_t g[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int p = 0; p < 4; ++p) {
        const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        const int32_t wv = wp[flat];
        c[p] = wv;
        c_sum += wv;
        const WacaKeyEmb8 emb = htbl[flat];
        for (int k = 0; k < 8; ++k) {
            const int32_t ev = emb.h[k];
            h[p][k] = ev;
            g[k] += ev;
        }
    }
    for (int p = 4; p < 8; ++p) {
        const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        const int32_t wv = wp[flat];
        e[p - 4] = wv;
        e_sum += wv;
        const WacaKeyEmb8 emb = htbl[flat];
        for (int k = 0; k < 8; ++k) {
            const int32_t ev = emb.h[k];
            h[p][k] = ev;
            g[k] += ev;
        }
    }
    for (int p = 8; p < NUM_KEY_PATTERNS; ++p) {
        const int32_t flat = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        l2d8_sum += wp[flat];
        const WacaKeyEmb8 emb = htbl[flat];
        for (int k = 0; k < 8; ++k) {
            const int32_t ev = emb.h[k];
            h[p][k] = ev;
            g[k] += ev;
        }
    }

    // 2) Fused Center-10 loop (p=14..17 Line3, p=18..21 Line4, p=62..63 Center2x4)
    int32_t ctr10_tier1_sum = 0;
    int32_t gc0 = 0, gc1 = 0, gc2 = 0, gc3 = 0;
    for (int p = 14; p < 18; ++p) {
        const int32_t flat  = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        const int32_t cflat = IS_WHITE ? (6560 - static_cast<int32_t>(idx[p])) : static_cast<int32_t>(idx[p]);
        ctr10_tier1_sum += wp[flat];
        const WacaKeyEmb4 ce = ctbl[cflat];
        gc0 += ce.h[0]; gc1 += ce.h[1]; gc2 += ce.h[2]; gc3 += ce.h[3];
    }
    for (int p = 18; p < 22; ++p) {
        const int32_t flat  = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        const int32_t cflat = IS_WHITE ? (13121 - static_cast<int32_t>(idx[p])) : (6561 + static_cast<int32_t>(idx[p]));
        ctr10_tier1_sum += wp[flat];
        const WacaKeyEmb4 ce = ctbl[cflat];
        gc0 += ce.h[0]; gc1 += ce.h[1]; gc2 += ce.h[2]; gc3 += ce.h[3];
    }
    for (int p = 62; p < 64; ++p) {
        const int32_t flat  = IS_WHITE ? (meta[p].raw_off_end - idx[p]) : (meta[p].raw_off + idx[p]);
        const int32_t cflat = IS_WHITE ? (19682 - static_cast<int32_t>(idx[p])) : (13122 + static_cast<int32_t>(idx[p]));
        ctr10_tier1_sum += wp[flat];
        const WacaKeyEmb4 ce = ctbl[cflat];
        gc0 += ce.h[0]; gc1 += ce.h[1]; gc2 += ce.h[2]; gc3 += ce.h[3];
    }
    // 0-load exact restoration of remaining 40 outer patterns!
    const int32_t outer_rest_sum = v1 - (c_sum + e_sum + l2d8_sum + ctr10_tier1_sum);

    // 3) Populate f[8..23] in-place (f[0..7] already written by Stage 2.5 -> 0 copy, 0 xmm!)
    extract_waca_v33_stage3_extra_features(P, O, mvP, mvO, ply, c, e,
                                           c_sum, e_sum, ctr10_tier1_sum, outer_rest_sum, f + 8);

    // 4) 44-term SCReLU + Pairwise + Center-Outer Cross-Coupling Head
    const WacaV33SCReLUPly& sc = w.screlu[ply];
    int32_t s0_sq[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int32_t p0[4]    = {0, 0, 0, 0};
    for (int r = 0; r < 4; ++r) {
        const int c1 = (r + 1) & 3;
        int32_t u[8];
        for (int k = 0; k < 8; ++k) {
            u[k] = clamp_i32(h[4 + r][k] + h[8 + r][k] + h[r][k] + h[c1][k] + sc.mu_key[0][k], SCRELU_Q);
            s0_sq[k] += u[k] * u[k];
        }
        p0[0] += u[0] * u[4];
        p0[1] += u[1] * u[5];
        p0[2] += u[2] * u[6];
        p0[3] += u[3] * u[7];
    }

    int32_t s1_sq[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int32_t p1[4]    = {0, 0, 0, 0};
    for (int d = 0; d < 2; ++d) {
        int32_t u[8];
        for (int k = 0; k < 8; ++k) {
            u[k] = clamp_i32(h[12 + d][k] + h[d][k] + h[d + 2][k] + sc.mu_key[1][k], SCRELU_Q);
            s1_sq[k] += u[k] * u[k];
        }
        p1[0] += u[0] * u[4];
        p1[1] += u[1] * u[5];
        p1[2] += u[2] * u[6];
        p1[3] += u[3] * u[7];
    }

    int32_t ug[8], s2_sq[8], p2[4];
    for (int k = 0; k < 8; ++k) {
        ug[k]    = clamp_i32(g[k] + sc.mu_key[2][k], SCRELU_Q);
        s2_sq[k] = ug[k] * ug[k];
    }
    p2[0] = ug[0] * ug[4];
    p2[1] = ug[1] * ug[5];
    p2[2] = ug[2] * ug[6];
    p2[3] = ug[3] * ug[7];

    const int32_t uc0 = clamp_i32(gc0 + sc.mu_ctr[0], SCRELU_Q);
    const int32_t uc1 = clamp_i32(gc1 + sc.mu_ctr[1], SCRELU_Q);
    const int32_t uc2 = clamp_i32(gc2 + sc.mu_ctr[2], SCRELU_Q);
    const int32_t uc3 = clamp_i32(gc3 + sc.mu_ctr[3], SCRELU_Q);
    const int32_t s3_sq[4] = {uc0 * uc0, uc1 * uc1, uc2 * uc2, uc3 * uc3};
    const int32_t pc[4]    = {ug[0] * uc0, ug[1] * uc1, ug[2] * uc2, ug[3] * uc3};

    int32_t sc_acc = 0;
    for (int k = 0; k < 8; ++k) sc_acc += static_cast<int32_t>(sc.beta[0 + k])  * s0_sq[k];
    for (int k = 0; k < 4; ++k) sc_acc += static_cast<int32_t>(sc.beta[8 + k])  * p0[k];
    for (int k = 0; k < 8; ++k) sc_acc += static_cast<int32_t>(sc.beta[12 + k]) * s1_sq[k];
    for (int k = 0; k < 4; ++k) sc_acc += static_cast<int32_t>(sc.beta[20 + k]) * p1[k];
    for (int k = 0; k < 8; ++k) sc_acc += static_cast<int32_t>(sc.beta[24 + k]) * s2_sq[k];
    for (int k = 0; k < 4; ++k) sc_acc += static_cast<int32_t>(sc.beta[32 + k]) * p2[k];
    for (int k = 0; k < 4; ++k) sc_acc += static_cast<int32_t>(sc.beta[36 + k]) * s3_sq[k];
    for (int k = 0; k < 4; ++k) sc_acc += static_cast<int32_t>(sc.beta[40 + k]) * pc[k];
    const int32_t r_key_screlu = sym_rshift(sc_acc, SCRELU_SHIFT);

    // 5) f[24..31]: 8 sparse nonlinear summary features
    for (int k = 0; k < 4; ++k) {
        const int32_t sum_k = (s0_sq[k] + s1_sq[k] + s2_sq[k]) + (s0_sq[k + 4] + s1_sq[k + 4] + s2_sq[k + 4]);
        f[24 + k] = clamp_feat_i32(sum_k >> 9);
    }
    f[28] = clamp_feat_i32((p0[0] + p1[0] + p2[0] + p0[1] + p1[1] + p2[1]) >> 9);
    f[29] = clamp_feat_i32((p0[2] + p1[2] + p2[2] + p0[3] + p1[3] + p2[3]) >> 9);
    f[30] = clamp_feat_i32((s3_sq[0] + s3_sq[1] + s3_sq[2] + s3_sq[3]) >> 8);
    f[31] = clamp_feat_i32((pc[0] + pc[1] + pc[2] + pc[3]) >> 8);

    // 6) 60-ply Compact Low-Jitter Pure-Integer SCReLU MLP (32 -> H[=16] -> 1, 4-way j-tile scalar ILP!)
    const WacaV33MlpPly& mp = w.mlp[ply];
    const int H = w.mlp_hidden;
    int32_t acc2 = mp.b2;
    int j = 0;
    for (; j + 3 < H; j += 4) {
        const int8_t* __restrict w0 = mp.w1[j + 0];
        const int8_t* __restrict w1 = mp.w1[j + 1];
        const int8_t* __restrict w2 = mp.w1[j + 2];
        const int8_t* __restrict w3 = mp.w1[j + 3];
        int32_t z0 = mp.b1[j + 0];
        int32_t z1 = mp.b1[j + 1];
        int32_t z2 = mp.b1[j + 2];
        int32_t z3 = mp.b1[j + 3];
        for (int i = 0; i < WACA_V33_F3; ++i) {
            const int32_t fv = f[i];
            z0 += static_cast<int32_t>(w0[i]) * fv;
            z1 += static_cast<int32_t>(w1[i]) * fv;
            z2 += static_cast<int32_t>(w2[i]) * fv;
            z3 += static_cast<int32_t>(w3[i]) * fv;
        }
        const int32_t u0 = (z0 <= 0) ? 0 : (((z0 >> WACA_MLP_S1) > SCRELU_Q) ? SCRELU_Q : (z0 >> WACA_MLP_S1));
        const int32_t u1 = (z1 <= 0) ? 0 : (((z1 >> WACA_MLP_S1) > SCRELU_Q) ? SCRELU_Q : (z1 >> WACA_MLP_S1));
        const int32_t u2 = (z2 <= 0) ? 0 : (((z2 >> WACA_MLP_S1) > SCRELU_Q) ? SCRELU_Q : (z2 >> WACA_MLP_S1));
        const int32_t u3 = (z3 <= 0) ? 0 : (((z3 >> WACA_MLP_S1) > SCRELU_Q) ? SCRELU_Q : (z3 >> WACA_MLP_S1));
        acc2 += static_cast<int32_t>(mp.w2[j + 0]) * (u0 * u0)
              + static_cast<int32_t>(mp.w2[j + 1]) * (u1 * u1)
              + static_cast<int32_t>(mp.w2[j + 2]) * (u2 * u2)
              + static_cast<int32_t>(mp.w2[j + 3]) * (u3 * u3);
    }
    for (; j < H; ++j) {
        const int8_t* __restrict w0 = mp.w1[j];
        int32_t z = mp.b1[j];
        for (int i = 0; i < WACA_V33_F3; ++i) {
            z += static_cast<int32_t>(w0[i]) * f[i];
        }
        const int32_t u = (z <= 0) ? 0 : (((z >> WACA_MLP_S1) > SCRELU_Q) ? SCRELU_Q : (z >> WACA_MLP_S1));
        acc2 += static_cast<int32_t>(mp.w2[j]) * (u * u);
    }
    const int32_t r_mlp = sym_rshift(acc2, WACA_MLP_S2);

    return std::clamp<int32_t>(r25 + r_mlp + r_key_screlu, -WACA_MAX_RES, WACA_MAX_RES);
}

// Evaluate all 3 stages unconditionally (used for bit-exact testing, Val RMSE, and GateMode::FULL)
inline void eval_waca_g2_v33_stages(const WacaG2V33Weights& w,
                                    const PatternMeta* __restrict meta,
                                    const uint16_t* __restrict idx,
                                    uint64_t B, uint64_t W, int stm, int ply,
                                    int32_t& out_v1, int32_t& out_v25, int32_t& out_v3) noexcept {
    const bool is_w = (stm < 0);
    const uint64_t P = is_w ? W : B;
    const uint64_t O = is_w ? B : W;
    alignas(64) int32_t f[WACA_V33_F3];
    uint64_t mvP = 0, mvO = 0;
    const int32_t v1  = is_w ? eval_waca_v33_stage1_ptm<true>(w.tier1, meta, idx, ply)
                             : eval_waca_v33_stage1_ptm<false>(w.tier1, meta, idx, ply);
    const int32_t r25 = eval_waca_v33_stage25_kernel(w, P, O, v1, ply, f, mvP, mvO);
    const int32_t rt  = is_w ? eval_waca_v33_stage3_ptm<true>(w, meta, idx, P, O, mvP, mvO, v1, f, r25, ply)
                             : eval_waca_v33_stage3_ptm<false>(w, meta, idx, P, O, mvP, mvO, v1, f, r25, ply);
    out_v1  = v1;
    out_v25 = v1 + r25;
    out_v3  = v1 + rt;
}

inline int32_t eval_waca_g2_v33_full(const WacaG2V33Weights& w,
                                     const PatternMeta* __restrict meta,
                                     const uint16_t* __restrict idx,
                                     uint64_t B, uint64_t W, int stm, int ply) noexcept {
    int32_t v1 = 0, v25 = 0, v3 = 0;
    eval_waca_g2_v33_stages(w, meta, idx, B, W, stm, ply, v1, v25, v3);
    return v3;
}

// Search-integrated 2-stage gated evaluator for WACA_G2_V33 (SAFE_BND / FAIL_HARD / FULL / WIDE_SAFE_BND)
inline int32_t eval_waca_g2_v33_gated(const WacaG2V33Weights& w,
                                      const PatternMeta* __restrict meta,
                                      const uint16_t* __restrict idx,
                                      uint64_t B, uint64_t W, int stm, int ply,
                                      int32_t alpha, int32_t beta,
                                      bool in_mpc_probe,
                                      GateMode gate_mode,
                                      int32_t m25_units,
                                      int32_t m3_units,
                                      uint64_t& stat_s25,
                                      uint64_t& stat_s3,
                                      int32_t tt_guard_units = 576) noexcept {
    const bool is_w = (stm < 0);
    const int32_t v1 = is_w ? eval_waca_v33_stage1_ptm<true>(w.tier1, meta, idx, ply)
                            : eval_waca_v33_stage1_ptm<false>(w.tier1, meta, idx, ply);

    if (in_mpc_probe) return v1;

    const bool is_pv = (beta > alpha + 1);
    if (!is_pv && gate_mode != GateMode::FULL) {
        if (v1 + m25_units <= alpha) return gate_return_low(gate_mode, v1, m25_units, tt_guard_units, alpha);
        if (v1 - m25_units >= beta)  return gate_return_high(gate_mode, v1, m25_units, tt_guard_units, beta);
    }

    // 12-dim polynomial Stage 2.5 (~62 ns, 0 Tier-1 reloads, 3 SWAR popcounts!)
    stat_s25++;
    const uint64_t P = is_w ? W : B;
    const uint64_t O = is_w ? B : W;
    alignas(64) int32_t f[WACA_V33_F3];
    uint64_t mvP = 0, mvO = 0;
    const int32_t r25 = eval_waca_v33_stage25_kernel(w, P, O, v1, ply, f, mvP, mvO);
    const int32_t v25 = v1 + r25;

    if (!is_pv && gate_mode != GateMode::FULL) {
        if (v25 + m3_units <= alpha) return gate_return_low(gate_mode, v25, m3_units, tt_guard_units, alpha);
        if (v25 - m3_units >= beta)  return gate_return_high(gate_mode, v25, m3_units, tt_guard_units, beta);
    }

    // Sparse-rich Stage 3 (~5% leaves: Key-14 8ch + Center-10 4ch + 44-term Pairwise SCReLU + H=16 j-tile MLP)
    stat_s3++;
    const int32_t rt = is_w ? eval_waca_v33_stage3_ptm<true>(w, meta, idx, P, O, mvP, mvO, v1, f, r25, ply)
                            : eval_waca_v33_stage3_ptm<false>(w, meta, idx, P, O, mvP, mvO, v1, f, r25, ply);
    return v1 + rt;
}

} // namespace p2c64
