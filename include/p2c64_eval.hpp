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
// Branchless, Zero-Multiply Integer Primitives
// ============================================================================
static inline constexpr int32_t neg_if(int32_t v, int32_t flip) noexcept {
    return (v ^ -flip) + flip; // flip in {0, 1}; exact for all int32 except INT32_MIN
}

// Mask-based branchless negation: mask is 0 or -1 (all 1s from arithmetic right shift)
static inline constexpr int32_t neg_mask(int32_t v, int32_t mask) noexcept {
    return (v ^ mask) - mask;
}

static inline constexpr int32_t clamp_i32(int32_t x, int32_t hi) noexcept {
    return (x < 0) ? 0 : ((x > hi) ? hi : x);
}

// Round-toward-zero symmetric right shift: sym_rshift(-x, S) == -sym_rshift(x, S)
static inline constexpr int32_t sym_rshift(int32_t x, int shift) noexcept {
    return (x + ((x >> 31) & ((1 << shift) - 1))) >> shift;
}

// Exact 3-bit color-inversion involution on {0..7}:
//   3 color-dual pairs: {0<->1, 2<->3, 4<->5}
//   2 self-dual neutral fixed points: {6<->6, 7<->7} (for all-empty & anti-mirror rev(x)==x_bar)
static inline constexpr uint32_t color_inv_z3(uint32_t z, int32_t flip) noexcept {
    const uint32_t toggle = static_cast<uint32_t>(flip) & ((~((z >> 1) & (z >> 2))) & 1U);
    return z ^ toggle;
}

static inline constexpr uint32_t color_inv_z3_mask(uint32_t z, int32_t mask) noexcept {
    const uint32_t toggle = static_cast<uint32_t>(mask) & ((~((z >> 1) & (z >> 2))) & 1U);
    return z ^ toggle;
}

// Constexpr ply -> (block, sub) LUTs to avoid magic-multiply `imul` for ply/6!
static constexpr std::array<uint8_t, NUM_PLIES> PLY_TO_BLK = {
    0,0,0,0,0,0, 1,1,1,1,1,1, 2,2,2,2,2,2, 3,3,3,3,3,3, 4,4,4,4,4,4,
    5,5,5,5,5,5, 6,6,6,6,6,6, 7,7,7,7,7,7, 8,8,8,8,8,8, 9,9,9,9,9,9
};
static constexpr std::array<uint8_t, NUM_PLIES> PLY_TO_SUB = {
    0,1,2,3,4,5, 0,1,2,3,4,5, 0,1,2,3,4,5, 0,1,2,3,4,5, 0,1,2,3,4,5,
    0,1,2,3,4,5, 0,1,2,3,4,5, 0,1,2,3,4,5, 0,1,2,3,4,5, 0,1,2,3,4,5
};

static constexpr int32_t CORNER_CANON_SIZE = (POW3[9] + 1) / 2; // 9,842
static constexpr int32_t CORNER_RAW_SIZE   = POW3[9];           // 19,683

// ============================================================================
// Core 16-Byte Block-Packed Entry for Key 14 Nodes (4 entries per 64B cache line)
// ============================================================================
struct alignas(16) BlockEntry16 {
    int16_t w[PLIES_PER_BLOCK]; // 12B: 6 consecutive plies of antisymmetric linear weight W
    uint8_t aux[4];             //  4B: 100% dedicated to nonlinear residual data of Key 14 nodes
};
static_assert(sizeof(BlockEntry16) == 16 && alignof(BlockEntry16) == 16, "Must be 16B aligned");

enum class EvalMode : int {
    RAW_NTUPLE_64            = 0,
    LINEAR_64                = 1,
    CAND_A_SCRELU_K2         = 2,
    CAND_B_SCONED_3W3B_NAIVE = 3,
    CAND_B_SCONED_3W3B       = 4,
    CAND_C_HYBRID            = 5
};

// ============================================================================
// SCReLU Parameters (per block, shared across 3 D4-invariant region groups)
//   Group 0: 4 Edge strips  (r = 0..3)
//   Group 1: 2 Diag strips  (r = 4..5)
//   Group 2: 1 Global sum   (r = 6)
// Safety bounds: Q = 127, |beta| <= 4095 => max |sum_g| <= 7*3*4095*16129 = 1.387e9 < INT32_MAX
// ============================================================================
static constexpr int32_t SCRELU_Q        = 127;
static constexpr int32_t SCRELU_BETA_MAX = 4095;
static constexpr int     SCRELU_SHIFT    = 7;

struct SCReLUBlockParams {
    int32_t mu[3][2]      = {{32, 32}, {32, 32}, {32, 32}};
    int32_t beta_sq[3][2] = {{0, 0}, {0, 0}, {0, 0}};
    int32_t beta_pr[3]    = {0, 0, 0};
};

// ============================================================================
// SCONE-D 3way-3bit Macro-LUT Block (3 tables x 512 int16 = 3.0 KiB / block)
//   T_CW[512]: (C3x3[i], E2X[i].fwd, E2X[(i+3)&3].rev)   x4
//   T_EA[512]: (E2X[i].fwd, E2X[i].rev, L2[i])           x4
//   T_CD[512]: (C3x3[d], Diag8[d], C3x3[d+2])            x2
// ============================================================================
struct alignas(64) MacroLUTBlock {
    int16_t t_cw[512]{};
    int16_t t_ea[512]{};
    int16_t t_cd[512]{};
};

// ============================================================================
// Traditional 2B-per-ply Raw N-Tuple Weights (Fair Benchmark Baseline)
//   All 64 patterns: 2B int16_t per ply (46.92 MiB)
//   4-Corner tempo:  1B int8_t per ply  ( 1.13 MiB, apples-to-apples with P2C-64)
// ============================================================================
struct RawNTuple64Weights {
    std::array<std::vector<int16_t>, NUM_PLIES> w_ply;
    std::array<std::vector<int8_t>,  NUM_PLIES> corner_tempo;

    RawNTuple64Weights() {
        const auto& pt = get_pattern_tables();
        for (int p = 0; p < NUM_PLIES; ++p) {
            w_ply[p].assign(static_cast<size_t>(pt.total_raw_entries), 0);
            corner_tempo[p].assign(static_cast<size_t>(CORNER_RAW_SIZE), 0);
        }
    }

    // Symmetrize raw 3^len tables under all internal pattern permutations AND Z2 color inversion
    void symmetrize() {
        const auto& pt = get_pattern_tables();
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            int16_t* wp = w_ply[ply].data();
            int8_t*  tp = corner_tempo[ply].data();
            for (int t = 0; t < NUM_PATTERN_TYPES; ++t) {
                const int32_t off  = pt.type_raw_off[t];
                const int32_t sz   = pt.type_raw_size[t];
                const int32_t maxv = sz - 1;
                const int ng = pt.type_sym[t].num_gens;
                std::vector<int8_t> vis(static_cast<size_t>(sz), 0);
                for (int32_t x = 0; x < sz; ++x) {
                    if (vis[x]) continue;
                    std::vector<std::pair<int32_t, int>> orb;
                    orb.push_back({x, 1});
                    vis[x] = 1;
                    bool force_zero_w = false;
                    for (size_t qi = 0; qi < orb.size(); ++qi) {
                        int32_t cur = orb[qi].first;
                        int csgn    = orb[qi].second;
                        int32_t inv = maxv - cur;
                        if (!vis[inv]) {
                            vis[inv] = static_cast<int8_t>(csgn > 0 ? -1 : 1);
                            orb.push_back({inv, -csgn});
                        } else if (vis[inv] != -csgn) {
                            force_zero_w = true;
                        }
                        for (int g = 0; g < ng; ++g) {
                            int32_t nxt = pt.permute_raw_index(t, g, cur);
                            if (!vis[nxt]) {
                                vis[nxt] = static_cast<int8_t>(csgn);
                                orb.push_back({nxt, csgn});
                            } else if (vis[nxt] != csgn) {
                                force_zero_w = true;
                            }
                        }
                    }
                    int16_t raw_w = wp[off + x];
                    if (raw_w == INT16_MIN) raw_w = -32767;
                    int16_t rep_w = force_zero_w ? int16_t(0) : raw_w;
                    int8_t  rep_t = (t == 0) ? tp[x] : int8_t(0);
                    for (const auto& [idx, sgn] : orb) {
                        wp[off + idx] = (sgn > 0) ? rep_w : static_cast<int16_t>(-rep_w);
                        if (t == 0) tp[idx] = rep_t;
                    }
                }
            }
        }
    }
};

// ============================================================================
// P2C-64 v1.1 Asymmetric Memory Layout Weights Container (43.51 MiB total)
//   - Key 14 nodes (types 0..3):     16B BlockEntry16 x 45,929 x 10 blk =  7.01 MiB
//   - Non-key 50 nodes (types 4..15): 2B int16_t x 318,168 x 60 plies   = 36.41 MiB
//   - Corner-4 Tempo T(B):            1B int8_t x 9,842 x 10 blk        = 96.1 KiB (9.6 KiB/blk)
//   - SCONE-D Macro-LUTs:             3.0 KiB x 10 blk                  = 30.0 KiB
// ============================================================================
struct P2C64Weights {
    std::array<std::vector<BlockEntry16>, NUM_BLOCKS> key_blocks;
    std::array<std::vector<int16_t>,      NUM_PLIES>  nonkey_w;
    std::array<std::vector<int8_t>,       NUM_BLOCKS> corner_tempo;
    std::array<SCReLUBlockParams, NUM_BLOCKS>         screlu{};
    std::array<MacroLUTBlock,     NUM_BLOCKS>         macro{};

    P2C64Weights() {
        const auto& pt = get_pattern_tables();
        for (int b = 0; b < NUM_BLOCKS; ++b) {
            key_blocks[b].resize(static_cast<size_t>(pt.key_canon_entries));
            std::memset(key_blocks[b].data(), 0, key_blocks[b].size() * sizeof(BlockEntry16));
            corner_tempo[b].assign(static_cast<size_t>(CORNER_CANON_SIZE), 0);
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            nonkey_w[ply].assign(static_cast<size_t>(pt.nonkey_raw_entries), 0);
        }
    }

    // Symmetrize all Macro-LUTs for exact D4 mirror + Z2 color-inversion antisymmetry
    void symmetrize_macro_luts() {
        auto sym_table = [](int16_t tbl[512], bool swap_12, bool swap_01, bool swap_02) {
            std::array<int8_t, 512> vis{};
            for (int k = 0; k < 512; ++k) {
                if (vis[k]) continue;
                std::vector<std::pair<int, int>> orb;
                orb.push_back({k, 1});
                vis[k] = 1;
                bool force_zero = false;
                for (size_t qi = 0; qi < orb.size(); ++qi) {
                    int ck = orb[qi].first;
                    int sg = orb[qi].second;
                    int z0 = (ck >> 6) & 7, z1 = (ck >> 3) & 7, z2 = ck & 7;

                    int iz0 = static_cast<int>(color_inv_z3(static_cast<uint32_t>(z0), 1));
                    int iz1 = static_cast<int>(color_inv_z3(static_cast<uint32_t>(z1), 1));
                    int iz2 = static_cast<int>(color_inv_z3(static_cast<uint32_t>(z2), 1));
                    int ik  = (iz0 << 6) | (iz1 << 3) | iz2;
                    if (!vis[ik]) { vis[ik] = static_cast<int8_t>(-sg); orb.push_back({ik, -sg}); }
                    else if (vis[ik] != -sg) force_zero = true;

                    auto try_mirror = [&](int mk) {
                        if (!vis[mk]) { vis[mk] = static_cast<int8_t>(sg); orb.push_back({mk, sg}); }
                        else if (vis[mk] != sg) force_zero = true;
                    };
                    if (swap_12) try_mirror((z0 << 6) | (z2 << 3) | z1);
                    if (swap_01) try_mirror((z1 << 6) | (z0 << 3) | z2);
                    if (swap_02) try_mirror((z2 << 6) | (z1 << 3) | z0);
                }
                int16_t rep = force_zero ? int16_t(0) : tbl[k];
                if (rep == INT16_MIN) rep = -32767;
                for (const auto& [idx, sg] : orb) {
                    tbl[idx] = (sg > 0) ? rep : static_cast<int16_t>(-rep);
                }
            }
        };
        for (int b = 0; b < NUM_BLOCKS; ++b) {
            sym_table(macro[b].t_cw, true,  false, false);
            sym_table(macro[b].t_ea, false, true,  false);
            sym_table(macro[b].t_cd, false, false, true);
        }
    }

    // Symmetrize Asymmetric Tables (Key-14 BlockEntry16 + Corner-4 Tempo + Non-Key-50 2B Direct + MacroLUT)
    void symmetrize_for_mode(EvalMode mode) {
        const auto& pt = get_pattern_tables();
        auto clamp_safe_i16 = [](int16_t v) -> int16_t { return (v == INT16_MIN) ? int16_t(-32767) : v; };
        auto clamp_safe_i8  = [](int8_t  v) -> int8_t  { return (v == INT8_MIN)  ? int8_t(-127)    : v; };

        // 1) Key 4 pattern types (0..3) in BlockEntry16 + Corner-4 Tempo (type 0)
        for (int b = 0; b < NUM_BLOCKS; ++b) {
            BlockEntry16* tbl = key_blocks[b].data();
            int8_t*       ct  = corner_tempo[b].data();

            for (int t = 0; t < 4; ++t) {
                const int32_t off  = pt.type_canon_off[t];
                const int32_t csz  = pt.type_canon_size[t];
                const int32_t maxv = POW3[pt.type_len[t]] - 1;
                const int32_t half = maxv >> 1;
                const int     ng   = pt.type_sym[t].num_gens;

                std::vector<int8_t> vis(static_cast<size_t>(csz), 0);
                for (int32_t cx = 0; cx < csz; ++cx) {
                    if (vis[cx]) continue;
                    std::vector<std::pair<int32_t, int>> orb;
                    orb.push_back({cx, 1});
                    vis[cx] = 1;
                    bool anti_mirror_or_self_dual = (cx == half);
                    for (size_t qi = 0; qi < orb.size(); ++qi) {
                        int32_t cur = orb[qi].first;
                        int csgn    = orb[qi].second;
                        for (int g = 0; g < ng; ++g) {
                            int32_t y  = pt.permute_raw_index(t, g, cur);
                            int fl     = (y > half) ? 1 : 0;
                            int32_t cy = fl ? (maxv - y) : y;
                            int nsgn   = fl ? -csgn : csgn;
                            if (cy == half) anti_mirror_or_self_dual = true;
                            if (!vis[cy]) {
                                vis[cy] = static_cast<int8_t>(nsgn);
                                orb.push_back({cy, nsgn});
                            } else if (vis[cy] != nsgn) {
                                anti_mirror_or_self_dual = true;
                            }
                        }
                    }

                    // Linear weights w[6]
                    BlockEntry16 rep = tbl[off + cx];
                    for (int k = 0; k < PLIES_PER_BLOCK; ++k) {
                        int16_t v = anti_mirror_or_self_dual ? int16_t(0) : clamp_safe_i16(rep.w[k]);
                        for (const auto& [idx, sgn] : orb) {
                            tbl[off + idx].w[k] = (sgn > 0) ? v : static_cast<int16_t>(-v);
                        }
                    }
                    // Corner-4 color-symmetric tempo T(B) (type 0 only)
                    if (t == 0) {
                        int8_t rep_t = ct[cx];
                        for (const auto& [idx, sgn] : orb) {
                            (void)sgn;
                            ct[idx] = rep_t;
                        }
                    }

                    // Key-14 nonlinear aux[4]
                    if (mode == EvalMode::CAND_A_SCRELU_K2) {
                        int8_t o0 = anti_mirror_or_self_dual ? int8_t(0) : clamp_safe_i8(static_cast<int8_t>(rep.aux[0]));
                        int8_t o1 = anti_mirror_or_self_dual ? int8_t(0) : clamp_safe_i8(static_cast<int8_t>(rep.aux[1]));
                        uint8_t e0 = rep.aux[2], e1 = rep.aux[3];
                        for (const auto& [idx, sgn] : orb) {
                            tbl[off + idx].aux[0] = static_cast<uint8_t>((sgn > 0) ? o0 : static_cast<int8_t>(-o0));
                            tbl[off + idx].aux[1] = static_cast<uint8_t>((sgn > 0) ? o1 : static_cast<int8_t>(-o1));
                            tbl[off + idx].aux[2] = e0;
                            tbl[off + idx].aux[3] = e1;
                        }
                    } else if (mode == EvalMode::CAND_B_SCONED_3W3B || mode == EvalMode::CAND_B_SCONED_3W3B_NAIVE) {
                        if (t == 1) {
                            uint8_t zf = rep.aux[0] & 7U;
                            uint8_t zr = rep.aux[1] & 7U;
                            if (cx == half) {
                                uint8_t zsd = static_cast<uint8_t>(6U | (zf & 1U));
                                tbl[off + cx].aux[0] = zsd;
                                tbl[off + cx].aux[1] = zsd;
                            } else if (orb.size() == 1) {
                                int32_t y = pt.permute_raw_index(t, 0, cx);
                                int fl_y  = (y > half) ? 1 : 0;
                                tbl[off + cx].aux[0] = zf;
                                tbl[off + cx].aux[1] = fl_y ? static_cast<uint8_t>(color_inv_z3(zf, 1)) : zf;
                            } else {
                                int32_t cy = orb[1].first;
                                int sgn_y  = orb[1].second;
                                tbl[off + cx].aux[0] = zf;
                                tbl[off + cx].aux[1] = zr;
                                tbl[off + cy].aux[0] = (sgn_y > 0) ? zr : static_cast<uint8_t>(color_inv_z3(zr, 1));
                                tbl[off + cy].aux[1] = (sgn_y > 0) ? zf : static_cast<uint8_t>(color_inv_z3(zf, 1));
                            }
                        } else {
                            uint8_t z = rep.aux[0] & 7U;
                            if (anti_mirror_or_self_dual) z = static_cast<uint8_t>(6U | (z & 1U));
                            for (const auto& [idx, sgn] : orb) {
                                uint8_t zi = (sgn > 0) ? z : static_cast<uint8_t>(color_inv_z3(z, 1));
                                tbl[off + idx].aux[0] = zi;
                                tbl[off + idx].aux[1] = zi;
                            }
                        }
                    } else if (mode == EvalMode::CAND_C_HYBRID) {
                        int8_t  o0 = anti_mirror_or_self_dual ? int8_t(0) : clamp_safe_i8(static_cast<int8_t>(rep.aux[0]));
                        uint8_t e0 = rep.aux[1];
                        uint8_t zf = rep.aux[2] & 7U;
                        uint8_t zr = (rep.aux[2] >> 3) & 7U;
                        if (t == 1) {
                            if (cx == half) {
                                uint8_t zsd = static_cast<uint8_t>(6U | (zf & 1U));
                                tbl[off + cx].aux[0] = 0;
                                tbl[off + cx].aux[1] = e0;
                                tbl[off + cx].aux[2] = static_cast<uint8_t>(zsd | (zsd << 3));
                                tbl[off + cx].aux[3] = 0;
                            } else if (orb.size() == 1) {
                                int32_t y = pt.permute_raw_index(t, 0, cx);
                                int fl_y  = (y > half) ? 1 : 0;
                                uint8_t zrev = fl_y ? static_cast<uint8_t>(color_inv_z3(zf, 1)) : zf;
                                tbl[off + cx].aux[0] = static_cast<uint8_t>(o0);
                                tbl[off + cx].aux[1] = e0;
                                tbl[off + cx].aux[2] = static_cast<uint8_t>(zf | (zrev << 3));
                                tbl[off + cx].aux[3] = 0;
                            } else {
                                int32_t cy = orb[1].first;
                                int sgn_y  = orb[1].second;
                                uint8_t y_zf = (sgn_y > 0) ? zr : static_cast<uint8_t>(color_inv_z3(zr, 1));
                                uint8_t y_zr = (sgn_y > 0) ? zf : static_cast<uint8_t>(color_inv_z3(zf, 1));
                                tbl[off + cx].aux[0] = static_cast<uint8_t>(o0);
                                tbl[off + cx].aux[1] = e0;
                                tbl[off + cx].aux[2] = static_cast<uint8_t>(zf | (zr << 3));
                                tbl[off + cx].aux[3] = 0;
                                tbl[off + cy].aux[0] = static_cast<uint8_t>((sgn_y > 0) ? o0 : static_cast<int8_t>(-o0));
                                tbl[off + cy].aux[1] = e0;
                                tbl[off + cy].aux[2] = static_cast<uint8_t>(y_zf | (y_zr << 3));
                                tbl[off + cy].aux[3] = 0;
                            }
                        } else {
                            uint8_t z = zf;
                            if (anti_mirror_or_self_dual) z = static_cast<uint8_t>(6U | (z & 1U));
                            for (const auto& [idx, sgn] : orb) {
                                uint8_t zi = (sgn > 0) ? z : static_cast<uint8_t>(color_inv_z3(z, 1));
                                tbl[off + idx].aux[0] = static_cast<uint8_t>((sgn > 0) ? o0 : static_cast<int8_t>(-o0));
                                tbl[off + idx].aux[1] = e0;
                                tbl[off + idx].aux[2] = static_cast<uint8_t>(zi | (zi << 3));
                                tbl[off + idx].aux[3] = 0;
                            }
                        }
                    }
                }
            }
            for (int g = 0; g < 3; ++g) {
                screlu[b].beta_sq[g][0] = std::clamp(screlu[b].beta_sq[g][0], -SCRELU_BETA_MAX, SCRELU_BETA_MAX);
                screlu[b].beta_sq[g][1] = std::clamp(screlu[b].beta_sq[g][1], -SCRELU_BETA_MAX, SCRELU_BETA_MAX);
                screlu[b].beta_pr[g]    = std::clamp(screlu[b].beta_pr[g],    -SCRELU_BETA_MAX, SCRELU_BETA_MAX);
            }
        }

        // 2) Non-key 12 pattern types (4..15) in 2B direct-indexed per-ply arrays
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            int16_t* wp = nonkey_w[ply].data();
            for (int t = 4; t < NUM_PATTERN_TYPES; ++t) {
                const int32_t off  = pt.type_nonkey_off[t];
                const int32_t sz   = pt.type_raw_size[t];
                const int32_t maxv = sz - 1;
                const int ng       = pt.type_sym[t].num_gens;
                std::vector<int8_t> vis(static_cast<size_t>(sz), 0);
                for (int32_t x = 0; x < sz; ++x) {
                    if (vis[x]) continue;
                    std::vector<std::pair<int32_t, int>> orb;
                    orb.push_back({x, 1});
                    vis[x] = 1;
                    bool force_zero_w = false;
                    for (size_t qi = 0; qi < orb.size(); ++qi) {
                        int32_t cur = orb[qi].first;
                        int csgn    = orb[qi].second;
                        int32_t inv = maxv - cur;
                        if (!vis[inv]) {
                            vis[inv] = static_cast<int8_t>(csgn > 0 ? -1 : 1);
                            orb.push_back({inv, -csgn});
                        } else if (vis[inv] != -csgn) {
                            force_zero_w = true;
                        }
                        for (int g = 0; g < ng; ++g) {
                            int32_t nxt = pt.permute_raw_index(t, g, cur);
                            if (!vis[nxt]) {
                                vis[nxt] = static_cast<int8_t>(csgn);
                                orb.push_back({nxt, csgn});
                            } else if (vis[nxt] != csgn) {
                                force_zero_w = true;
                            }
                        }
                    }
                    int16_t rep_w = force_zero_w ? int16_t(0) : clamp_safe_i16(wp[off + x]);
                    for (const auto& [idx, sgn] : orb) {
                        wp[off + idx] = (sgn > 0) ? rep_w : static_cast<int16_t>(-rep_w);
                    }
                }
            }
        }
        symmetrize_macro_luts();
    }
};

// ============================================================================
// Pure-Scalar Evaluation Kernels (v1.1: Asymmetric Layout + Factored SCReLU)
// ============================================================================

__attribute__((noinline))
inline int32_t eval_raw_ntuple64(const RawNTuple64Weights& raw,
                                 const PatternMeta* __restrict meta,
                                 const uint16_t* __restrict idx,
                                 int stm, int ply) noexcept {
    const int16_t* __restrict wp = raw.w_ply[ply].data();
    const int8_t*  __restrict ct = raw.corner_tempo[ply].data();
    int32_t sum_A = 0;
    int32_t sum_T = 0;
    for (int p = 0; p < 4; ++p) {
        const int32_t x = idx[p];
        sum_A += wp[meta[p].raw_off + x];
        sum_T += ct[x];
    }
    for (int p = 4; p < NUM_PATTERNS; ++p) {
        sum_A += wp[meta[p].raw_off + idx[p]];
    }
    return neg_if(sum_A, stm < 0) + sum_T;
}

template<EvalMode MODE>
__attribute__((noinline))
int32_t evaluate_p2c64(const P2C64Weights& weights,
                       const PatternMeta* __restrict meta,
                       const uint16_t* __restrict idx,
                       int stm, int ply) noexcept {
    const uint8_t blk = PLY_TO_BLK[ply];
    const uint8_t sub = PLY_TO_SUB[ply];
    const BlockEntry16* __restrict ktbl = weights.key_blocks[blk].data();
    const int16_t*      __restrict nk_w = weights.nonkey_w[ply].data();
    const int8_t*       __restrict ct   = weights.corner_tempo[blk].data();

    int32_t sum_A = 0;
    int32_t sum_T = 0;

    if constexpr (MODE == EvalMode::LINEAR_64) {
        // Key 14 nodes (16B BlockEntry16 + Corner-4 tempo reusing cx)
        for (int p = 0; p < 4; ++p) {
            const int32_t x    = idx[p];
            const int32_t d    = meta[p].half - x;
            const int32_t mask = d >> 31;
            const int32_t cx   = x + ((d + d) & mask);
            sum_A += neg_mask(ktbl[meta[p].canon_off + cx].w[sub], mask);
            sum_T += ct[cx];
        }
        for (int p = 4; p < NUM_KEY_PATTERNS; ++p) {
            const int32_t x    = idx[p];
            const int32_t d    = meta[p].half - x;
            const int32_t mask = d >> 31;
            const int32_t cx   = x + ((d + d) & mask);
            sum_A += neg_mask(ktbl[meta[p].canon_off + cx].w[sub], mask);
        }
        // Non-key 50 nodes: direct 2B lookup, ZERO canonicalization, max cache density!
        for (int p = NUM_KEY_PATTERNS; p < NUM_PATTERNS; ++p) {
            sum_A += nk_w[meta[p].nonkey_off + idx[p]];
        }
        return neg_if(sum_A, stm < 0) + sum_T;
    }
    else if constexpr (MODE == EvalMode::CAND_A_SCRELU_K2) {
        int32_t ko0[NUM_KEY_PATTERNS], ko1[NUM_KEY_PATTERNS];
        int32_t ke0[NUM_KEY_PATTERNS], ke1[NUM_KEY_PATTERNS];
        int32_t g_o0 = 0, g_o1 = 0, g_e0 = 0, g_e1 = 0;

        for (int p = 0; p < NUM_KEY_PATTERNS; ++p) {
            const int32_t x    = idx[p];
            const int32_t d    = meta[p].half - x;
            const int32_t mask = d >> 31;
            const int32_t cx   = x + ((d + d) & mask);
            const BlockEntry16& e = ktbl[meta[p].canon_off + cx];
            sum_A += neg_mask(e.w[sub], mask);
            if (p < 4) sum_T += ct[cx];

            const int32_t o0 = neg_mask(static_cast<int8_t>(e.aux[0]), mask);
            const int32_t o1 = neg_mask(static_cast<int8_t>(e.aux[1]), mask);
            const int32_t e0 = static_cast<int8_t>(e.aux[2]);
            const int32_t e1 = static_cast<int8_t>(e.aux[3]);
            ko0[p] = o0; ko1[p] = o1; ke0[p] = e0; ke1[p] = e1;
            g_o0 += o0;  g_o1 += o1;  g_e0 += e0;  g_e1 += e1;
        }
        // Non-key 50 nodes: direct 2B lookup
        for (int p = NUM_KEY_PATTERNS; p < NUM_PATTERNS; ++p) {
            sum_A += nk_w[meta[p].nonkey_off + idx[p]];
        }

        // Factored Dual-SCReLU K=2:
        //   (1) b^2 - w^2 = (b - w) * (b + w)           [1 imul instead of 2]
        //   (2) Distributive group beta factoring        [9 beta imuls instead of 21]
        //   Total: 14 + 14 + 9 = 37 imuls (down from 63!)
        const SCReLUBlockParams& sp = weights.screlu[blk];
        int32_t res_acc = 0;

        // Group 0: 4 Edge strips r=0..3
        {
            const int32_t mu0 = sp.mu[0][0], mu1 = sp.mu[0][1];
            int32_t s_sq0 = 0, s_sq1 = 0, s_pr = 0;
            for (int r = 0; r < 4; ++r) {
                const int c1 = (r + 1) & 3;
                const int32_t o0 = ko0[4 + r] + ko0[8 + r] + ko0[r] + ko0[c1];
                const int32_t e0 = ke0[4 + r] + ke0[8 + r] + ke0[r] + ke0[c1] + mu0;
                const int32_t o1 = ko1[4 + r] + ko1[8 + r] + ko1[r] + ko1[c1];
                const int32_t e1 = ke1[4 + r] + ke1[8 + r] + ke1[r] + ke1[c1] + mu1;
                const int32_t b0 = clamp_i32(e0 + o0, SCRELU_Q);
                const int32_t w0 = clamp_i32(e0 - o0, SCRELU_Q);
                const int32_t b1 = clamp_i32(e1 + o1, SCRELU_Q);
                const int32_t w1 = clamp_i32(e1 - o1, SCRELU_Q);
                s_sq0 += (b0 - w0) * (b0 + w0);
                s_sq1 += (b1 - w1) * (b1 + w1);
                s_pr  += b0 * b1 - w0 * w1;
            }
            res_acc += sp.beta_sq[0][0] * s_sq0
                     + sp.beta_sq[0][1] * s_sq1
                     + sp.beta_pr[0]    * s_pr;
        }
        // Group 1: 2 Diag strips d=0..1
        {
            const int32_t mu0 = sp.mu[1][0], mu1 = sp.mu[1][1];
            int32_t s_sq0 = 0, s_sq1 = 0, s_pr = 0;
            for (int d = 0; d < 2; ++d) {
                const int32_t o0 = ko0[12 + d] + ko0[d] + ko0[d + 2];
                const int32_t e0 = ke0[12 + d] + ke0[d] + ke0[d + 2] + mu0;
                const int32_t o1 = ko1[12 + d] + ko1[d] + ko1[d + 2];
                const int32_t e1 = ke1[12 + d] + ke1[d] + ke1[d + 2] + mu1;
                const int32_t b0 = clamp_i32(e0 + o0, SCRELU_Q);
                const int32_t w0 = clamp_i32(e0 - o0, SCRELU_Q);
                const int32_t b1 = clamp_i32(e1 + o1, SCRELU_Q);
                const int32_t w1 = clamp_i32(e1 - o1, SCRELU_Q);
                s_sq0 += (b0 - w0) * (b0 + w0);
                s_sq1 += (b1 - w1) * (b1 + w1);
                s_pr  += b0 * b1 - w0 * w1;
            }
            res_acc += sp.beta_sq[1][0] * s_sq0
                     + sp.beta_sq[1][1] * s_sq1
                     + sp.beta_pr[1]    * s_pr;
        }
        // Group 2: 1 Global sum
        {
            const int32_t e0 = g_e0 + sp.mu[2][0];
            const int32_t e1 = g_e1 + sp.mu[2][1];
            const int32_t b0 = clamp_i32(e0 + g_o0, SCRELU_Q);
            const int32_t w0 = clamp_i32(e0 - g_o0, SCRELU_Q);
            const int32_t b1 = clamp_i32(e1 + g_o1, SCRELU_Q);
            const int32_t w1 = clamp_i32(e1 - g_o1, SCRELU_Q);
            res_acc += sp.beta_sq[2][0] * ((b0 - w0) * (b0 + w0))
                     + sp.beta_sq[2][1] * ((b1 - w1) * (b1 + w1))
                     + sp.beta_pr[2]    * (b0 * b1 - w0 * w1);
        }

        const int32_t a_res = sym_rshift(res_acc, SCRELU_SHIFT);
        return neg_if(sum_A + a_res, stm < 0) + sum_T;
    }
    else if constexpr (MODE == EvalMode::CAND_B_SCONED_3W3B || MODE == EvalMode::CAND_B_SCONED_3W3B_NAIVE) {
        if constexpr (MODE == EvalMode::CAND_B_SCONED_3W3B_NAIVE) {
            // Naive placement: 50 non-key 2B loads FIRST, then Key-14 + 10 macro LUTs at end
            for (int p = NUM_KEY_PATTERNS; p < NUM_PATTERNS; ++p) {
                sum_A += nk_w[meta[p].nonkey_off + idx[p]];
            }
        }

        // Key 14 nodes (ZERO multiplications, ZERO branches!)
        uint32_t zf[NUM_KEY_PATTERNS];
        uint32_t zr[4];
        for (int p = 0; p < NUM_KEY_PATTERNS; ++p) {
            const int32_t x    = idx[p];
            const int32_t d    = meta[p].half - x;
            const int32_t mask = d >> 31;
            const int32_t cx   = x + ((d + d) & mask);
            const BlockEntry16& e = ktbl[meta[p].canon_off + cx];
            sum_A += neg_mask(e.w[sub], mask);
            if (p < 4) sum_T += ct[cx];
            zf[p] = color_inv_z3_mask(e.aux[0] & 7U, mask);
            if (p >= 4 && p < 8) {
                zr[p - 4] = color_inv_z3_mask(e.aux[1] & 7U, mask);
            }
        }

        // Issue 10 L1-resident 3way-3bit Macro-LUT lookups IMMEDIATELY (0 imul, 0 branches!)
        const MacroLUTBlock& ml = weights.macro[blk];
        int32_t a_res = 0;
        for (int i = 0; i < 4; ++i) {
            a_res += ml.t_cw[(zf[i] << 6) | (zf[4 + i] << 3) | zr[(i + 3) & 3]];
            a_res += ml.t_ea[(zf[4 + i] << 6) | (zr[i] << 3) | zf[8 + i]];
        }
        a_res += ml.t_cd[(zf[0] << 6) | (zf[12] << 3) | zf[2]];
        a_res += ml.t_cd[(zf[1] << 6) | (zf[13] << 3) | zf[3]];

        if constexpr (MODE == EvalMode::CAND_B_SCONED_3W3B) {
            // Key-14 First: overlap remaining 50 direct 2B loads with the 10 macro-LUT loads!
            for (int p = NUM_KEY_PATTERNS; p < NUM_PATTERNS; ++p) {
                sum_A += nk_w[meta[p].nonkey_off + idx[p]];
            }
        }
        return neg_if(sum_A + a_res, stm < 0) + sum_T;
    }
    else { // EvalMode::CAND_C_HYBRID (Factored Dual-SCReLU K=1 [10 imul] + SCONE-D 3way-3bit, Key-14 First)
        int32_t ko0[NUM_KEY_PATTERNS], ke0[NUM_KEY_PATTERNS];
        int32_t g_o0 = 0, g_e0 = 0;
        uint32_t zf[NUM_KEY_PATTERNS];
        uint32_t zr[4];

        for (int p = 0; p < NUM_KEY_PATTERNS; ++p) {
            const int32_t x    = idx[p];
            const int32_t d    = meta[p].half - x;
            const int32_t mask = d >> 31;
            const int32_t cx   = x + ((d + d) & mask);
            const BlockEntry16& e = ktbl[meta[p].canon_off + cx];
            sum_A += neg_mask(e.w[sub], mask);
            if (p < 4) sum_T += ct[cx];

            const int32_t o0 = neg_mask(static_cast<int8_t>(e.aux[0]), mask);
            const int32_t e0 = static_cast<int8_t>(e.aux[1]);
            ko0[p] = o0; ke0[p] = e0;
            g_o0 += o0;  g_e0 += e0;

            const uint32_t packed = e.aux[2];
            zf[p] = color_inv_z3_mask(packed & 7U, mask);
            if (p >= 4 && p < 8) {
                zr[p - 4] = color_inv_z3_mask((packed >> 3) & 7U, mask);
            }
        }

        // Issue 10 Macro-LUT lookups immediately (Key-14 First!)
        const MacroLUTBlock& ml = weights.macro[blk];
        int32_t a_lut = 0;
        for (int i = 0; i < 4; ++i) {
            a_lut += ml.t_cw[(zf[i] << 6) | (zf[4 + i] << 3) | zr[(i + 3) & 3]];
            a_lut += ml.t_ea[(zf[4 + i] << 6) | (zr[i] << 3) | zf[8 + i]];
        }
        a_lut += ml.t_cd[(zf[0] << 6) | (zf[12] << 3) | zf[2]];
        a_lut += ml.t_cd[(zf[1] << 6) | (zf[13] << 3) | zf[3]];

        // Remaining 50 direct 2B pattern loads (overlaps with macro-LUT fetches)
        for (int p = NUM_KEY_PATTERNS; p < NUM_PATTERNS; ++p) {
            sum_A += nk_w[meta[p].nonkey_off + idx[p]];
        }

        // Factored Dual-SCReLU K=1 (7 + 3 = 10 imuls total!)
        const SCReLUBlockParams& sp = weights.screlu[blk];
        int32_t res_acc = 0;
        {
            const int32_t mu0 = sp.mu[0][0];
            int32_t s_sq0 = 0;
            for (int r = 0; r < 4; ++r) {
                const int c1 = (r + 1) & 3;
                const int32_t o0 = ko0[4 + r] + ko0[8 + r] + ko0[r] + ko0[c1];
                const int32_t e0 = ke0[4 + r] + ke0[8 + r] + ke0[r] + ke0[c1] + mu0;
                const int32_t b0 = clamp_i32(e0 + o0, SCRELU_Q);
                const int32_t w0 = clamp_i32(e0 - o0, SCRELU_Q);
                s_sq0 += (b0 - w0) * (b0 + w0);
            }
            res_acc += sp.beta_sq[0][0] * s_sq0;
        }
        {
            const int32_t mu0 = sp.mu[1][0];
            int32_t s_sq0 = 0;
            for (int d = 0; d < 2; ++d) {
                const int32_t o0 = ko0[12 + d] + ko0[d] + ko0[d + 2];
                const int32_t e0 = ke0[12 + d] + ke0[d] + ke0[d + 2] + mu0;
                const int32_t b0 = clamp_i32(e0 + o0, SCRELU_Q);
                const int32_t w0 = clamp_i32(e0 - o0, SCRELU_Q);
                s_sq0 += (b0 - w0) * (b0 + w0);
            }
            res_acc += sp.beta_sq[1][0] * s_sq0;
        }
        {
            const int32_t e0 = g_e0 + sp.mu[2][0];
            const int32_t b0 = clamp_i32(e0 + g_o0, SCRELU_Q);
            const int32_t w0 = clamp_i32(e0 - g_o0, SCRELU_Q);
            res_acc += sp.beta_sq[2][0] * ((b0 - w0) * (b0 + w0));
        }

        const int32_t a_sc = sym_rshift(res_acc, SCRELU_SHIFT);
        return neg_if(sum_A + a_sc + a_lut, stm < 0) + sum_T;
    }
}

} // namespace p2c64
