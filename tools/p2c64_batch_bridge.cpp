#include "p2c64_eval.hpp"
#include "p2c64_io.hpp"
#include "p2c64_patterns.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace p2c64;

static inline uint64_t mix64(uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

extern "C" {

// ============================================================================
// 1) High-throughput OpenMP PTM batch extractor for Python/PyTorch v2.0
//    Outputs Player-to-Move (PTM) ternary indices directly:
//      Opponent = 0, Empty = 1, Player = 2
//    (Z2 color inversion is automatic; optional D4 augmentation applied via sym8)
// ============================================================================
void p2c64_extract_batch_ptm(const uint64_t* black,
                             const uint64_t* white,
                             const int8_t*   stm_in,
                             int32_t         n,
                             uint64_t        aug_seed,
                             int32_t         enable_aug,
                             int32_t*        out_key_ptm,     // [n, 14] flat 0..91853 (optional)
                             int32_t*        out_nonkey_ptm,  // [n, 50] flat 0..318167 (optional)
                             int32_t*        out_raw_ptm) {   // [n, 64] flat 0..410021 (optional)
    const auto& pt = get_pattern_tables();

    #pragma omp parallel for schedule(static) if(n >= 1024)
    for (int32_t i = 0; i < n; ++i) {
        // Convert to Player-to-Move (P = side to move, O = opponent)
        uint64_t P = (stm_in[i] > 0) ? black[i] : white[i];
        uint64_t O = (stm_in[i] > 0) ? white[i] : black[i];

        if (enable_aug != 0) {
            int d4 = static_cast<int>(mix64(aug_seed ^ (static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL)) & 7ULL);
            P = transform_bb_d4(P, d4);
            O = transform_bb_d4(O, d4);
        }

        uint16_t idx[NUM_PATTERNS];
        pt.extract_indices(P, O, idx); // P=2 (Player), E=1, O=0 (Opponent) -> exact PTM!

        if (out_key_ptm) {
            int32_t* kp = out_key_ptm + static_cast<size_t>(i) * NUM_KEY_PATTERNS;
            for (int p = 0; p < NUM_KEY_PATTERNS; ++p) {
                kp[p] = pt.meta[p].raw_off + idx[p];
            }
        }
        if (out_nonkey_ptm) {
            int32_t* nkp = out_nonkey_ptm + static_cast<size_t>(i) * (NUM_PATTERNS - NUM_KEY_PATTERNS);
            for (int p = NUM_KEY_PATTERNS; p < NUM_PATTERNS; ++p) {
                nkp[p - NUM_KEY_PATTERNS] = pt.meta[p].nonkey_off + idx[p];
            }
        }
        if (out_raw_ptm) {
            int32_t* rp = out_raw_ptm + static_cast<size_t>(i) * NUM_PATTERNS;
            for (int p = 0; p < NUM_PATTERNS; ++p) {
                rp[p] = pt.meta[p].raw_off + idx[p];
            }
        }
    }
}

// ============================================================================
// 2) Export exact C++ PatternTables constants & D4 internal mirror orbits (v2.0 PTM)
// ============================================================================
void p2c64_get_constants_v2(int32_t* out_key_raw_entries,
                            int32_t* out_nonkey_raw_entries,
                            int32_t* out_total_raw_entries,
                            int32_t* out_type_raw_off,    // [16]
                            int32_t* out_type_raw_size,   // [16]
                            int32_t* out_type_nonkey_off) // [16]
{
    const auto& pt = get_pattern_tables();
    if (out_key_raw_entries)    *out_key_raw_entries    = pt.key_raw_entries;
    if (out_nonkey_raw_entries) *out_nonkey_raw_entries = pt.nonkey_raw_entries;
    if (out_total_raw_entries)  *out_total_raw_entries  = pt.total_raw_entries;
    for (int t = 0; t < NUM_PATTERN_TYPES; ++t) {
        if (out_type_raw_off)    out_type_raw_off[t]    = pt.type_raw_off[t];
        if (out_type_raw_size)   out_type_raw_size[t]   = pt.type_raw_size[t];
        if (out_type_nonkey_off) out_type_nonkey_off[t] = pt.type_nonkey_off[t];
    }
}

void p2c64_get_orbit_tables_v2(int32_t* key_orb_rep,    // [91854]
                               int32_t* edge_mirror_x,  // [59049] local mirror index in [0..59048]
                               int32_t* nk_orb_rep,     // [318168]
                               int32_t* raw_orb_rep) {  // [410022]
    const auto& pt = get_pattern_tables();

    for (int t = 0; t < NUM_PATTERN_TYPES; ++t) {
        const int32_t roff  = pt.type_raw_off[t];
        const int32_t nkoff = pt.type_nonkey_off[t];
        const int32_t sz    = pt.type_raw_size[t];
        const int ng        = pt.type_sym[t].num_gens;

        if (t == 1 && edge_mirror_x) {
            for (int32_t x = 0; x < sz; ++x) {
                edge_mirror_x[x] = pt.permute_raw_index(1, 0, x);
            }
        }

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
            for (int32_t idx : orb) {
                if (raw_orb_rep) raw_orb_rep[roff + idx] = roff + x;
                if (t < 4 && key_orb_rep) key_orb_rep[roff + idx] = roff + x;
                if (t >= 4 && nk_orb_rep) nk_orb_rep[nkoff + idx] = nkoff + x;
            }
        }
    }
}

// ============================================================================
// 3) Evaluate a batch of positions directly from a .p2c v2.0/v3.0 file using C++ kernel
// ============================================================================
int32_t p2c64_eval_file_batch(const char*     p2c_path,
                              const uint64_t* black,
                              const uint64_t* white,
                              const int8_t*   stm,
                              const uint8_t*  ply,
                              int32_t         n,
                              int32_t*        out_scores,
                              int32_t*        out_max_sym_err) {
    LoadedModel model;
    std::string err;
    if (!load_p2c_model(p2c_path, model, &err)) {
        std::fprintf(stderr, "[p2c64_batch_bridge] load_p2c_model failed: %s\n", err.c_str());
        return -1;
    }
    const auto& pt = get_pattern_tables();
    int32_t max_sym_err = 0;

    for (int32_t i = 0; i < n; ++i) {
        uint16_t idx[NUM_PATTERNS];
        pt.extract_indices(black[i], white[i], idx);
        const int32_t base = model.evaluate_board(pt.meta, idx, black[i], white[i],
                                                  static_cast<int>(stm[i]), static_cast<int>(ply[i]));
        out_scores[i] = base;

        if (out_max_sym_err) {
            for (int sym = 0; sym < 16; ++sym) {
                uint64_t tb = 0, tw = 0;
                int tstm = 0;
                transform_board_d4z2(black[i], white[i], static_cast<int>(stm[i]), sym, tb, tw, tstm);
                uint16_t tidx[NUM_PATTERNS];
                pt.extract_indices(tb, tw, tidx);
                const int32_t tv = model.evaluate_board(pt.meta, tidx, tb, tw, tstm, static_cast<int>(ply[i]));
                const int32_t diff = (tv >= base) ? (tv - base) : (base - tv);
                if (diff > max_sym_err) max_sym_err = diff;
            }
        }
    }
    if (out_max_sym_err) *out_max_sym_err = max_sym_err;
    return static_cast<int32_t>(model.mode);
}

// ============================================================================
// 4) High-throughput OpenMP WACA-G2 feature extractor for PyTorch v3.0 training
//    Uses exact C++ Stage 1 / Stage 2.5 / Stage 3 feature formulas -> 0-bit skew!
// ============================================================================
void waca_g2_extract_batch_features(const uint64_t* black,
                                    const uint64_t* white,
                                    const int8_t*   stm_in,
                                    const uint8_t*  ply_in,
                                    int32_t         n,
                                    const int16_t*  tier1_w_flat, // [60 * 410022] PTM int16
                                    uint64_t        aug_seed,
                                    int32_t         enable_aug,
                                    int32_t*        out_v1,       // [n]
                                    int32_t*        out_f25,      // [n, 16]
                                    int32_t*        out_f3_geom,  // [n, 4] (optional)
                                    int32_t*        out_key_ptm)  // [n, 14] (optional)
{
    const auto& pt = get_pattern_tables();
    const size_t stride = static_cast<size_t>(pt.total_raw_entries);

    #pragma omp parallel for schedule(static) if(n >= 1024)
    for (int32_t i = 0; i < n; ++i) {
        uint64_t P = (stm_in[i] > 0) ? black[i] : white[i];
        uint64_t O = (stm_in[i] > 0) ? white[i] : black[i];
        if (enable_aug != 0) {
            const int d4 = static_cast<int>(
                mix64(aug_seed ^ (static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL)) & 7ULL);
            P = transform_bb_d4(P, d4);
            O = transform_bb_d4(O, d4);
        }

        uint16_t idx[NUM_PATTERNS];
        pt.extract_indices(P, O, idx); // P=2 (Player), E=1, O=0 (Opponent) -> exact PTM!

        const int ply = static_cast<int>(ply_in[i]);
        const int16_t* wp = tier1_w_flat + static_cast<size_t>(ply) * stride;

        int32_t c[4], e[4];
        int32_t v1 = 0;
        for (int p = 0; p < 4; ++p) {
            const int32_t wv = wp[pt.meta[p].raw_off + idx[p]];
            c[p] = wv;
            v1 += wv;
        }
        for (int p = 4; p < 8; ++p) {
            const int32_t wv = wp[pt.meta[p].raw_off + idx[p]];
            e[p - 4] = wv;
            v1 += wv;
        }
        for (int p = 8; p < NUM_PATTERNS; ++p) {
            v1 += wp[pt.meta[p].raw_off + idx[p]];
        }

        if (out_v1) out_v1[i] = v1;

        int32_t f25[WACA_F25];
        extract_waca_stage25_features(P, O, v1, c, e, f25);
        if (out_f25) {
            int32_t* dst = out_f25 + static_cast<size_t>(i) * WACA_F25;
            for (int k = 0; k < WACA_F25; ++k) dst[k] = f25[k];
        }
        if (out_f3_geom) {
            int32_t fex[4];
            extract_waca_stage3_geom_features(P, O, c, e, fex);
            int32_t* dst = out_f3_geom + static_cast<size_t>(i) * 4;
            for (int k = 0; k < 4; ++k) dst[k] = fex[k];
        }
        if (out_key_ptm) {
            int32_t* kp = out_key_ptm + static_cast<size_t>(i) * NUM_KEY_PATTERNS;
            for (int p = 0; p < NUM_KEY_PATTERNS; ++p) {
                kp[p] = pt.meta[p].raw_off + idx[p];
            }
        }
    }
}

// ============================================================================
// 5) Evaluate all 3 stages (V1, V25, V3) of a .p2c v3.0 WACA_G2 file + 16-sym check
// ============================================================================
int32_t waca_g2_eval_file_batch_stages(const char*     p2c_path,
                                       const uint64_t* black,
                                       const uint64_t* white,
                                       const int8_t*   stm,
                                       const uint8_t*  ply,
                                       int32_t         n,
                                       int32_t*        out_v1,
                                       int32_t*        out_v25,
                                       int32_t*        out_v3,
                                       int32_t*        out_max_sym_err3) { // [3] (optional)
    LoadedModel model;
    std::string err;
    if (!load_p2c_model(p2c_path, model, &err) || !model.is_waca) {
        std::fprintf(stderr, "[p2c64_batch_bridge] load WACA_G2 failed: %s\n", err.c_str());
        return -1;
    }
    const auto& pt = get_pattern_tables();
    int32_t me[3] = {0, 0, 0};

    for (int32_t i = 0; i < n; ++i) {
        uint16_t idx[NUM_PATTERNS];
        pt.extract_indices(black[i], white[i], idx);
        int32_t v1 = 0, v25 = 0, v3 = 0;
        eval_waca_g2_stages(model.waca, pt.meta, idx, black[i], white[i],
                            static_cast<int>(stm[i]), static_cast<int>(ply[i]),
                            v1, v25, v3);
        if (out_v1)  out_v1[i]  = v1;
        if (out_v25) out_v25[i] = v25;
        if (out_v3)  out_v3[i]  = v3;

        if (out_max_sym_err3) {
            for (int sym = 0; sym < 16; ++sym) {
                uint64_t tb = 0, tw = 0;
                int tstm = 0;
                transform_board_d4z2(black[i], white[i], static_cast<int>(stm[i]), sym, tb, tw, tstm);
                uint16_t tidx[NUM_PATTERNS];
                pt.extract_indices(tb, tw, tidx);
                int32_t tv1 = 0, tv25 = 0, tv3 = 0;
                eval_waca_g2_stages(model.waca, pt.meta, tidx, tb, tw, tstm, static_cast<int>(ply[i]),
                                    tv1, tv25, tv3);
                const int32_t d1 = abs_i32(tv1 - v1);
                const int32_t d2 = abs_i32(tv25 - v25);
                const int32_t d3 = abs_i32(tv3 - v3);
                if (d1 > me[0]) me[0] = d1;
                if (d2 > me[1]) me[1] = d2;
                if (d3 > me[2]) me[2] = d3;
            }
        }
    }
    if (out_max_sym_err3) {
        out_max_sym_err3[0] = me[0];
        out_max_sym_err3[1] = me[1];
        out_max_sym_err3[2] = me[2];
    }
    return model.waca.mlp_hidden;
}

uint64_t p2c64_fnv1a_64(const void* data, size_t len) {
    return fnv1a_64_update(fnv1a_64_init(), data, len);
}

} // extern "C"
