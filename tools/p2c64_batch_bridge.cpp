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
// 1) High-throughput OpenMP batch extractor for Python/PyTorch training
//    Directly reuses include/p2c64_patterns.hpp (zero pattern-definition drift!)
// ============================================================================
void p2c64_extract_batch(const uint64_t* black,
                         const uint64_t* white,
                         const int8_t*   stm_in,
                         int32_t         n,
                         uint64_t        aug_seed,
                         int32_t         enable_aug,
                         uint16_t*       out_raw_idx,       // [n, 64] optional (may be null)
                         int32_t*        out_key_cx,        // [n, 14] flat canon_off + cx (optional)
                         int8_t*         out_key_flip,      // [n, 14] 0 or 1 (optional)
                         int32_t*        out_nonkey_flat,   // [n, 50] flat nonkey_off + idx (optional)
                         int32_t*        out_raw_flat,      // [n, 64] flat raw_off + idx (optional)
                         int8_t*         out_stm) {         // [n] transformed stm (optional)
    const auto& pt = get_pattern_tables();

    #pragma omp parallel for schedule(static) if(n >= 1024)
    for (int32_t i = 0; i < n; ++i) {
        uint64_t b = black[i];
        uint64_t w = white[i];
        int stm    = static_cast<int>(stm_in[i]);

        if (enable_aug != 0) {
            int sym16 = static_cast<int>(mix64(aug_seed ^ (static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL)) & 15ULL);
            uint64_t tb = 0, tw = 0;
            int tstm = 0;
            transform_board_d4z2(b, w, stm, sym16, tb, tw, tstm);
            b = tb; w = tw; stm = tstm;
        }

        if (out_stm) out_stm[i] = static_cast<int8_t>(stm);

        uint16_t idx[NUM_PATTERNS];
        pt.extract_indices(b, w, idx);

        if (out_raw_idx) {
            std::memcpy(out_raw_idx + static_cast<size_t>(i) * NUM_PATTERNS, idx, sizeof(idx));
        }
        if (out_key_cx || out_key_flip) {
            int32_t* kcx = out_key_cx   ? (out_key_cx   + static_cast<size_t>(i) * NUM_KEY_PATTERNS) : nullptr;
            int8_t*  kfl = out_key_flip ? (out_key_flip + static_cast<size_t>(i) * NUM_KEY_PATTERNS) : nullptr;
            for (int p = 0; p < NUM_KEY_PATTERNS; ++p) {
                const int32_t x    = idx[p];
                const int32_t d    = pt.meta[p].half - x;
                const int32_t mask = d >> 31;
                const int32_t cx   = x + ((d + d) & mask);
                if (kcx) kcx[p] = pt.meta[p].canon_off + cx;
                if (kfl) kfl[p] = static_cast<int8_t>(mask & 1);
            }
        }
        if (out_nonkey_flat) {
            int32_t* nkf = out_nonkey_flat + static_cast<size_t>(i) * (NUM_PATTERNS - NUM_KEY_PATTERNS);
            for (int p = NUM_KEY_PATTERNS; p < NUM_PATTERNS; ++p) {
                nkf[p - NUM_KEY_PATTERNS] = pt.meta[p].nonkey_off + idx[p];
            }
        }
        if (out_raw_flat) {
            int32_t* rf = out_raw_flat + static_cast<size_t>(i) * NUM_PATTERNS;
            for (int p = 0; p < NUM_PATTERNS; ++p) {
                rf[p] = pt.meta[p].raw_off + idx[p];
            }
        }
    }
}

// ============================================================================
// 2) Export exact C++ PatternTables constants & internal symmetry orbit tables
//    so Python training & .p2c baking match C++ symmetrization 100% identically.
// ============================================================================
void p2c64_get_constants(int32_t* out_key_canon_entries,
                         int32_t* out_nonkey_raw_entries,
                         int32_t* out_total_raw_entries,
                         int32_t* out_corner_canon_size,
                         int32_t* out_corner_raw_size,
                         int32_t* out_type_canon_off,   // [16]
                         int32_t* out_type_canon_size,  // [16]
                         int32_t* out_type_raw_off,     // [16]
                         int32_t* out_type_raw_size,    // [16]
                         int32_t* out_type_nonkey_off)  // [16]
{
    const auto& pt = get_pattern_tables();
    if (out_key_canon_entries)  *out_key_canon_entries  = pt.key_canon_entries;
    if (out_nonkey_raw_entries) *out_nonkey_raw_entries = pt.nonkey_raw_entries;
    if (out_total_raw_entries)  *out_total_raw_entries  = pt.total_raw_entries;
    if (out_corner_canon_size)  *out_corner_canon_size  = CORNER_CANON_SIZE;
    if (out_corner_raw_size)    *out_corner_raw_size    = CORNER_RAW_SIZE;
    for (int t = 0; t < NUM_PATTERN_TYPES; ++t) {
        if (out_type_canon_off)  out_type_canon_off[t]  = pt.type_canon_off[t];
        if (out_type_canon_size) out_type_canon_size[t] = pt.type_canon_size[t];
        if (out_type_raw_off)    out_type_raw_off[t]    = pt.type_raw_off[t];
        if (out_type_raw_size)   out_type_raw_size[t]   = pt.type_raw_size[t];
        if (out_type_nonkey_off) out_type_nonkey_off[t] = pt.type_nonkey_off[t];
    }
}

void p2c64_get_orbit_tables(int32_t* key_orb_rep,      // [45929]
                            int8_t*  key_orb_sgn,      // [45929] (-1, 0, +1) [0 = anti-mirror/self-dual]
                            int8_t*  key_anti_mirror,  // [45929] (1 if anti-mirror or cx==half)
                            int32_t* edge_mirror_cx,   // [29525] local cx in [0..29524]
                            int8_t*  edge_mirror_flip, // [29525] 0 or 1
                            int32_t* nk_orb_rep,       // [318168]
                            int8_t*  nk_orb_sgn,       // [318168] (-1, 0, +1)
                            int32_t* raw_orb_rep,      // [410022]
                            int8_t*  raw_orb_sgn,      // [410022] (-1, 0, +1)
                            int32_t* ct_canon_rep,     // [9842] even symmetry
                            int32_t* ct_raw_rep) {     // [19683] even symmetry
    const auto& pt = get_pattern_tables();

    // 1) Key 4 types (0..3) canonical orbits
    for (int t = 0; t < 4; ++t) {
        const int32_t off  = pt.type_canon_off[t];
        const int32_t csz  = pt.type_canon_size[t];
        const int32_t maxv = POW3[pt.type_len[t]] - 1;
        const int32_t half = maxv >> 1;
        const int ng       = pt.type_sym[t].num_gens;
        std::vector<int8_t> vis(static_cast<size_t>(csz), 0);

        for (int32_t cx = 0; cx < csz; ++cx) {
            if (t == 1) {
                int32_t y = pt.permute_raw_index(1, 0, cx);
                int fl    = (y > half) ? 1 : 0;
                int32_t cy = fl ? (maxv - y) : y;
                if (edge_mirror_cx)   edge_mirror_cx[cx]   = cy;
                if (edge_mirror_flip) edge_mirror_flip[cx] = static_cast<int8_t>(fl);
            }
            if (vis[cx]) continue;
            std::vector<std::pair<int32_t, int>> orb;
            orb.push_back({cx, 1});
            vis[cx] = 1;
            bool anti_mirror_or_sd = (cx == half);
            for (size_t qi = 0; qi < orb.size(); ++qi) {
                int32_t cur = orb[qi].first;
                int csgn    = orb[qi].second;
                for (int g = 0; g < ng; ++g) {
                    int32_t y  = pt.permute_raw_index(t, g, cur);
                    int fl     = (y > half) ? 1 : 0;
                    int32_t cy = fl ? (maxv - y) : y;
                    int nsgn   = fl ? -csgn : csgn;
                    if (cy == half) anti_mirror_or_sd = true;
                    if (!vis[cy]) {
                        vis[cy] = static_cast<int8_t>(nsgn);
                        orb.push_back({cy, nsgn});
                    } else if (vis[cy] != nsgn) {
                        anti_mirror_or_sd = true;
                    }
                }
            }
            for (const auto& [idx, sgn] : orb) {
                if (key_orb_rep)     key_orb_rep[off + idx]     = off + cx;
                if (key_orb_sgn)     key_orb_sgn[off + idx]     = anti_mirror_or_sd ? int8_t(0) : static_cast<int8_t>(sgn);
                if (key_anti_mirror) key_anti_mirror[off + idx] = anti_mirror_or_sd ? int8_t(1) : int8_t(0);
                if (t == 0 && ct_canon_rep) ct_canon_rep[idx]   = cx;
            }
        }
    }

    // 2) All 16 raw types (also populates NonKey 4..15 and Corner-4 raw tempo)
    for (int t = 0; t < NUM_PATTERN_TYPES; ++t) {
        const int32_t roff  = pt.type_raw_off[t];
        const int32_t nkoff = pt.type_nonkey_off[t];
        const int32_t sz    = pt.type_raw_size[t];
        const int32_t maxv  = sz - 1;
        const int ng        = pt.type_sym[t].num_gens;
        std::vector<int8_t> vis(static_cast<size_t>(sz), 0);

        for (int32_t x = 0; x < sz; ++x) {
            if (vis[x]) continue;
            std::vector<std::pair<int32_t, int>> orb;
            orb.push_back({x, 1});
            vis[x] = 1;
            bool force_zero = false;
            for (size_t qi = 0; qi < orb.size(); ++qi) {
                int32_t cur = orb[qi].first;
                int csgn    = orb[qi].second;
                int32_t inv = maxv - cur;
                if (!vis[inv]) {
                    vis[inv] = static_cast<int8_t>(csgn > 0 ? -1 : 1);
                    orb.push_back({inv, -csgn});
                } else if (vis[inv] != -csgn) {
                    force_zero = true;
                }
                for (int g = 0; g < ng; ++g) {
                    int32_t nxt = pt.permute_raw_index(t, g, cur);
                    if (!vis[nxt]) {
                        vis[nxt] = static_cast<int8_t>(csgn);
                        orb.push_back({nxt, csgn});
                    } else if (vis[nxt] != csgn) {
                        force_zero = true;
                    }
                }
            }
            for (const auto& [idx, sgn] : orb) {
                if (raw_orb_rep) raw_orb_rep[roff + idx] = roff + x;
                if (raw_orb_sgn) raw_orb_sgn[roff + idx] = force_zero ? int8_t(0) : static_cast<int8_t>(sgn);
                if (t >= 4) {
                    if (nk_orb_rep) nk_orb_rep[nkoff + idx] = nkoff + x;
                    if (nk_orb_sgn) nk_orb_sgn[nkoff + idx] = force_zero ? int8_t(0) : static_cast<int8_t>(sgn);
                }
                if (t == 0 && ct_raw_rep) ct_raw_rep[idx] = x;
            }
        }
    }
}

// ============================================================================
// 3) Evaluate a batch of positions directly from a .p2c file using C++ kernel
//    (Used by bit-exact C++/Python verification test!)
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
        const int32_t base = model.evaluate(pt.meta, idx, static_cast<int>(stm[i]), static_cast<int>(ply[i]));
        out_scores[i] = base;

        if (out_max_sym_err) {
            for (int sym = 0; sym < 16; ++sym) {
                uint64_t tb = 0, tw = 0;
                int tstm = 0;
                transform_board_d4z2(black[i], white[i], static_cast<int>(stm[i]), sym, tb, tw, tstm);
                uint16_t tidx[NUM_PATTERNS];
                pt.extract_indices(tb, tw, tidx);
                const int32_t tv = model.evaluate(pt.meta, tidx, tstm, static_cast<int>(ply[i]));
                const int32_t diff = (tv >= base) ? (tv - base) : (base - tv);
                if (diff > max_sym_err) max_sym_err = diff;
            }
        }
    }
    if (out_max_sym_err) *out_max_sym_err = max_sym_err;
    return static_cast<int32_t>(model.mode);
}

uint64_t p2c64_fnv1a_64(const void* data, size_t len) {
    return fnv1a_64_update(fnv1a_64_init(), data, len);
}

} // extern "C"
