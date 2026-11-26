#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace p2c64 {

static constexpr int NUM_PATTERN_TYPES = 16;
static constexpr int NUM_PATTERNS      = 64;
static constexpr int NUM_KEY_PATTERNS  = 14;
static constexpr int NUM_PLIES         = 60;
static constexpr int PLIES_PER_BLOCK   = 6;
static constexpr int NUM_BLOCKS        = NUM_PLIES / PLIES_PER_BLOCK; // 10
static constexpr int MAX_PATTERN_LEN   = 10;

static constexpr std::array<int32_t, 11> POW3 = {
    1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049
};

static inline constexpr int rc(int r, int c) noexcept { return r * 8 + c; }
static inline constexpr int rot90_sq(int sq) noexcept { return (sq & 7) * 8 + (7 - (sq >> 3)); }
static inline constexpr int flipH_sq(int sq) noexcept { return (sq & 56) + (7 - (sq & 7)); }

// ============================================================================
// D4 x Z2 (16 total board symmetries) on 64-bit bitboards
// sym in [0..7]: D4 geometric transform (bit 2 = flipH, bits 0..1 = rot90 count)
// color_inv in {0, 1}: Z2 color swap (B <-> W, stm -> -stm)
// ============================================================================
static inline int transform_sq_d4(int sq, int d4_sym) noexcept {
    int s = sq;
    if (d4_sym & 4) s = flipH_sq(s);
    int r = d4_sym & 3;
    for (int i = 0; i < r; ++i) s = rot90_sq(s);
    return s;
}

static inline uint64_t transform_bb_d4(uint64_t bb, int d4_sym) noexcept {
    uint64_t out = 0;
    while (bb) {
        int sq = __builtin_ctzll(bb);
        bb &= bb - 1;
        out |= 1ULL << transform_sq_d4(sq, d4_sym);
    }
    return out;
}

static inline void transform_board_d4z2(uint64_t black, uint64_t white, int stm,
                                        int sym16,
                                        uint64_t& out_black, uint64_t& out_white, int& out_stm) noexcept {
    int d4 = sym16 & 7;
    int z2 = (sym16 >> 3) & 1;
    uint64_t tb = transform_bb_d4(black, d4);
    uint64_t tw = transform_bb_d4(white, d4);
    if (z2) {
        out_black = tw;
        out_white = tb;
        out_stm   = -stm;
    } else {
        out_black = tb;
        out_white = tw;
        out_stm   = stm;
    }
}

// ============================================================================
// Pattern metadata & square->pattern incremental update tables
// ============================================================================
struct PatternDef {
    uint8_t type;
    uint8_t len;
    uint8_t sq[MAX_PATTERN_LEN];
};

struct PatternMeta {
    int32_t maxv;       // 3^len - 1
    int32_t half;       // (3^len - 1) / 2
    int32_t canon_off;  // offset in canonical BlockEntry16 array (types 0..3 -> 0..45928)
    int32_t raw_off;    // offset in full 64-pat RawNTuple64 array
    int32_t nonkey_off; // offset in non-key 50-pat 2B direct array (types 4..15 -> 0..318167)
};

struct SqPatternUpdate {
    uint8_t  pat;
    uint16_t pow3;
};

// Internal symmetry group generators for each of the 16 pattern types
// (max 2 involutions: enough for id, Z2 reflection, and V4 Klein-four for Center 2x4)
struct TypeSymSpec {
    uint8_t len;
    uint8_t num_gens;                     // 0 (8-inst asym), 1 (mirror/transpose), 2 (V4)
    uint8_t perm[2][MAX_PATTERN_LEN];     // dst_pos -> src_pos (involution)
};

struct PatternTables {
    PatternDef      pats[NUM_PATTERNS]{};
    PatternMeta     meta[NUM_PATTERNS]{};
    int32_t         type_len[NUM_PATTERN_TYPES]{};
    int32_t         type_canon_size[NUM_PATTERN_TYPES]{};
    int32_t         type_canon_off[NUM_PATTERN_TYPES]{};
    int32_t         type_raw_size[NUM_PATTERN_TYPES]{};
    int32_t         type_raw_off[NUM_PATTERN_TYPES]{};
    int32_t         type_nonkey_off[NUM_PATTERN_TYPES]{};
    int32_t         total_canon_entries  = 0; // All 16 types: 205,019
    int32_t         key_canon_entries    = 0; // Key 4 types (0..3): 45,929
    int32_t         total_raw_entries    = 0; // All 16 types: 410,022
    int32_t         nonkey_raw_entries   = 0; // Non-key 12 types (4..15): 318,168
    TypeSymSpec     type_sym[NUM_PATTERN_TYPES]{};

    SqPatternUpdate sq_upd[64][16]{};
    uint8_t         sq_upd_cnt[64]{};

    // Precomputed canonical mirror/involution mapping for offline table symmetrization
    // Only built on demand via helper functions to keep PatternTables compact.
    PatternTables() { init(); }

    void init() {
        int pi = 0;
        auto add_rot4 = [&](int type, std::initializer_list<int> base_list) {
            std::vector<int> base(base_list);
            type_len[type] = static_cast<int32_t>(base.size());
            for (int r = 0; r < 4; ++r) {
                pats[pi].type = static_cast<uint8_t>(type);
                pats[pi].len  = static_cast<uint8_t>(base.size());
                for (size_t j = 0; j < base.size(); ++j) pats[pi].sq[j] = static_cast<uint8_t>(base[j]);
                ++pi;
                for (int& s : base) s = rot90_sq(s);
            }
        };
        auto add_sym8 = [&](int type, std::initializer_list<int> base_list) {
            std::vector<int> base(base_list);
            type_len[type] = static_cast<int32_t>(base.size());
            for (int m = 0; m < 2; ++m) {
                std::vector<int> cur = base;
                if (m) for (int& s : cur) s = flipH_sq(s);
                for (int r = 0; r < 4; ++r) {
                    pats[pi].type = static_cast<uint8_t>(type);
                    pats[pi].len  = static_cast<uint8_t>(cur.size());
                    for (size_t j = 0; j < cur.size(); ++j) pats[pi].sq[j] = static_cast<uint8_t>(cur[j]);
                    ++pi;
                    for (int& s : cur) s = rot90_sq(s);
                }
            }
        };
        auto add_rot2 = [&](int type, std::initializer_list<int> base_list) {
            std::vector<int> base(base_list);
            type_len[type] = static_cast<int32_t>(base.size());
            for (int r = 0; r < 2; ++r) {
                pats[pi].type = static_cast<uint8_t>(type);
                pats[pi].len  = static_cast<uint8_t>(base.size());
                for (size_t j = 0; j < base.size(); ++j) pats[pi].sq[j] = static_cast<uint8_t>(base[j]);
                ++pi;
                for (int& s : base) s = rot90_sq(s);
            }
        };

        // --------------------------------------------------------------------
        // Key 14 patterns FIRST [0..13] (clockwise order for D4 alignment)
        // --------------------------------------------------------------------
        // Type 0: Corner 3x3 (9 sqs) x4 [NW, NE, SE, SW]
        add_rot4(0, {rc(0,0),rc(0,1),rc(0,2), rc(1,0),rc(1,1),rc(1,2), rc(2,0),rc(2,1),rc(2,2)});
        // Type 1: Edge+2X (10 sqs) x4 [N: NW->NE, E: NE->SE, S: SE->SW, W: SW->NW]
        add_rot4(1, {rc(0,0),rc(0,1),rc(0,2),rc(0,3),rc(0,4),rc(0,5),rc(0,6),rc(0,7), rc(1,1),rc(1,6)});
        // Type 2: Line 2 (8 sqs) x4 [N, E, S, W]
        add_rot4(2, {rc(1,0),rc(1,1),rc(1,2),rc(1,3),rc(1,4),rc(1,5),rc(1,6),rc(1,7)});
        // Type 3: Diag 8 (8 sqs) x2 [0: NW->SE, 1: NE->SW]
        add_rot2(3, {rc(0,0),rc(1,1),rc(2,2),rc(3,3),rc(4,4),rc(5,5),rc(6,6),rc(7,7)});

        // --------------------------------------------------------------------
        // Non-key 50 patterns [14..63] (remaining 12 Egaroucid-compatible types)
        // --------------------------------------------------------------------
        // Type 4: Line 3 (8 sqs) x4
        add_rot4(4, {rc(2,0),rc(2,1),rc(2,2),rc(2,3),rc(2,4),rc(2,5),rc(2,6),rc(2,7)});
        // Type 5: Line 4 (8 sqs) x4
        add_rot4(5, {rc(3,0),rc(3,1),rc(3,2),rc(3,3),rc(3,4),rc(3,5),rc(3,6),rc(3,7)});
        // Type 6: Diag 7 (7 sqs) x4
        add_rot4(6, {rc(0,1),rc(1,2),rc(2,3),rc(3,4),rc(4,5),rc(5,6),rc(6,7)});
        // Type 7: Diag 6 (6 sqs) x4
        add_rot4(7, {rc(0,2),rc(1,3),rc(2,4),rc(3,5),rc(4,6),rc(5,7)});
        // Type 8: Diag 5 (5 sqs) x4
        add_rot4(8, {rc(0,3),rc(1,4),rc(2,5),rc(3,6),rc(4,7)});
        // Type 9: Diag 4 (4 sqs) x4
        add_rot4(9, {rc(0,4),rc(1,5),rc(2,6),rc(3,7)});
        // Type 10: Corner 2x5 (10 sqs) x8 (full D4 orbit, asymmetric)
        add_sym8(10, {rc(0,0),rc(0,1),rc(0,2),rc(0,3),rc(0,4), rc(1,0),rc(1,1),rc(1,2),rc(1,3),rc(1,4)});
        // Type 11: Triangle 10 (10 sqs) x4
        add_rot4(11, {rc(0,0),rc(0,1),rc(0,2),rc(0,3), rc(1,0),rc(1,1),rc(1,2), rc(2,0),rc(2,1), rc(3,0)});
        // Type 12: Edge+2C (10 sqs) x4
        add_rot4(12, {rc(0,0),rc(0,1),rc(0,2),rc(0,3),rc(0,4),rc(0,5),rc(0,6),rc(0,7), rc(1,0),rc(1,7)});
        // Type 13: Corner-L10 (10 sqs) x4 (Corner 3x3 + d4)
        add_rot4(13, {rc(0,0),rc(0,1),rc(0,2), rc(1,0),rc(1,1),rc(1,2), rc(2,0),rc(2,1),rc(2,2), rc(3,3)});
        // Type 14: Edge-Shift10 (10 sqs) x4
        add_rot4(14, {rc(0,1),rc(0,2),rc(0,3),rc(0,4),rc(0,5),rc(0,6), rc(1,1),rc(1,2),rc(1,5),rc(1,6)});
        // Type 15: Center 2x4 (8 sqs) x2 (Horizontal 2x4 & Vertical 4x2; exact 2-instance D4 orbit)
        add_rot2(15, {rc(3,2),rc(3,3),rc(3,4),rc(3,5), rc(4,2),rc(4,3),rc(4,4),rc(4,5)});

        // Offsets and canonical/raw sizes (Asymmetric Layout: Key 0..3 vs Non-key 4..15)
        total_canon_entries = 0;
        key_canon_entries   = 0;
        total_raw_entries   = 0;
        nonkey_raw_entries  = 0;
        for (int t = 0; t < NUM_PATTERN_TYPES; ++t) {
            type_canon_off[t]  = total_canon_entries;
            type_canon_size[t] = (POW3[type_len[t]] + 1) / 2;
            total_canon_entries += type_canon_size[t];
            if (t < 4) key_canon_entries += type_canon_size[t];

            type_raw_off[t]  = total_raw_entries;
            type_raw_size[t] = POW3[type_len[t]];
            total_raw_entries += type_raw_size[t];

            if (t >= 4) {
                type_nonkey_off[t] = nonkey_raw_entries;
                nonkey_raw_entries += type_raw_size[t];
            } else {
                type_nonkey_off[t] = 0;
            }
        }

        for (int p = 0; p < NUM_PATTERNS; ++p) {
            int t = pats[p].type;
            int len = pats[p].len;
            meta[p].maxv       = POW3[len] - 1;
            meta[p].half       = (POW3[len] - 1) >> 1;
            meta[p].canon_off  = type_canon_off[t];
            meta[p].raw_off    = type_raw_off[t];
            meta[p].nonkey_off = type_nonkey_off[t];
        }

        // Square -> Pattern incremental update table
        std::memset(sq_upd_cnt, 0, sizeof(sq_upd_cnt));
        for (int p = 0; p < NUM_PATTERNS; ++p) {
            for (int j = 0; j < pats[p].len; ++j) {
                int s = pats[p].sq[j];
                sq_upd[s][sq_upd_cnt[s]++] = {static_cast<uint8_t>(p), static_cast<uint16_t>(POW3[j])};
            }
        }

        // Internal symmetry permutations per pattern type
        auto set_sym1 = [&](int t, std::initializer_list<uint8_t> perm0) {
            type_sym[t].len = static_cast<uint8_t>(perm0.size());
            type_sym[t].num_gens = 1;
            int j = 0;
            for (uint8_t v : perm0) type_sym[t].perm[0][j++] = v;
        };
        auto set_sym2 = [&](int t, std::initializer_list<uint8_t> p0, std::initializer_list<uint8_t> p1) {
            type_sym[t].len = static_cast<uint8_t>(p0.size());
            type_sym[t].num_gens = 2;
            int j = 0; for (uint8_t v : p0) type_sym[t].perm[0][j++] = v;
            j = 0;     for (uint8_t v : p1) type_sym[t].perm[1][j++] = v;
        };
        // Type 0: Corner 3x3 diagonal transpose
        set_sym1(0,  {0, 3, 6, 1, 4, 7, 2, 5, 8});
        // Type 1: Edge+2X horizontal mirror
        set_sym1(1,  {7, 6, 5, 4, 3, 2, 1, 0, 9, 8});
        // Type 2, 3, 4, 5: 8-square line/diag reversal
        set_sym1(2,  {7, 6, 5, 4, 3, 2, 1, 0});
        set_sym1(3,  {7, 6, 5, 4, 3, 2, 1, 0});
        set_sym1(4,  {7, 6, 5, 4, 3, 2, 1, 0});
        set_sym1(5,  {7, 6, 5, 4, 3, 2, 1, 0});
        // Type 6..9: Diag 7, 6, 5, 4 reversal
        set_sym1(6,  {6, 5, 4, 3, 2, 1, 0});
        set_sym1(7,  {5, 4, 3, 2, 1, 0});
        set_sym1(8,  {4, 3, 2, 1, 0});
        set_sym1(9,  {3, 2, 1, 0});
        // Type 10: Corner 2x5 (8 instances, full D4 orbit -> 0 internal symmetry constraints)
        type_sym[10].len = 10; type_sym[10].num_gens = 0;
        // Type 11: Triangle 10 diagonal transpose
        set_sym1(11, {0, 4, 7, 9, 1, 5, 8, 2, 6, 3});
        // Type 12: Edge+2C horizontal mirror
        set_sym1(12, {7, 6, 5, 4, 3, 2, 1, 0, 9, 8});
        // Type 13: Corner-L10 diagonal transpose
        set_sym1(13, {0, 3, 6, 1, 4, 7, 2, 5, 8, 9});
        // Type 14: Edge-Shift10 horizontal mirror
        set_sym1(14, {5, 4, 3, 2, 1, 0, 9, 8, 7, 6});
        // Type 15: Center 2x4 (flipH and flipV -> Klein four-group V4)
        set_sym2(15, {3, 2, 1, 0, 7, 6, 5, 4}, {4, 5, 6, 7, 0, 1, 2, 3});
    }

    // Apply internal permutation generator g_idx to raw base-3 index x
    int32_t permute_raw_index(int type, int g_idx, int32_t x) const noexcept {
        const TypeSymSpec& sp = type_sym[type];
        int digits[MAX_PATTERN_LEN];
        int32_t tmp = x;
        for (int j = 0; j < sp.len; ++j) {
            digits[j] = tmp % 3;
            tmp /= 3;
        }
        int32_t out = 0;
        for (int j = 0; j < sp.len; ++j) {
            out += digits[sp.perm[g_idx][j]] * POW3[j];
        }
        return out;
    }

    // Extract all 64 ternary indices from (black, white) bitboards
    // Encoding: White = 0, Empty = 1, Black = 2
    void extract_indices(uint64_t black, uint64_t white, uint16_t idx[NUM_PATTERNS]) const noexcept {
        for (int p = 0; p < NUM_PATTERNS; ++p) {
            int32_t val = 0;
            const PatternDef& pd = pats[p];
            for (int j = 0; j < pd.len; ++j) {
                int s = pd.sq[j];
                int d = ((black >> s) & 1ULL) ? 2 : (((white >> s) & 1ULL) ? 0 : 1);
                val += d * POW3[j];
            }
            idx[p] = static_cast<uint16_t>(val);
        }
    }

    // Incremental update of all 64 ternary indices from a move at `sq` flipping `flips`
    // stm > 0 (Black): Empty(1)->Black(2) [+1], White(0)->Black(2) [+2]
    // stm < 0 (White): Empty(1)->White(0) [-1], Black(2)->White(0) [-2]
    void update_indices_move(uint16_t dst[NUM_PATTERNS],
                             const uint16_t src[NUM_PATTERNS],
                             int sq, uint64_t flips, int stm) const noexcept {
        std::memcpy(dst, src, sizeof(uint16_t) * NUM_PATTERNS);
        const int32_t d_put = (stm > 0) ? 1 : -1;
        const int32_t d_flp = (stm > 0) ? 2 : -2;
        const int cnt = sq_upd_cnt[sq];
        for (int i = 0; i < cnt; ++i) {
            dst[sq_upd[sq][i].pat] = static_cast<uint16_t>(
                static_cast<int32_t>(dst[sq_upd[sq][i].pat]) + d_put * sq_upd[sq][i].pow3);
        }
        uint64_t m = flips;
        while (m) {
            int s = __builtin_ctzll(m);
            m &= m - 1;
            const int fcnt = sq_upd_cnt[s];
            for (int i = 0; i < fcnt; ++i) {
                dst[sq_upd[s][i].pat] = static_cast<uint16_t>(
                    static_cast<int32_t>(dst[sq_upd[s][i].pat]) + d_flp * sq_upd[s][i].pow3);
            }
        }
    }
};

inline const PatternTables& get_pattern_tables() {
    static const PatternTables tables;
    return tables;
}

// ============================================================================
// Minimal bitboard Othello move generation / flip utilities (for tests & bench)
// ============================================================================
static inline uint64_t generate_moves(uint64_t P, uint64_t O) noexcept {
    const uint64_t E  = ~(P | O);
    const uint64_t mO = O & 0x7e7e7e7e7e7e7e7eULL;
    uint64_t f, res = 0;

    f = mO & (P << 1); f |= mO & (f << 1); f |= mO & (f << 1);
    f |= mO & (f << 1); f |= mO & (f << 1); f |= mO & (f << 1);
    res |= E & (f << 1);

    f = mO & (P >> 1); f |= mO & (f >> 1); f |= mO & (f >> 1);
    f |= mO & (f >> 1); f |= mO & (f >> 1); f |= mO & (f >> 1);
    res |= E & (f >> 1);

    f = O & (P << 8); f |= O & (f << 8); f |= O & (f << 8);
    f |= O & (f << 8); f |= O & (f << 8); f |= O & (f << 8);
    res |= E & (f << 8);

    f = O & (P >> 8); f |= O & (f >> 8); f |= O & (f >> 8);
    f |= O & (f >> 8); f |= O & (f >> 8); f |= O & (f >> 8);
    res |= E & (f >> 8);

    f = mO & (P << 9); f |= mO & (f << 9); f |= mO & (f << 9);
    f |= mO & (f << 9); f |= mO & (f << 9); f |= mO & (f << 9);
    res |= E & (f << 9);

    f = mO & (P >> 9); f |= mO & (f >> 9); f |= mO & (f >> 9);
    f |= mO & (f >> 9); f |= mO & (f >> 9); f |= mO & (f >> 9);
    res |= E & (f >> 9);

    f = mO & (P << 7); f |= mO & (f << 7); f |= mO & (f << 7);
    f |= mO & (f << 7); f |= mO & (f << 7); f |= mO & (f << 7);
    res |= E & (f << 7);

    f = mO & (P >> 7); f |= mO & (f >> 7); f |= mO & (f >> 7);
    f |= mO & (f >> 7); f |= mO & (f >> 7); f |= mO & (f >> 7);
    res |= E & (f >> 7);

    return res;
}

static inline uint64_t calc_flips(int sq, uint64_t P, uint64_t O) noexcept {
    const uint64_t pos = 1ULL << sq;
    const uint64_t mO  = O & 0x7e7e7e7e7e7e7e7eULL;
    uint64_t flips = 0, f;

    f = mO & (pos << 1); f |= mO & (f << 1); f |= mO & (f << 1); f |= mO & (f << 1); f |= mO & (f << 1); f |= mO & (f << 1);
    if ((f << 1) & P & 0xfefefefefefefefeULL) flips |= f;
    f = mO & (pos >> 1); f |= mO & (f >> 1); f |= mO & (f >> 1); f |= mO & (f >> 1); f |= mO & (f >> 1); f |= mO & (f >> 1);
    if ((f >> 1) & P & 0x7f7f7f7f7f7f7f7fULL) flips |= f;
    f = O & (pos << 8); f |= O & (f << 8); f |= O & (f << 8); f |= O & (f << 8); f |= O & (f << 8); f |= O & (f << 8);
    if ((f << 8) & P) flips |= f;
    f = O & (pos >> 8); f |= O & (f >> 8); f |= O & (f >> 8); f |= O & (f >> 8); f |= O & (f >> 8); f |= O & (f >> 8);
    if ((f >> 8) & P) flips |= f;
    f = mO & (pos << 9); f |= mO & (f << 9); f |= mO & (f << 9); f |= mO & (f << 9); f |= mO & (f << 9); f |= mO & (f << 9);
    if ((f << 9) & P & 0xfefefefefefefefeULL) flips |= f;
    f = mO & (pos >> 9); f |= mO & (f >> 9); f |= mO & (f >> 9); f |= mO & (f >> 9); f |= mO & (f >> 9); f |= mO & (f >> 9);
    if ((f >> 9) & P & 0x7f7f7f7f7f7f7f7fULL) flips |= f;
    f = mO & (pos << 7); f |= mO & (f << 7); f |= mO & (f << 7); f |= mO & (f << 7); f |= mO & (f << 7); f |= mO & (f << 7);
    if ((f << 7) & P & 0x7f7f7f7f7f7f7f7fULL) flips |= f;
    f = mO & (pos >> 7); f |= mO & (f >> 7); f |= mO & (f >> 7); f |= mO & (f >> 7); f |= mO & (f >> 7); f |= mO & (f >> 7);
    if ((f >> 7) & P & 0xfefefefefefefefeULL) flips |= f;
    return flips;
}

} // namespace p2c64
