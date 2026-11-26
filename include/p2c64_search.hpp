#pragma once

#include "p2c64_io.hpp"
#include "p2c64_patterns.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace p2c64 {

// ============================================================================
// Pure-Scalar PVS (Negascout) + Transposition Table + Incremental Patterns
// + Exact Endgame Solver for P2C-64 Models
// ============================================================================
struct ZobristKeys {
    uint64_t sq_b[64]{};
    uint64_t sq_w[64]{};
    uint64_t stm_key = 0;

    ZobristKeys() {
        auto sm64 = [](uint64_t x) -> uint64_t {
            x += 0x9e3779b97f4a7c15ULL;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            return x ^ (x >> 31);
        };
        uint64_t s = 0x20261125cafebabeULL;
        for (int i = 0; i < 64; ++i) {
            s += 0x9e3779b97f4a7c15ULL; sq_b[i] = sm64(s);
            s += 0x9e3779b97f4a7c15ULL; sq_w[i] = sm64(s);
        }
        s += 0x9e3779b97f4a7c15ULL;
        stm_key = sm64(s);
    }

    inline uint64_t hash_board(uint64_t black, uint64_t white, int stm) const noexcept {
        uint64_t h = (stm < 0) ? stm_key : 0ULL;
        uint64_t b = black;
        while (b) { int s = __builtin_ctzll(b); b &= b - 1; h ^= sq_b[s]; }
        uint64_t w = white;
        while (w) { int s = __builtin_ctzll(w); w &= w - 1; h ^= sq_w[s]; }
        return h;
    }
};

inline const ZobristKeys& get_zobrist() {
    static const ZobristKeys zk;
    return zk;
}

static constexpr uint8_t TT_NONE  = 0;
static constexpr uint8_t TT_EXACT = 1;
static constexpr uint8_t TT_LOWER = 2;
static constexpr uint8_t TT_UPPER = 3;

struct alignas(16) SearchTTEntry {
    uint64_t key;
    int32_t  score;
    int8_t   depth;
    uint8_t  flag;
    uint8_t  best_move;
    uint8_t  gen;
};
static_assert(sizeof(SearchTTEntry) == 16, "SearchTTEntry must be 16 bytes");

static constexpr int32_t SEARCH_INF = 100000000;

static constexpr int8_t STATIC_MOVE_ORDER[64] = {
    120, -25,  20,  10,  10,  20, -25, 120,
    -25, -45,  -4,  -2,  -2,  -4, -45, -25,
     20,  -4,   8,   2,   2,   8,  -4,  20,
     10,  -2,   2,   0,   0,   2,  -2,  10,
     10,  -2,   2,   0,   0,   2,  -2,  10,
     20,  -4,   8,   2,   2,   8,  -4,  20,
    -25, -45,  -4,  -2,  -2,  -4, -45, -25,
    120, -25,  20,  10,  10,  20, -25, 120
};

enum class SearchLimitMode : int {
    DEPTH = 0,
    NODES = 1,
    TIME  = 2
};

struct SearchResult {
    int      best_move       = -1;
    int32_t  score           = 0;
    int      completed_depth = 0;
    uint64_t nodes           = 0;
    double   elapsed_ms      = 0.0;
    bool     exact_solved    = false;
};

class PVSEngine {
public:
    const LoadedModel* model_ = nullptr;
    const PatternTables& pt_;
    const ZobristKeys& zk_;

    std::vector<SearchTTEntry> tt_;
    size_t   tt_mask_         = 0;
    uint8_t  tt_gen_          = 1;
    int      endgame_empties_ = 10;

    // Runtime limits & counters
    uint64_t nodes_           = 0;
    uint64_t node_limit_      = 0;
    bool     use_time_limit_  = false;
    bool     aborted_         = false;
    std::chrono::steady_clock::time_point deadline_;

    explicit PVSEngine(const LoadedModel* model, size_t tt_mb = 64, int endgame_empties = 10)
        : model_(model), pt_(get_pattern_tables()), zk_(get_zobrist()), endgame_empties_(endgame_empties) {
        allocate_tt(tt_mb);
    }

    void allocate_tt(size_t tt_mb) {
        size_t bytes = std::max<size_t>(1, tt_mb) * 1024ULL * 1024ULL;
        size_t entries = 1;
        while ((entries << 1) * sizeof(SearchTTEntry) <= bytes) entries <<= 1;
        tt_.assign(entries, SearchTTEntry{0, 0, -1, TT_NONE, 64, 0});
        tt_mask_ = entries - 1;
    }

    void clear_tt() {
        std::memset(tt_.data(), 0, tt_.size() * sizeof(SearchTTEntry));
        tt_gen_ = 1;
    }

    inline int32_t scale_exact(int exact_disc) const noexcept {
        return static_cast<int32_t>(exact_disc * static_cast<int32_t>(model_->score_scale + 0.5f));
    }

    inline bool check_abort() noexcept {
        if (aborted_) return true;
        if ((nodes_ & 1023ULL) == 0ULL) {
            if (node_limit_ > 0 && nodes_ >= node_limit_) {
                aborted_ = true;
                return true;
            }
            if (use_time_limit_ && std::chrono::steady_clock::now() >= deadline_) {
                aborted_ = true;
                return true;
            }
        }
        return false;
    }

    // Exact endgame PVS solver (returns exact disc difference * score_scale)
    int32_t pvs_endgame(uint64_t B, uint64_t W, int stm, int empties,
                        uint64_t hash, int32_t alpha, int32_t beta, bool prev_pass) noexcept {
        nodes_++;
        if (check_abort()) return 0;

        uint64_t P = (stm > 0) ? B : W;
        uint64_t O = (stm > 0) ? W : B;

        if (empties == 0) {
            int p_cnt = __builtin_popcountll(P);
            int o_cnt = __builtin_popcountll(O);
            return scale_exact(p_cnt - o_cnt);
        }

        const int8_t tt_depth = static_cast<int8_t>(64 + empties);
        uint8_t tt_move = 64;
        int32_t orig_alpha = alpha;

        if (empties >= 4) {
            const SearchTTEntry& e = tt_[hash & tt_mask_];
            if (e.flag != TT_NONE && e.key == hash) {
                tt_move = e.best_move;
                if (e.depth >= tt_depth) {
                    if (e.flag == TT_EXACT) return e.score;
                    if (e.flag == TT_LOWER && e.score >= beta) return e.score;
                    if (e.flag == TT_UPPER && e.score <= alpha) return e.score;
                }
            }
        }

        uint64_t mv = generate_moves(P, O);
        if (!mv) {
            if (prev_pass) {
                int p_cnt = __builtin_popcountll(P);
                int o_cnt = __builtin_popcountll(O);
                int diff = p_cnt - o_cnt;
                if (diff > 0) diff += empties;
                else if (diff < 0) diff -= empties;
                return scale_exact(diff);
            }
            return -pvs_endgame(B, W, -stm, empties, hash ^ zk_.stm_key, -beta, -alpha, true);
        }

        // Order moves (TT move first, then static corner/edge priority)
        int moves[32];
        int scores[32];
        int n_moves = 0;
        uint64_t m = mv;
        while (m) {
            int sq = __builtin_ctzll(m);
            m &= m - 1;
            moves[n_moves]  = sq;
            scores[n_moves] = (sq == tt_move) ? 1000 : static_cast<int>(STATIC_MOVE_ORDER[sq]);
            n_moves++;
        }

        int32_t best_val = -SEARCH_INF;
        uint8_t best_sq  = static_cast<uint8_t>(moves[0]);

        for (int i = 0; i < n_moves; ++i) {
            // Selection sort next best move
            int best_idx = i;
            for (int j = i + 1; j < n_moves; ++j) {
                if (scores[j] > scores[best_idx]) best_idx = j;
            }
            if (best_idx != i) {
                std::swap(moves[i], moves[best_idx]);
                std::swap(scores[i], scores[best_idx]);
            }

            const int sq = moves[i];
            const uint64_t fl = calc_flips(sq, P, O);
            uint64_t nB = B, nW = W;
            uint64_t nh = hash ^ zk_.stm_key;
            if (stm > 0) {
                nB |= (1ULL << sq) | fl; nW ^= fl;
                nh ^= zk_.sq_b[sq];
                uint64_t fm = fl;
                while (fm) { int s = __builtin_ctzll(fm); fm &= fm - 1; nh ^= zk_.sq_b[s] ^ zk_.sq_w[s]; }
            } else {
                nW |= (1ULL << sq) | fl; nB ^= fl;
                nh ^= zk_.sq_w[sq];
                uint64_t fm = fl;
                while (fm) { int s = __builtin_ctzll(fm); fm &= fm - 1; nh ^= zk_.sq_b[s] ^ zk_.sq_w[s]; }
            }

            int32_t val;
            if (i == 0) {
                val = -pvs_endgame(nB, nW, -stm, empties - 1, nh, -beta, -alpha, false);
            } else {
                val = -pvs_endgame(nB, nW, -stm, empties - 1, nh, -alpha - 1, -alpha, false);
                if (!aborted_ && val > alpha && val < beta) {
                    val = -pvs_endgame(nB, nW, -stm, empties - 1, nh, -beta, -alpha, false);
                }
            }
            if (aborted_) return 0;

            if (val > best_val) {
                best_val = val;
                best_sq  = static_cast<uint8_t>(sq);
            }
            if (val > alpha) {
                alpha = val;
                if (alpha >= beta) break;
            }
        }

        if (empties >= 4 && !aborted_) {
            SearchTTEntry& e = tt_[hash & tt_mask_];
            if (e.flag == TT_NONE || e.gen != tt_gen_ || e.depth <= tt_depth) {
                e.key       = hash;
                e.score     = best_val;
                e.depth     = tt_depth;
                e.flag      = (best_val <= orig_alpha) ? TT_UPPER : ((best_val >= beta) ? TT_LOWER : TT_EXACT);
                e.best_move = best_sq;
                e.gen       = tt_gen_;
            }
        }
        return best_val;
    }

    // Midgame PVS with Incremental 64-Pattern Index Updates
    int32_t pvs_midgame(uint64_t B, uint64_t W, int stm, int ply, int depth,
                        uint64_t hash, const uint16_t idx[NUM_PATTERNS],
                        int32_t alpha, int32_t beta, bool prev_pass, int* out_best_move = nullptr) noexcept {
        nodes_++;
        if (check_abort()) return 0;

        const int empties = 60 - ply;
        if (empties <= 0) {
            uint64_t P = (stm > 0) ? B : W;
            uint64_t O = (stm > 0) ? W : B;
            return scale_exact(__builtin_popcountll(P) - __builtin_popcountll(O));
        }
        if (empties <= endgame_empties_ && !out_best_move) {
            return pvs_endgame(B, W, stm, empties, hash, alpha, beta, prev_pass);
        }
        if (depth <= 0) {
            return model_->evaluate(pt_.meta, idx, stm, ply);
        }

        uint8_t tt_move = 64;
        const int32_t orig_alpha = alpha;
        {
            const SearchTTEntry& e = tt_[hash & tt_mask_];
            if (e.flag != TT_NONE && e.key == hash) {
                tt_move = e.best_move;
                if (e.depth >= depth && !out_best_move) {
                    if (e.flag == TT_EXACT) return e.score;
                    if (e.flag == TT_LOWER && e.score >= beta) return e.score;
                    if (e.flag == TT_UPPER && e.score <= alpha) return e.score;
                }
            }
        }

        uint64_t P = (stm > 0) ? B : W;
        uint64_t O = (stm > 0) ? W : B;
        uint64_t mv = generate_moves(P, O);
        if (!mv) {
            if (prev_pass) {
                int p_cnt = __builtin_popcountll(P);
                int o_cnt = __builtin_popcountll(O);
                int diff = p_cnt - o_cnt;
                if (diff > 0) diff += empties;
                else if (diff < 0) diff -= empties;
                return scale_exact(diff);
            }
            return -pvs_midgame(B, W, -stm, ply, depth, hash ^ zk_.stm_key, idx, -beta, -alpha, true, nullptr);
        }

        int moves[32];
        int scores[32];
        int n_moves = 0;
        uint64_t m = mv;
        while (m) {
            int sq = __builtin_ctzll(m);
            m &= m - 1;
            moves[n_moves]  = sq;
            scores[n_moves] = (sq == tt_move) ? 10000 : static_cast<int>(STATIC_MOVE_ORDER[sq]);
            n_moves++;
        }

        int32_t best_val = -SEARCH_INF;
        uint8_t best_sq  = static_cast<uint8_t>(moves[0]);
        uint16_t child_idx[NUM_PATTERNS];

        for (int i = 0; i < n_moves; ++i) {
            int best_idx = i;
            for (int j = i + 1; j < n_moves; ++j) {
                if (scores[j] > scores[best_idx]) best_idx = j;
            }
            if (best_idx != i) {
                std::swap(moves[i], moves[best_idx]);
                std::swap(scores[i], scores[best_idx]);
            }

            const int sq = moves[i];
            const uint64_t fl = calc_flips(sq, P, O);
            uint64_t nB = B, nW = W;
            uint64_t nh = hash ^ zk_.stm_key;
            if (stm > 0) {
                nB |= (1ULL << sq) | fl; nW ^= fl;
                nh ^= zk_.sq_b[sq];
                uint64_t fm = fl;
                while (fm) { int s = __builtin_ctzll(fm); fm &= fm - 1; nh ^= zk_.sq_b[s] ^ zk_.sq_w[s]; }
            } else {
                nW |= (1ULL << sq) | fl; nB ^= fl;
                nh ^= zk_.sq_w[sq];
                uint64_t fm = fl;
                while (fm) { int s = __builtin_ctzll(fm); fm &= fm - 1; nh ^= zk_.sq_b[s] ^ zk_.sq_w[s]; }
            }

            int32_t val;
            if (empties - 1 <= endgame_empties_) {
                if (i == 0) {
                    val = -pvs_endgame(nB, nW, -stm, empties - 1, nh, -beta, -alpha, false);
                } else {
                    val = -pvs_endgame(nB, nW, -stm, empties - 1, nh, -alpha - 1, -alpha, false);
                    if (!aborted_ && val > alpha && val < beta) {
                        val = -pvs_endgame(nB, nW, -stm, empties - 1, nh, -beta, -alpha, false);
                    }
                }
            } else {
                pt_.update_indices_move(child_idx, idx, sq, fl, stm);
                if (i == 0) {
                    val = -pvs_midgame(nB, nW, -stm, ply + 1, depth - 1, nh, child_idx, -beta, -alpha, false, nullptr);
                } else {
                    val = -pvs_midgame(nB, nW, -stm, ply + 1, depth - 1, nh, child_idx, -alpha - 1, -alpha, false, nullptr);
                    if (!aborted_ && val > alpha && val < beta) {
                        val = -pvs_midgame(nB, nW, -stm, ply + 1, depth - 1, nh, child_idx, -beta, -alpha, false, nullptr);
                    }
                }
            }
            if (aborted_) return 0;

            if (val > best_val) {
                best_val = val;
                best_sq  = static_cast<uint8_t>(sq);
            }
            if (val > alpha) {
                alpha = val;
                if (alpha >= beta) break;
            }
        }

        if (!aborted_) {
            SearchTTEntry& e = tt_[hash & tt_mask_];
            if (e.flag == TT_NONE || e.gen != tt_gen_ || e.depth <= depth) {
                e.key       = hash;
                e.score     = best_val;
                e.depth     = static_cast<int8_t>(depth);
                e.flag      = (best_val <= orig_alpha) ? TT_UPPER : ((best_val >= beta) ? TT_LOWER : TT_EXACT);
                e.best_move = best_sq;
                e.gen       = tt_gen_;
            }
            if (out_best_move) *out_best_move = static_cast<int>(best_sq);
        }
        return best_val;
    }

    // Root search entrypoint supporting depth / nodes / time modes
    SearchResult search(uint64_t B, uint64_t W, int stm,
                        SearchLimitMode mode, int64_t limit_val) {
        SearchResult res{};
        const auto t0 = std::chrono::steady_clock::now();

        uint64_t P = (stm > 0) ? B : W;
        uint64_t O = (stm > 0) ? W : B;
        uint64_t mv = generate_moves(P, O);
        if (!mv) return res;

        int first_legal = __builtin_ctzll(mv);
        res.best_move = first_legal;

        // If only 1 legal move and not measuring exact score in depth mode, still evaluate if needed
        const int discs   = __builtin_popcountll(B | W);
        const int ply     = discs - 4;
        const int empties = 64 - discs;
        const uint64_t hash = zk_.hash_board(B, W, stm);

        uint16_t root_idx[NUM_PATTERNS];
        pt_.extract_indices(B, W, root_idx);

        tt_gen_ = static_cast<uint8_t>((tt_gen_ + 1) & 0xFFU);
        if (tt_gen_ == 0) tt_gen_ = 1;

        nodes_          = 0;
        aborted_        = false;
        node_limit_     = (mode == SearchLimitMode::NODES) ? static_cast<uint64_t>(limit_val) : 0ULL;
        use_time_limit_ = (mode == SearchLimitMode::TIME);
        if (use_time_limit_) {
            deadline_ = t0 + std::chrono::milliseconds(limit_val);
        }

        // Exact endgame directly when empties <= endgame_empties_
        if (empties <= endgame_empties_) {
            // Root exact solve (allow completion unless strict time/node abort after move 0)
            int moves[32], n_moves = 0;
            uint64_t m = mv;
            while (m) { moves[n_moves++] = __builtin_ctzll(m); m &= m - 1; }
            int32_t alpha = -SEARCH_INF, beta = SEARCH_INF;
            int best_sq = moves[0];
            int32_t best_val = -SEARCH_INF;
            for (int i = 0; i < n_moves; ++i) {
                int sq = moves[i];
                uint64_t fl = calc_flips(sq, P, O);
                uint64_t nB = (stm > 0) ? (B | (1ULL << sq) | fl) : (B ^ fl);
                uint64_t nW = (stm > 0) ? (W ^ fl) : (W | (1ULL << sq) | fl);
                uint64_t nh = zk_.hash_board(nB, nW, -stm);
                int32_t val = -pvs_endgame(nB, nW, -stm, empties - 1, nh, -beta, -alpha, false);
                if (aborted_ && i > 0) break;
                if (val > best_val) { best_val = val; best_sq = sq; }
                if (val > alpha) alpha = val;
            }
            res.best_move       = best_sq;
            res.score           = best_val;
            res.completed_depth = empties;
            res.exact_solved    = !aborted_;
            res.nodes           = nodes_;
            res.elapsed_ms      = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            return res;
        }

        const int max_depth = (mode == SearchLimitMode::DEPTH)
                            ? static_cast<int>(std::min<int64_t>(limit_val, empties))
                            : std::min(30, empties);

        // Iterative deepening (guaranteed at least depth 1 without abort)
        for (int d = 1; d <= max_depth; ++d) {
            const uint64_t saved_nlim = node_limit_;
            const bool saved_tlim     = use_time_limit_;
            if (d == 1) { node_limit_ = 0; use_time_limit_ = false; }

            int cur_best = res.best_move;
            int32_t val = pvs_midgame(B, W, stm, ply, d, hash, root_idx, -SEARCH_INF, SEARCH_INF, false, &cur_best);

            if (d == 1) { node_limit_ = saved_nlim; use_time_limit_ = saved_tlim; }

            if (!aborted_) {
                res.best_move       = cur_best;
                res.score           = val;
                res.completed_depth = d;
            } else {
                break;
            }
            if (mode == SearchLimitMode::NODES && nodes_ >= node_limit_) break;
            if (mode == SearchLimitMode::TIME && std::chrono::steady_clock::now() >= deadline_) break;
        }

        res.nodes      = nodes_;
        res.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return res;
    }

    // Fixed-depth score evaluation (used by MPC sigma measurer, no endgame override if disabled)
    int32_t eval_fixed_depth(uint64_t B, uint64_t W, int stm, int depth) {
        const int discs = __builtin_popcountll(B | W);
        const int ply   = discs - 4;
        const uint64_t hash = zk_.hash_board(B, W, stm);
        uint16_t root_idx[NUM_PATTERNS];
        pt_.extract_indices(B, W, root_idx);

        tt_gen_ = static_cast<uint8_t>((tt_gen_ + 1) & 0xFFU);
        if (tt_gen_ == 0) tt_gen_ = 1;
        nodes_ = 0; node_limit_ = 0; use_time_limit_ = false; aborted_ = false;
        return pvs_midgame(B, W, stm, ply, depth, hash, root_idx, -SEARCH_INF, SEARCH_INF, false, nullptr);
    }
};

} // namespace p2c64
