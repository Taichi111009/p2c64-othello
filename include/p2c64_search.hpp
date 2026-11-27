#pragma once

#include "p2c64_io.hpp"
#include "p2c64_patterns.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
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

static constexpr int8_t ADJ_CORNER_SQ[64] = {
    -1,  0, -1, -1, -1, -1,  7, -1,
     0,  0, -1, -1, -1, -1,  7,  7,
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
    56, 56, -1, -1, -1, -1, 63, 63,
    -1, 56, -1, -1, -1, -1, 63, -1
};

static inline int context_sq_score(int sq, uint64_t occ) noexcept {
    int s = static_cast<int>(STATIC_MOVE_ORDER[sq]);
    const int corner = static_cast<int>(ADJ_CORNER_SQ[sq]);
    if (corner >= 0 && ((occ >> corner) & 1ULL)) {
        // Adjacent corner already occupied: remove C/X penalty and reward safe corner-adjacent play
        s = (s <= -40) ? 18 : 28;
    }
    return s;
}

enum class MoveOrderMode : int {
    STATIC    = 0,
    B4_HYBRID = 1
};

static inline const char* move_order_mode_name(MoveOrderMode mo) noexcept {
    return (mo == MoveOrderMode::B4_HYBRID) ? "b4_hybrid" : "static";
}

enum class DynGateMode : int {
    OFF           = 0,
    PHASE         = 1,
    PHASE_TENSION = 2
};

static inline const char* dyn_gate_mode_name(DynGateMode dm) noexcept {
    switch (dm) {
        case DynGateMode::OFF:           return "off";
        case DynGateMode::PHASE:         return "phase";
        case DynGateMode::PHASE_TENSION: return "phase_tension";
    }
    return "off";
}

// Per-model calibrated Multi-ProbCut table (4 depth pairs x 60 plies = 2.88 KiB, L1D-resident!)
struct MpcCalEntry {
    float inv_slope       = 1.0f;   // 1.0f / clamp(slope, 0.85, 1.15)
    float intercept_units = 0.0f;   // intercept_discs * score_scale
    float sigma_units     = 256.0f; // clamp(sigma_discs, 0.80, 6.00) * score_scale
};

struct MpcCalTable {
    bool        loaded = false;
    MpcCalEntry table[4][NUM_PLIES]{}; // index 0..3 -> d_shallow = 1..4 (d_deep = 5..8)
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
    uint64_t main_leaves     = 0;
    uint64_t mpc_leaves      = 0;
    uint64_t s25_runs        = 0;
    uint64_t s3_runs         = 0;
    uint64_t mpc_tries       = 0;
    uint64_t mpc_cuts        = 0;
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

    // Standard Multi-ProbCut, Move Ordering & WACA-G2 2-stage gating parameters (100% fair across models!)
    bool          use_mpc_        = false;
    double        mpc_t_          = 1.80;
    bool          mpc_pure_v64_   = false; // Default OFF: fair gated WACA_G2 + TT write on MPC probes!
    MpcCalTable   mpc_cal_{};              // Optional model-specific calibrated MPC table (L1D-resident)
    MoveOrderMode mo_mode_        = MoveOrderMode::STATIC;
    GateMode      gate_mode_      = GateMode::SAFE_BND;
    DynGateMode   dyn_gate_mode_  = DynGateMode::OFF;
    double        dyn_r25_k_      = 0.50;
    int32_t       dyn_r25_q4_     = 0;     // round(dyn_r25_k * 16) when PHASE_TENSION else 0
    int32_t       m25_units_      = 320;   // 2.5 discs * 128
    int32_t       m3_units_       = 192;   // 1.5 discs * 128
    int32_t       tt_guard_units_ = 576;   // 4.5 discs * 128 (used by GateMode::WIDE_SAFE_BND)
    int32_t       m25_ply_[NUM_PLIES]{};   // Precomputed 60-ply Stage 1->2.5 margins (0 ns lookup!)
    int32_t       m3_ply_[NUM_PLIES]{};    // Precomputed 60-ply Stage 2.5->3 margins (0 ns lookup!)

    // B4_HYBRID Killer (2 slots per ply) & History (2 sides x 64 sq) tables
    uint8_t  killer_[64][2]{};
    int32_t  history_[2][64]{};

    // Runtime limits & counters
    uint64_t nodes_           = 0;
    uint64_t node_limit_      = 0;
    bool     use_time_limit_  = false;
    bool     aborted_         = false;
    std::chrono::steady_clock::time_point deadline_;

    // Per-search telemetry
    uint64_t stat_main_leaves_ = 0;
    uint64_t stat_mpc_leaves_  = 0;
    uint64_t stat_s25_runs_    = 0;
    uint64_t stat_s3_runs_     = 0;
    uint64_t stat_mpc_tries_   = 0;
    uint64_t stat_mpc_cuts_    = 0;

    explicit PVSEngine(const LoadedModel* model, size_t tt_mb = 64, int endgame_empties = 10)
        : model_(model), pt_(get_pattern_tables()), zk_(get_zobrist()), endgame_empties_(endgame_empties) {
        allocate_tt(tt_mb);
        clear_heuristics();
        rebuild_gate_tables();
    }

    void set_mpc(bool enable, double t_val = 1.80) noexcept {
        use_mpc_ = enable;
        mpc_t_   = t_val;
    }

    void set_mpc_pure_v64(bool enable) noexcept {
        mpc_pure_v64_ = enable;
    }

    void set_move_order_mode(MoveOrderMode mo) noexcept {
        mo_mode_ = mo;
    }

    void rebuild_gate_tables() noexcept {
        if (dyn_gate_mode_ == DynGateMode::OFF) {
            for (int p = 0; p < NUM_PLIES; ++p) {
                m25_ply_[p] = m25_units_;
                m3_ply_[p]  = m3_units_;
            }
            dyn_r25_q4_ = 0;
        } else {
            for (int p = 0; p < NUM_PLIES; ++p) {
                double mult = 1.00;
                if (p <= 15) {
                    mult = 0.85;
                } else if (p <= 27) {
                    mult = 0.85 + 0.30 * static_cast<double>(p - 15) / 12.0;
                } else if (p <= 47) {
                    mult = 1.15;
                } else {
                    mult = 1.00;
                }
                const int32_t m25_p = static_cast<int32_t>(std::lround(static_cast<double>(m25_units_) * mult));
                const int32_t m3_p  = static_cast<int32_t>(std::lround(static_cast<double>(m3_units_)  * mult));
                m25_ply_[p] = std::clamp<int32_t>(m25_p, 0, tt_guard_units_);
                m3_ply_[p]  = std::clamp<int32_t>(m3_p,  0, tt_guard_units_);
            }
            dyn_r25_q4_ = (dyn_gate_mode_ == DynGateMode::PHASE_TENSION)
                ? static_cast<int32_t>(std::lround(std::max(0.0, dyn_r25_k_) * 16.0))
                : 0;
        }
    }

    void set_waca_gate(GateMode gm, double m25_discs, double m3_discs, double tt_guard_discs = 4.5) noexcept {
        gate_mode_      = gm;
        const double sc = static_cast<double>(model_->score_scale);
        m25_units_      = static_cast<int32_t>(m25_discs * sc + 0.5);
        m3_units_       = static_cast<int32_t>(m3_discs * sc + 0.5);
        tt_guard_units_ = static_cast<int32_t>(tt_guard_discs * sc + 0.5);
        rebuild_gate_tables();
    }

    void set_dyn_gate(DynGateMode dm, double r25_k = 0.50) noexcept {
        dyn_gate_mode_ = dm;
        dyn_r25_k_     = r25_k;
        rebuild_gate_tables();
    }

    void clear_mpc_calibration() noexcept {
        mpc_cal_.loaded = false;
    }

    // Load per-model MPC calibration CSV:
    //   d_shallow,d_deep,ply_lo,ply_hi,ply_center,n,slope,intercept,sigma,r2
    // Builds piecewise-linear interpolated 4 x 60 L1D lookup table (d_sh = 1..4).
    bool load_mpc_calibration(const std::string& csv_path, std::string* err_msg = nullptr) {
        auto fail = [&](const std::string& msg) -> bool {
            if (err_msg) *err_msg = msg;
            return false;
        };
        std::ifstream ifs(csv_path);
        if (!ifs) return fail("Cannot open MPC calibration CSV: " + csv_path);

        struct CalPt {
            double ply_center;
            double slope;
            double intercept;
            double sigma;
        };
        std::vector<CalPt> pts[4];
        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            if (line.rfind("d_shallow", 0) == 0) continue; // header
            std::stringstream ss(line);
            std::string tok;
            std::vector<std::string> cols;
            while (std::getline(ss, tok, ',')) cols.push_back(tok);
            if (cols.size() < 9) continue;

            const int d_sh = std::atoi(cols[0].c_str());
            if (d_sh < 1 || d_sh > 4) continue;
            const double ply_c = std::atof(cols[4].c_str());
            const double slope = std::atof(cols[6].c_str());
            const double bias  = std::atof(cols[7].c_str());
            const double sig   = std::atof(cols[8].c_str());
            if (!std::isfinite(ply_c) || !std::isfinite(slope) || !std::isfinite(bias) || !std::isfinite(sig)) {
                continue;
            }
            pts[d_sh - 1].push_back({ply_c, slope, bias, sig});
        }

        const double sc = static_cast<double>(model_->score_scale);
        for (int d_idx = 0; d_idx < 4; ++d_idx) {
            auto& vec = pts[d_idx];
            if (vec.empty()) {
                return fail("Missing calibration rows for d_shallow=" + std::to_string(d_idx + 1) + " in " + csv_path);
            }
            std::sort(vec.begin(), vec.end(), [](const CalPt& a, const CalPt& b) {
                return a.ply_center < b.ply_center;
            });
            for (int p = 0; p < NUM_PLIES; ++p) {
                const double pf = static_cast<double>(p);
                double sl = vec.front().slope;
                double bi = vec.front().intercept;
                double sg = vec.front().sigma;
                if (pf <= vec.front().ply_center) {
                    sl = vec.front().slope;
                    bi = vec.front().intercept;
                    sg = vec.front().sigma;
                } else if (pf >= vec.back().ply_center) {
                    sl = vec.back().slope;
                    bi = vec.back().intercept;
                    sg = vec.back().sigma;
                } else {
                    for (size_t k = 0; k + 1 < vec.size(); ++k) {
                        if (pf >= vec[k].ply_center && pf <= vec[k + 1].ply_center) {
                            const double denom = vec[k + 1].ply_center - vec[k].ply_center;
                            const double t = (denom > 1e-9) ? ((pf - vec[k].ply_center) / denom) : 0.0;
                            sl = vec[k].slope     + t * (vec[k + 1].slope     - vec[k].slope);
                            bi = vec[k].intercept + t * (vec[k + 1].intercept - vec[k].intercept);
                            sg = vec[k].sigma     + t * (vec[k + 1].sigma     - vec[k].sigma);
                            break;
                        }
                    }
                }
                const double sl_c = std::clamp(sl, 0.85, 1.15);
                const double sg_c = std::clamp(sg, 0.80, 6.00);
                mpc_cal_.table[d_idx][p].inv_slope       = static_cast<float>(1.0 / sl_c);
                mpc_cal_.table[d_idx][p].intercept_units = static_cast<float>(bi * sc);
                mpc_cal_.table[d_idx][p].sigma_units     = static_cast<float>(sg_c * sc);
            }
        }
        mpc_cal_.loaded = true;
        return true;
    }

    static inline double mpc_sigma_discs(int d_sh, int ply) noexcept {
        if (d_sh <= 1) return 2.65 + 0.030 * static_cast<double>(ply);
        if (d_sh == 2) return 2.15 + 0.035 * static_cast<double>(ply);
        if (d_sh == 3) return 1.95 + 0.032 * static_cast<double>(ply);
        return 1.70 + 0.030 * static_cast<double>(ply);
    }

    void allocate_tt(size_t tt_mb) {
        size_t bytes = std::max<size_t>(1, tt_mb) * 1024ULL * 1024ULL;
        size_t entries = 1;
        while ((entries << 1) * sizeof(SearchTTEntry) <= bytes) entries <<= 1;
        tt_.assign(entries, SearchTTEntry{0, 0, -1, TT_NONE, 64, 0});
        tt_mask_ = entries - 1;
    }

    void clear_killers() noexcept {
        std::memset(killer_, 64, sizeof(killer_));
    }

    void clear_heuristics() noexcept {
        clear_killers();
        std::memset(history_, 0, sizeof(history_));
    }

    void decay_history() noexcept {
        for (int s = 0; s < 2; ++s) {
            for (int sq = 0; sq < 64; ++sq) {
                history_[s][sq] >>= 1;
            }
        }
    }

    void clear_tt() {
        std::memset(tt_.data(), 0, tt_.size() * sizeof(SearchTTEntry));
        tt_gen_ = 1;
        clear_heuristics();
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

    // Midgame PVS with Incremental 64-Pattern Index Updates + Standard MPC + WACA-G2 2-Stage Gating
    int32_t pvs_midgame(uint64_t B, uint64_t W, int stm, int ply, int depth,
                        uint64_t hash, const uint16_t idx[NUM_PATTERNS],
                        int32_t alpha, int32_t beta, bool prev_pass,
                        int* out_best_move = nullptr, bool in_mpc_probe = false) noexcept {
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
            if (in_mpc_probe) stat_mpc_leaves_++;
            else              stat_main_leaves_++;
            const int cply = std::clamp(ply, 0, NUM_PLIES - 1);
            return model_->evaluate_search(pt_.meta, idx, B, W, stm, ply, alpha, beta,
                                           (in_mpc_probe && mpc_pure_v64_), gate_mode_,
                                           m25_ply_[cply], m3_ply_[cply],
                                           stat_s25_runs_, stat_s3_runs_, tt_guard_units_,
                                           dyn_r25_q4_);
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

        // Standard / Calibrated Multi-ProbCut (100% fair per-model Buro regression when mpc_cal_.loaded)
        if (use_mpc_ && !out_best_move && !in_mpc_probe &&
            beta == alpha + 1 && depth >= 5 && empties > endgame_empties_ + 4) {
            const int d_sh = depth - 4;
            int32_t hi = 0;
            int32_t lo = 0;
            if (mpc_cal_.loaded) {
                const int d_idx = std::clamp(d_sh, 1, 4) - 1;
                const int cply  = std::clamp(ply, 0, NUM_PLIES - 1);
                const MpcCalEntry& cal = mpc_cal_.table[d_idx][cply];
                const float delta = static_cast<float>(mpc_t_) * cal.sigma_units;
                hi = static_cast<int32_t>(std::lround((static_cast<float>(beta)  - cal.intercept_units + delta) * cal.inv_slope));
                lo = static_cast<int32_t>(std::lround((static_cast<float>(alpha) - cal.intercept_units - delta) * cal.inv_slope));
            } else {
                const double sig = mpc_sigma_discs(d_sh, ply);
                const int32_t bound_delta = static_cast<int32_t>(
                    mpc_t_ * sig * static_cast<double>(model_->score_scale) + 0.5);
                hi = beta + bound_delta;
                lo = alpha - bound_delta;
            }
            stat_mpc_tries_++;

            if (hi < SEARCH_INF / 2) {
                const int32_t v_sh = pvs_midgame(B, W, stm, ply, d_sh, hash, idx,
                                                 hi - 1, hi, prev_pass, nullptr, true);
                if (aborted_) return 0;
                if (v_sh >= hi) {
                    stat_mpc_cuts_++;
                    return beta;
                }
            }
            if (lo > -SEARCH_INF / 2) {
                const int32_t v_sh = pvs_midgame(B, W, stm, ply, d_sh, hash, idx,
                                                 lo, lo + 1, prev_pass, nullptr, true);
                if (aborted_) return 0;
                if (v_sh <= lo) {
                    stat_mpc_cuts_++;
                    return alpha;
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
            return -pvs_midgame(B, W, -stm, ply, depth, hash ^ zk_.stm_key, idx,
                                -beta, -alpha, true, nullptr, in_mpc_probe);
        }

        int moves[32];
        int scores[32];
        uint64_t cached_flips[32];
        int n_moves = 0;
        const int side = (stm > 0) ? 0 : 1;
        const int kply = std::clamp(ply, 0, 63);
        const bool use_b4 = (mo_mode_ == MoveOrderMode::B4_HYBRID && depth >= 2);
        const bool use_ff = (use_b4 && depth >= 4);

        uint64_t m = mv;
        if (!use_b4) {
            while (m) {
                int sq = __builtin_ctzll(m);
                m &= m - 1;
                moves[n_moves]        = sq;
                scores[n_moves]       = (sq == tt_move) ? 1000000 : static_cast<int>(STATIC_MOVE_ORDER[sq]);
                cached_flips[n_moves] = 0ULL;
                n_moves++;
            }
        } else {
            const uint64_t occ = B | W;
            const uint8_t k0   = killer_[kply][0];
            const uint8_t k1   = killer_[kply][1];
            if (use_ff) {
                while (m) {
                    int sq = __builtin_ctzll(m);
                    m &= m - 1;
                    moves[n_moves] = sq;
                    if (sq == tt_move) {
                        scores[n_moves]       = 1000000;
                        cached_flips[n_moves] = 0ULL;
                    } else {
                        const uint64_t fl = calc_flips(sq, P, O);
                        cached_flips[n_moves] = fl;
                        const uint64_t nP = P | (1ULL << sq) | fl;
                        const uint64_t nO = O ^ fl;
                        const int opp_mob = popcount64_swar(generate_moves(nO, nP));
                        int sc = -opp_mob * 128 + context_sq_score(sq, occ) * 4;
                        if (opp_mob == 0) sc += 400;
                        if (sq == k0)      sc += 320;
                        else if (sq == k1) sc += 200;
                        sc += (history_[side][sq] >> 4);
                        scores[n_moves] = sc;
                    }
                    n_moves++;
                }
            } else {
                // depth == 2..3: zero calc_flips corner-occupancy context + 2-slot Killer + History
                while (m) {
                    int sq = __builtin_ctzll(m);
                    m &= m - 1;
                    moves[n_moves]        = sq;
                    cached_flips[n_moves] = 0ULL;
                    if (sq == tt_move) {
                        scores[n_moves] = 1000000;
                    } else {
                        int sc = context_sq_score(sq, occ) * 8;
                        if (sq == k0)      sc += 600;
                        else if (sq == k1) sc += 350;
                        sc += (history_[side][sq] >> 3);
                        scores[n_moves] = sc;
                    }
                    n_moves++;
                }
            }
        }

        int32_t best_val = -SEARCH_INF;
        uint8_t best_sq  = static_cast<uint8_t>(moves[0]);
        uint16_t child_idx[NUM_PATTERNS];
        const bool need_child_hash = !(depth == 1 && empties - 1 > endgame_empties_);

        for (int i = 0; i < n_moves; ++i) {
            int best_idx = i;
            for (int j = i + 1; j < n_moves; ++j) {
                if (scores[j] > scores[best_idx]) best_idx = j;
            }
            if (best_idx != i) {
                std::swap(moves[i], moves[best_idx]);
                std::swap(scores[i], scores[best_idx]);
                std::swap(cached_flips[i], cached_flips[best_idx]);
            }

            const int sq = moves[i];
            const uint64_t fl = cached_flips[i] ? cached_flips[i] : calc_flips(sq, P, O);
            uint64_t nB = B, nW = W;
            uint64_t nh = hash ^ zk_.stm_key;
            if (stm > 0) {
                nB |= (1ULL << sq) | fl; nW ^= fl;
                if (need_child_hash) {
                    nh ^= zk_.sq_b[sq];
                    uint64_t fm = fl;
                    while (fm) { int s = __builtin_ctzll(fm); fm &= fm - 1; nh ^= zk_.sq_b[s] ^ zk_.sq_w[s]; }
                }
            } else {
                nW |= (1ULL << sq) | fl; nB ^= fl;
                if (need_child_hash) {
                    nh ^= zk_.sq_w[sq];
                    uint64_t fm = fl;
                    while (fm) { int s = __builtin_ctzll(fm); fm &= fm - 1; nh ^= zk_.sq_b[s] ^ zk_.sq_w[s]; }
                }
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
                    val = -pvs_midgame(nB, nW, -stm, ply + 1, depth - 1, nh, child_idx,
                                       -beta, -alpha, false, nullptr, in_mpc_probe);
                } else {
                    val = -pvs_midgame(nB, nW, -stm, ply + 1, depth - 1, nh, child_idx,
                                       -alpha - 1, -alpha, false, nullptr, in_mpc_probe);
                    if (!aborted_ && val > alpha && val < beta) {
                        val = -pvs_midgame(nB, nW, -stm, ply + 1, depth - 1, nh, child_idx,
                                           -beta, -alpha, false, nullptr, in_mpc_probe);
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
                if (alpha >= beta) {
                    if (use_b4) {
                        const uint8_t usq = static_cast<uint8_t>(sq);
                        if (killer_[kply][0] != usq) {
                            killer_[kply][1] = killer_[kply][0];
                            killer_[kply][0] = usq;
                        }
                        history_[side][sq] += depth * depth;
                        if (history_[side][sq] > 65536) {
                            for (int hsq = 0; hsq < 64; ++hsq) history_[side][hsq] >>= 1;
                        }
                    }
                    break;
                }
            }
        }

        if (!aborted_) {
            if (!(in_mpc_probe && model_->is_any_waca() && mpc_pure_v64_)) {
                SearchTTEntry& e = tt_[hash & tt_mask_];
                if (e.flag == TT_NONE || e.gen != tt_gen_ || e.depth <= depth) {
                    e.key       = hash;
                    e.score     = best_val;
                    e.depth     = static_cast<int8_t>(depth);
                    e.flag      = (best_val <= orig_alpha) ? TT_UPPER : ((best_val >= beta) ? TT_LOWER : TT_EXACT);
                    e.best_move = best_sq;
                    e.gen       = tt_gen_;
                }
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

        const int discs   = __builtin_popcountll(B | W);
        const int ply     = discs - 4;
        const int empties = 64 - discs;
        const uint64_t hash = zk_.hash_board(B, W, stm);

        uint16_t root_idx[NUM_PATTERNS];
        pt_.extract_indices(B, W, root_idx);

        tt_gen_ = static_cast<uint8_t>((tt_gen_ + 1) & 0xFFU);
        if (tt_gen_ == 0) tt_gen_ = 1;
        if (mo_mode_ == MoveOrderMode::B4_HYBRID) {
            clear_killers();
            decay_history();
        }

        nodes_            = 0;
        aborted_          = false;
        stat_main_leaves_ = 0;
        stat_mpc_leaves_  = 0;
        stat_s25_runs_    = 0;
        stat_s3_runs_     = 0;
        stat_mpc_tries_   = 0;
        stat_mpc_cuts_    = 0;
        node_limit_       = (mode == SearchLimitMode::NODES) ? static_cast<uint64_t>(limit_val) : 0ULL;
        use_time_limit_   = (mode == SearchLimitMode::TIME);
        if (use_time_limit_) {
            deadline_ = t0 + std::chrono::milliseconds(limit_val);
        }

        // Exact endgame directly when empties <= endgame_empties_
        if (empties <= endgame_empties_) {
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
            int32_t val = pvs_midgame(B, W, stm, ply, d, hash, root_idx,
                                      -SEARCH_INF, SEARCH_INF, false, &cur_best, false);

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

        res.nodes       = nodes_;
        res.elapsed_ms  = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        res.main_leaves = stat_main_leaves_;
        res.mpc_leaves  = stat_mpc_leaves_;
        res.s25_runs    = stat_s25_runs_;
        res.s3_runs     = stat_s3_runs_;
        res.mpc_tries   = stat_mpc_tries_;
        res.mpc_cuts    = stat_mpc_cuts_;
        return res;
    }

    // Fixed-depth score evaluation (used by MPC sigma measurer; tier1_only=true measures MPC-PureV64!)
    int32_t eval_fixed_depth(uint64_t B, uint64_t W, int stm, int depth, bool tier1_only = false) {
        const int discs = __builtin_popcountll(B | W);
        const int ply   = discs - 4;
        const uint64_t hash = zk_.hash_board(B, W, stm);
        uint16_t root_idx[NUM_PATTERNS];
        pt_.extract_indices(B, W, root_idx);

        tt_gen_ = static_cast<uint8_t>((tt_gen_ + 1) & 0xFFU);
        if (tt_gen_ == 0) tt_gen_ = 1;
        nodes_ = 0; node_limit_ = 0; use_time_limit_ = false; aborted_ = false;
        const bool saved_mpc  = use_mpc_;
        const bool saved_pure = mpc_pure_v64_;
        use_mpc_ = false;
        if (tier1_only) mpc_pure_v64_ = true;
        int32_t sc = pvs_midgame(B, W, stm, ply, depth, hash, root_idx,
                                 -SEARCH_INF, SEARCH_INF, false, nullptr, tier1_only);
        use_mpc_      = saved_mpc;
        mpc_pure_v64_ = saved_pure;
        return sc;
    }
};

} // namespace p2c64
