#include "p2c64_data.hpp"
#include "p2c64_patterns.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace p2c64;

static inline uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Canonical D4 x Z2 hash so symmetric duplicates ALWAYS land in the same Train/Val split
static inline uint64_t canonical_board_hash(uint64_t black, uint64_t white) noexcept {
    uint64_t min_b = UINT64_MAX;
    uint64_t min_w = UINT64_MAX;
    for (int sym = 0; sym < 16; ++sym) {
        uint64_t tb = 0, tw = 0;
        int dummy = 0;
        transform_board_d4z2(black, white, 1, sym, tb, tw, dummy);
        if (tb < min_b || (tb == min_b && tw < min_w)) {
            min_b = tb;
            min_w = tw;
        }
    }
    return splitmix64(min_b ^ splitmix64(min_w + 0x517cc1b727220a95ULL));
}

// Fast exact endgame alpha-beta solver (used by --generate-smoke for realistic final disc scores)
static int exact_solve_ab(uint64_t P, uint64_t O, int alpha, int beta, bool prev_pass) noexcept {
    uint64_t mv = generate_moves(P, O);
    if (!mv) {
        if (prev_pass) {
            int p_cnt = __builtin_popcountll(P);
            int o_cnt = __builtin_popcountll(O);
            int empty = 64 - p_cnt - o_cnt;
            if (p_cnt > o_cnt) return p_cnt - o_cnt + empty;
            if (p_cnt < o_cnt) return p_cnt - o_cnt - empty;
            return 0;
        }
        return -exact_solve_ab(O, P, -beta, -alpha, true);
    }
    int best = -65;
    while (mv) {
        int sq = __builtin_ctzll(mv);
        mv &= mv - 1;
        uint64_t fl = calc_flips(sq, P, O);
        int sc = -exact_solve_ab(O ^ fl, P | (1ULL << sq) | fl, -beta, -alpha, false);
        if (sc > best) best = sc;
        if (sc > alpha) {
            alpha = sc;
            if (alpha >= beta) break;
        }
    }
    return best;
}

// Heuristic square priority for semi-intelligent self-play smoke generation
static constexpr int SQ_PRIO[64] = {
    100, -25, 10,  5,  5, 10, -25, 100,
    -25, -40, -2, -2, -2, -2, -40, -25,
     10,  -2,  4,  1,  1,  4,  -2,  10,
      5,  -2,  1,  0,  0,  1,  -2,   5,
      5,  -2,  1,  0,  0,  1,  -2,   5,
     10,  -2,  4,  1,  1,  4,  -2,  10,
    -25, -40, -2, -2, -2, -2, -40, -25,
    100, -25, 10,  5,  5, 10, -25, 100
};

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() noexcept { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    int range(int lo, int hi) noexcept { return lo + static_cast<int>(next() % static_cast<uint64_t>(hi - lo + 1)); }
};

static int generate_smoke_dataset(const std::string& out_train,
                                  const std::string& out_val,
                                  size_t target_samples,
                                  int val_permille,
                                  uint64_t seed) {
    std::ofstream f_tr(out_train, std::ios::binary | std::ios::trunc);
    std::ofstream f_va(out_val,   std::ios::binary | std::ios::trunc);
    if (!f_tr || !f_va) {
        std::fprintf(stderr, "ERROR: Failed to open output files '%s' / '%s'\n", out_train.c_str(), out_val.c_str());
        return 1;
    }

    Rng rng(seed);
    size_t total_written = 0, train_cnt = 0, val_cnt = 0, games = 0;
    std::vector<PackedSample> game_buf;
    game_buf.reserve(64);

    while (total_written < target_samples) {
        games++;
        uint64_t B = (1ULL << rc(3, 4)) | (1ULL << rc(4, 3));
        uint64_t W = (1ULL << rc(3, 3)) | (1ULL << rc(4, 4));
        int stm = 1;
        int pass_streak = 0;
        game_buf.clear();

        const int rand_opening_plies = rng.range(4, 14);
        int final_black_diff = 0;
        bool solved_early = false;

        while (pass_streak < 2) {
            int discs = __builtin_popcountll(B | W);
            int empties = 64 - discs;
            int ply = discs - 4;
            uint64_t P = (stm > 0) ? B : W;
            uint64_t O = (stm > 0) ? W : B;
            uint64_t mv = generate_moves(P, O);

            if (empties <= 8 && !solved_early) {
                int exact_stm = exact_solve_ab(P, O, -64, 64, pass_streak > 0);
                final_black_diff = (stm > 0) ? exact_stm : -exact_stm;
                solved_early = true;
            }

            if (ply < NUM_PLIES) {
                PackedSample s{};
                s.black = B;
                s.white = W;
                s.score = 0;
                s.stm   = static_cast<int8_t>(stm);
                s.ply   = static_cast<uint8_t>(ply);
                s.flags = (mv == 0) ? 1U : 0U;
                game_buf.push_back(s);
            }

            if (!mv) {
                pass_streak++;
                stm = -stm;
                continue;
            }
            pass_streak = 0;

            int best_sq = -1;
            if (ply < rand_opening_plies || (rng.next() & 7U) == 0) {
                int cnt = __builtin_popcountll(mv);
                int pick = rng.range(0, cnt - 1);
                uint64_t m = mv;
                for (int k = 0; k <= pick; ++k) { best_sq = __builtin_ctzll(m); m &= m - 1; }
            } else {
                int best_val = -100000;
                uint64_t m = mv;
                while (m) {
                    int sq = __builtin_ctzll(m);
                    m &= m - 1;
                    uint64_t fl = calc_flips(sq, P, O);
                    int opp_mob = __builtin_popcountll(generate_moves(O ^ fl, P | (1ULL << sq) | fl));
                    int val = SQ_PRIO[sq] - 4 * opp_mob + rng.range(-6, 6);
                    if (val > best_val) { best_val = val; best_sq = sq; }
                }
            }

            uint64_t fl = calc_flips(best_sq, P, O);
            if (stm > 0) { B |= (1ULL << best_sq) | fl; W ^= fl; }
            else         { W |= (1ULL << best_sq) | fl; B ^= fl; }
            stm = -stm;
        }

        if (!solved_early) {
            int bc = __builtin_popcountll(B);
            int wc = __builtin_popcountll(W);
            int em = 64 - bc - wc;
            if (bc > wc)      final_black_diff = bc - wc + em;
            else if (bc < wc) final_black_diff = bc - wc - em;
            else              final_black_diff = 0;
        }

        for (auto& s : game_buf) {
            int stm_diff = (s.stm > 0) ? final_black_diff : -final_black_diff;
            stm_diff = std::clamp(stm_diff, -64, 64);
            s.score = static_cast<int8_t>(stm_diff);

            uint64_t h = canonical_board_hash(s.black, s.white);
            if (static_cast<int>(h % 10000ULL) < val_permille) {
                f_va.write(reinterpret_cast<const char*>(&s), sizeof(s));
                val_cnt++;
            } else {
                f_tr.write(reinterpret_cast<const char*>(&s), sizeof(s));
                train_cnt++;
            }
            total_written++;
            if (total_written >= target_samples) break;
        }
    }

    std::printf("[preprocess_egaroucid] Generated smoke dataset (%zu games -> %zu samples)\n", games, total_written);
    std::printf("  Train : %zu samples -> %s (%.2f MB)\n", train_cnt, out_train.c_str(),
                static_cast<double>(train_cnt * sizeof(PackedSample)) / (1024.0 * 1024.0));
    std::printf("  Val   : %zu samples -> %s (%.2f MB)\n", val_cnt, out_val.c_str(),
                static_cast<double>(val_cnt * sizeof(PackedSample)) / (1024.0 * 1024.0));
    return 0;
}

// ============================================================================
// Robust Auto-Detecting Parser for Egaroucid / Othello Training Text Files
// Supported line formats:
//   (A) 64-char board + [optional STM char/int] + score
//       Board chars: X/O/-, X/O/., B/W/-, B/W/., */O/-, */O/., 0/1/., 0/1/-
//   (B) 65-char board+STM + score (64 board + 1 STM char)
//   (C) Move transcript ("f5d6c3d3...") + [optional final score]
// ============================================================================
static inline bool is_board_token_64(const std::string& tok) noexcept {
    if (tok.size() != 64 && tok.size() != 65) return false;
    for (size_t i = 0; i < 64; ++i) {
        char c = tok[i];
        bool ok = (c == 'X' || c == 'O' || c == 'x' || c == 'o' ||
                   c == 'B' || c == 'W' || c == 'b' || c == 'w' ||
                   c == '*' || c == '-' || c == '.' || c == '_' ||
                   c == '0' || c == '1' || c == '2');
        if (!ok) return false;
    }
    return true;
}

static inline bool is_transcript_token(const std::string& tok) noexcept {
    if (tok.size() < 8 || tok.size() > 120 || (tok.size() & 1U) != 0) return false;
    for (size_t i = 0; i < tok.size(); i += 2) {
        char c = static_cast<char>(std::tolower(static_cast<unsigned char>(tok[i])));
        char r = tok[i + 1];
        if (c < 'a' || c > 'h' || r < '1' || r > '8') return false;
    }
    return true;
}

static inline bool parse_board_64(const std::string& tok, uint64_t& out_b, uint64_t& out_w, int& out_stm, bool& has_explicit_stm) noexcept {
    out_b = 0;
    out_w = 0;
    out_stm = 1;
    has_explicit_stm = false;

    bool has_01 = false, has_2 = false;
    for (size_t i = 0; i < 64; ++i) {
        if (tok[i] == '0' || tok[i] == '1') has_01 = true;
        if (tok[i] == '2') has_2 = true;
    }

    for (int sq = 0; sq < 64; ++sq) {
        char c = tok[static_cast<size_t>(sq)];
        if (has_01 && !has_2) {
            // Egaroucid numeric style: '0'=player/Black, '1'=opponent/White, '.'/'-'=Empty
            if (c == '0') out_b |= (1ULL << sq);
            else if (c == '1') out_w |= (1ULL << sq);
            else if (c == '.' || c == '-' || c == '_') {}
            else return false;
        } else if (has_2) {
            // Ternary style: '0'=White, '1'=Empty, '2'=Black
            if (c == '2') out_b |= (1ULL << sq);
            else if (c == '0') out_w |= (1ULL << sq);
            else if (c == '1' || c == '.' || c == '-') {}
            else return false;
        } else {
            char cu = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            if (cu == 'X' || cu == 'B' || cu == '*') out_b |= (1ULL << sq);
            else if (cu == 'O' || cu == 'W')         out_w |= (1ULL << sq);
            else if (cu == '-' || cu == '.' || cu == '_') {}
            else return false;
        }
    }

    if (tok.size() == 65) {
        char s = static_cast<char>(std::toupper(static_cast<unsigned char>(tok[64])));
        if (s == 'X' || s == 'B' || s == '*' || s == '0') { out_stm = 1;  has_explicit_stm = true; }
        else if (s == 'O' || s == 'W' || s == '1')        { out_stm = -1; has_explicit_stm = true; }
        else return false;
    }
    return true;
}

static inline bool parse_score_double(const std::string& tok, double& val) noexcept {
    if (tok.empty()) return false;
    char* endp = nullptr;
    val = std::strtod(tok.c_str(), &endp);
    return (endp != tok.c_str() && *endp == '\0' && std::isfinite(val));
}

static std::vector<std::string> split_tokens(const std::string& line) {
    std::vector<std::string> out;
    size_t i = 0, n = line.size();
    while (i < n) {
        while (i < n && (std::isspace(static_cast<unsigned char>(line[i])) || line[i] == ',' || line[i] == ';')) ++i;
        if (i >= n) break;
        size_t j = i;
        while (j < n && !std::isspace(static_cast<unsigned char>(line[j])) && line[j] != ',' && line[j] != ';') ++j;
        out.emplace_back(line.substr(i, j - i));
        i = j;
    }
    return out;
}

static void print_usage(const char* prog) {
    std::printf("Usage:\n");
    std::printf("  %s --input-dir <dir> --out-train <train.bin> --out-val <val.bin> [--val-pct 5] [--max-samples 0] [--score-perspective stm|black]\n", prog);
    std::printf("  %s --generate-smoke <N> --out-train <train.bin> --out-val <val.bin> [--val-pct 5] [--seed 42]\n", prog);
}

int main(int argc, char** argv) {
    std::string input_dir;
    std::string out_train = "train.bin";
    std::string out_val   = "val.bin";
    std::string score_persp = "stm";
    size_t smoke_n = 0;
    size_t max_samples = 0;
    int val_permille = 500; // 5.0%
    uint64_t seed = 20261125ULL;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input-dir" && i + 1 < argc)         input_dir = argv[++i];
        else if (arg == "--out-train" && i + 1 < argc)    out_train = argv[++i];
        else if (arg == "--out-val" && i + 1 < argc)      out_val = argv[++i];
        else if (arg == "--val-pct" && i + 1 < argc)      val_permille = static_cast<int>(std::llround(std::atof(argv[++i]) * 100.0));
        else if (arg == "--generate-smoke" && i + 1 < argc) smoke_n = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        else if (arg == "--max-samples" && i + 1 < argc)  max_samples = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        else if (arg == "--seed" && i + 1 < argc)         seed = static_cast<uint64_t>(std::strtoull(argv[++i], nullptr, 10));
        else if (arg == "--score-perspective" && i + 1 < argc) score_persp = argv[++i];
        else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str()); print_usage(argv[0]); return 1; }
    }

    if (smoke_n > 0) {
        return generate_smoke_dataset(out_train, out_val, smoke_n, val_permille, seed);
    }
    if (input_dir.empty()) {
        std::fprintf(stderr, "ERROR: Specify --input-dir <dir> or --generate-smoke <N>\n");
        print_usage(argv[0]);
        return 1;
    }

    // Recursively discover text/data files
    std::vector<fs::path> files;
    for (const auto& entry : fs::recursive_directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (ext == ".txt" || ext == ".csv" || ext == ".dat" || ext == ".edax" || ext == ".obo" || ext.empty()) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        std::fprintf(stderr, "ERROR: No data files found recursively under '%s'\n", input_dir.c_str());
        return 1;
    }

    std::ofstream f_tr(out_train, std::ios::binary | std::ios::trunc);
    std::ofstream f_va(out_val,   std::ios::binary | std::ios::trunc);
    if (!f_tr || !f_va) {
        std::fprintf(stderr, "ERROR: Failed to open output binary files\n");
        return 1;
    }

    // Buffered writers for high throughput
    std::vector<PackedSample> tr_buf, va_buf;
    tr_buf.reserve(65536);
    va_buf.reserve(16384);
    auto flush_bufs = [&]() {
        if (!tr_buf.empty()) {
            f_tr.write(reinterpret_cast<const char*>(tr_buf.data()), static_cast<std::streamsize>(tr_buf.size() * sizeof(PackedSample)));
            tr_buf.clear();
        }
        if (!va_buf.empty()) {
            f_va.write(reinterpret_cast<const char*>(va_buf.data()), static_cast<std::streamsize>(va_buf.size() * sizeof(PackedSample)));
            va_buf.clear();
        }
    };

    size_t total_lines = 0, parsed_board_lines = 0, parsed_transcript_lines = 0, skipped_lines = 0;
    size_t train_cnt = 0, val_cnt = 0, total_samples = 0;
    int preview_logged = 0;

    auto emit_sample = [&](uint64_t B, uint64_t W, int stm, int score_stm, const char* fmt_tag) {
        int discs = __builtin_popcountll(B | W);
        int ply = discs - 4;
        if (ply < 0 || ply >= NUM_PLIES) return;
        if ((B & W) != 0) return;

        uint64_t P = (stm > 0) ? B : W;
        uint64_t O = (stm > 0) ? W : B;
        PackedSample s{};
        s.black = B;
        s.white = W;
        s.score = static_cast<int8_t>(std::clamp(score_stm, -64, 64));
        s.stm   = static_cast<int8_t>(stm);
        s.ply   = static_cast<uint8_t>(ply);
        s.flags = (generate_moves(P, O) == 0) ? 1U : 0U;

        if (preview_logged < 5) {
            std::printf("  [sample #%d | %s] ply=%2u discs=%2d(B=%2d,W=%2d) stm=%+d score=%+d pass=%u\n",
                        preview_logged + 1, fmt_tag, s.ply, discs,
                        __builtin_popcountll(B), __builtin_popcountll(W),
                        static_cast<int>(s.stm), static_cast<int>(s.score), static_cast<unsigned>(s.flags));
            preview_logged++;
        }

        uint64_t h = canonical_board_hash(B, W);
        if (static_cast<int>(h % 10000ULL) < val_permille) {
            va_buf.push_back(s);
            val_cnt++;
            if (va_buf.size() >= 16384) flush_bufs();
        } else {
            tr_buf.push_back(s);
            train_cnt++;
            if (tr_buf.size() >= 65536) flush_bufs();
        }
        total_samples++;
    };

    std::printf("[preprocess_egaroucid] Scanning %zu files under '%s'...\n", files.size(), input_dir.c_str());

    std::string line;
    for (const auto& fpath : files) {
        if (max_samples > 0 && total_samples >= max_samples) break;
        std::ifstream ifs(fpath);
        if (!ifs) continue;

        while (std::getline(ifs, line)) {
            if (max_samples > 0 && total_samples >= max_samples) break;
            total_lines++;
            if (line.empty() || line[0] == '#') { skipped_lines++; continue; }

            auto toks = split_tokens(line);
            if (toks.empty()) { skipped_lines++; continue; }

            // Case 1: 64-char (or 65-char) board string
            if (is_board_token_64(toks[0])) {
                uint64_t B = 0, W = 0;
                int stm = 1;
                bool has_stm = false;
                if (!parse_board_64(toks[0], B, W, stm, has_stm)) { skipped_lines++; continue; }

                double sc_d = 0.0;
                bool got_score = false;
                if (toks.size() >= 2 && parse_score_double(toks[1], sc_d)) {
                    got_score = true;
                } else if (toks.size() >= 3) {
                    // toks[1] might be explicit STM ('X'/'O'/'B'/'W'/'1'/'-1'), toks[2] score
                    std::string st = toks[1];
                    char c0 = static_cast<char>(std::toupper(static_cast<unsigned char>(st[0])));
                    if (st == "X" || st == "B" || st == "*" || st == "+1" || st == "1" || c0 == 'B') {
                        stm = 1; has_stm = true;
                    } else if (st == "O" || st == "W" || st == "-1" || c0 == 'W') {
                        stm = -1; has_stm = true;
                    }
                    if (parse_score_double(toks[2], sc_d)) got_score = true;
                }

                if (!got_score || sc_d < -64.5 || sc_d > 64.5) { skipped_lines++; continue; }
                int sc_i = static_cast<int>(std::llround(sc_d));
                if (has_stm && score_persp == "black" && stm < 0) sc_i = -sc_i;
                emit_sample(B, W, stm, sc_i, "board64");
                parsed_board_lines++;
                continue;
            }

            // Case 2: Transcript line ("f5d6c3..." [optional score])
            if (is_transcript_token(toks[0])) {
                uint64_t B = (1ULL << rc(3, 4)) | (1ULL << rc(4, 3));
                uint64_t W = (1ULL << rc(3, 3)) | (1ULL << rc(4, 4));
                int stm = 1;
                struct Step { uint64_t b, w; int stm; };
                std::vector<Step> steps;
                steps.reserve(64);
                bool valid = true;
                const std::string& tr = toks[0];
                for (size_t k = 0; k < tr.size(); k += 2) {
                    int col = static_cast<char>(std::tolower(static_cast<unsigned char>(tr[k]))) - 'a';
                    int row = tr[k + 1] - '1';
                    int sq = rc(row, col);
                    uint64_t P = (stm > 0) ? B : W;
                    uint64_t O = (stm > 0) ? W : B;
                    if (generate_moves(P, O) == 0) {
                        stm = -stm;
                        P = (stm > 0) ? B : W;
                        O = (stm > 0) ? W : B;
                    }
                    uint64_t fl = calc_flips(sq, P, O);
                    if (fl == 0 || ((B | W) & (1ULL << sq))) { valid = false; break; }
                    steps.push_back({B, W, stm});
                    if (stm > 0) { B |= (1ULL << sq) | fl; W ^= fl; }
                    else         { W |= (1ULL << sq) | fl; B ^= fl; }
                    stm = -stm;
                }
                if (!valid || steps.empty()) { skipped_lines++; continue; }

                int final_black_diff = 0;
                double sc_d = 0.0;
                if (toks.size() >= 2 && parse_score_double(toks[1], sc_d) && sc_d >= -64.5 && sc_d <= 64.5) {
                    final_black_diff = static_cast<int>(std::llround(sc_d));
                } else {
                    int bc = __builtin_popcountll(B), wc = __builtin_popcountll(W), em = 64 - bc - wc;
                    final_black_diff = (bc > wc) ? (bc - wc + em) : ((bc < wc) ? (bc - wc - em) : 0);
                }
                for (const auto& st : steps) {
                    if (max_samples > 0 && total_samples >= max_samples) break;
                    int s_stm = (st.stm > 0) ? final_black_diff : -final_black_diff;
                    emit_sample(st.b, st.w, st.stm, s_stm, "transcript");
                }
                parsed_transcript_lines++;
                continue;
            }

            skipped_lines++;
        }
    }
    flush_bufs();

    std::printf("[preprocess_egaroucid] Done!\n");
    std::printf("  Files scanned : %zu | Lines read: %zu\n", files.size(), total_lines);
    std::printf("  Parsed lines  : board64=%zu, transcript=%zu | Skipped=%zu\n",
                parsed_board_lines, parsed_transcript_lines, skipped_lines);
    std::printf("  Total samples : %zu (Train=%zu [%.2f%%], Val=%zu [%.2f%%])\n",
                total_samples,
                train_cnt, total_samples ? 100.0 * train_cnt / total_samples : 0.0,
                val_cnt,   total_samples ? 100.0 * val_cnt   / total_samples : 0.0);
    std::printf("  Output sizes  : train=%.2f MB, val=%.2f MB\n",
                static_cast<double>(train_cnt * sizeof(PackedSample)) / (1024.0 * 1024.0),
                static_cast<double>(val_cnt   * sizeof(PackedSample)) / (1024.0 * 1024.0));
    return (total_samples > 0) ? 0 : 1;
}
