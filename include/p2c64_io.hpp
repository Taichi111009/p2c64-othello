#pragma once

#include "p2c64_eval.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace p2c64 {

// ============================================================================
// P2C-64 Binary Weight File Format (.p2c: v2.0 / v3.0 / v3.1 / v3.2 backward-compatible)
//   - Fixed 64-byte little-endian header + FNV-1a 64-bit checksum
//   - v2.0 (magic="P2C64V20", version=2) : RAW_NTUPLE_64 (46.92 MiB) & v2.0 models (50.61 MiB)
//   - v3.0 (magic="P2C64V30", version=3) : WACA_G2     (53.96 MiB, 20-blk EO, F25=16, F3=24)
//   - v3.1 (magic="P2C64V31", version=4) : WACA_G2_V31 (57.48 MiB, 30-blk EO, F25=8,  F3=32)
//   - v3.2 (magic="P2C64V32", version=5) : WACA_G2_V32 (57.47 MiB, 30-blk EO, F25=8,  F3=28)
// ============================================================================
static constexpr char     P2C_MAGIC[8]           = {'P','2','C','6','4','V','2','0'};
static constexpr uint32_t P2C_FORMAT_VERSION     = 2U;
static constexpr char     P2C_MAGIC_V3[8]        = {'P','2','C','6','4','V','3','0'};
static constexpr uint32_t P2C_FORMAT_VERSION_V3  = 3U;
static constexpr char     P2C_MAGIC_V31[8]       = {'P','2','C','6','4','V','3','1'};
static constexpr uint32_t P2C_FORMAT_VERSION_V31 = 4U;
static constexpr char     P2C_MAGIC_V32[8]       = {'P','2','C','6','4','V','3','2'};
static constexpr uint32_t P2C_FORMAT_VERSION_V32 = 5U;
static constexpr float    P2C_DEFAULT_SCALE      = 128.0f;

#pragma pack(push, 1)
struct P2CFileHeader {
    char     magic[8];
    uint32_t version;            // 2 (v2.0), 3 (v3.0), 4 (v3.1), or 5 (v3.2 WACA_G2_V32)
    uint32_t eval_mode;          // EvalMode enum value [0..8]
    float    score_scale;        // Integer eval units per 1.0 disc
    uint32_t flags;              // Bit 0 = D4 symmetrized
    uint32_t key_raw_entries;    // 91,854
    uint32_t nonkey_raw_entries; // 318,168
    uint32_t total_raw_entries;  // 410,022
    uint32_t num_plies;          // 60
    uint32_t num_blocks;         // 10 (v2.0), 20 (v3.0 EO), or 30 (v3.1/v3.2 EO)
    uint32_t reserved;           // 0 (v2.0) or mlp_hidden (v3.0/v3.1/v3.2: 16 or 24)
    uint64_t payload_bytes;      // Exact payload length in bytes
    uint64_t checksum_fnv1a;     // 64-bit FNV-1a checksum of payload bytes
};
#pragma pack(pop)
static_assert(sizeof(P2CFileHeader) == 64, "P2CFileHeader must be strictly 64 bytes");

static inline uint64_t fnv1a_64_update(uint64_t h, const void* data, size_t len) noexcept {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

static inline uint64_t fnv1a_64_init() noexcept {
    return 14695981039346656037ULL;
}

static inline const char* eval_mode_name(EvalMode mode) noexcept {
    switch (mode) {
        case EvalMode::RAW_NTUPLE_64:         return "RAW_NTUPLE_64";
        case EvalMode::LINEAR_64:             return "LINEAR_64";
        case EvalMode::CAND_A_PTM_SCRELU4:    return "CAND_A_PTM_SCRELU4";
        case EvalMode::CAND_B_PTM_3W3B_NAIVE: return "CAND_B_PTM_3W3B_NAIVE";
        case EvalMode::CAND_B_PTM_3W3B:       return "CAND_B_PTM_3W3B";
        case EvalMode::CAND_C_PTM_HYBRID:     return "CAND_C_PTM_HYBRID";
        case EvalMode::WACA_G2:               return "WACA_G2";
        case EvalMode::WACA_G2_V31:           return "WACA_G2_V31";
        case EvalMode::WACA_G2_V32:           return "WACA_G2_V32";
    }
    return "UNKNOWN";
}

struct LoadedModel {
    EvalMode                          mode        = EvalMode::LINEAR_64;
    float                             score_scale = P2C_DEFAULT_SCALE;
    uint32_t                          flags       = 0;
    RawNTuple64Weights                raw;
    P2C64Weights                      p2c;
    WacaG2Weights                     waca;
    std::unique_ptr<WacaG2V31Weights> waca_v31;
    std::unique_ptr<WacaG2V32Weights> waca_v32;
    bool                              is_raw      = false;
    bool                              is_waca     = false;
    bool                              is_waca_v31 = false;
    bool                              is_waca_v32 = false;

    inline bool is_any_waca() const noexcept { return is_waca || is_waca_v31 || is_waca_v32; }

    inline int32_t evaluate(const PatternMeta* __restrict meta,
                            const uint16_t* __restrict idx,
                            int stm, int ply) const noexcept {
        switch (mode) {
            case EvalMode::RAW_NTUPLE_64:
                return eval_raw_ntuple64(raw, meta, idx, stm, ply);
            case EvalMode::LINEAR_64:
                return evaluate_p2c64<EvalMode::LINEAR_64>(p2c, meta, idx, stm, ply);
            case EvalMode::CAND_A_PTM_SCRELU4:
                return evaluate_p2c64<EvalMode::CAND_A_PTM_SCRELU4>(p2c, meta, idx, stm, ply);
            case EvalMode::CAND_B_PTM_3W3B_NAIVE:
                return evaluate_p2c64<EvalMode::CAND_B_PTM_3W3B_NAIVE>(p2c, meta, idx, stm, ply);
            case EvalMode::CAND_B_PTM_3W3B:
                return evaluate_p2c64<EvalMode::CAND_B_PTM_3W3B>(p2c, meta, idx, stm, ply);
            case EvalMode::CAND_C_PTM_HYBRID:
                return evaluate_p2c64<EvalMode::CAND_C_PTM_HYBRID>(p2c, meta, idx, stm, ply);
            case EvalMode::WACA_G2:
                return eval_raw_ntuple64(waca.tier1, meta, idx, stm, ply);
            case EvalMode::WACA_G2_V31:
                return eval_raw_ntuple64(waca_v31->tier1, meta, idx, stm, ply);
            case EvalMode::WACA_G2_V32:
                return eval_raw_ntuple64(waca_v32->tier1, meta, idx, stm, ply);
        }
        return 0;
    }

    inline int32_t evaluate_board(const PatternMeta* __restrict meta,
                                  const uint16_t* __restrict idx,
                                  uint64_t B, uint64_t W,
                                  int stm, int ply) const noexcept {
        if (is_waca_v32) {
            return eval_waca_g2_v32_full(*waca_v32, meta, idx, B, W, stm, ply);
        }
        if (is_waca_v31) {
            return eval_waca_g2_v31_full(*waca_v31, meta, idx, B, W, stm, ply);
        }
        if (is_waca) {
            return eval_waca_g2_full(waca, meta, idx, B, W, stm, ply);
        }
        return evaluate(meta, idx, stm, ply);
    }

    inline int32_t evaluate_tier1_only(const PatternMeta* __restrict meta,
                                       const uint16_t* __restrict idx,
                                       int stm, int ply) const noexcept {
        if (is_waca_v32) {
            return eval_raw_ntuple64(waca_v32->tier1, meta, idx, stm, ply);
        }
        if (is_waca_v31) {
            return eval_raw_ntuple64(waca_v31->tier1, meta, idx, stm, ply);
        }
        if (is_waca) {
            return eval_raw_ntuple64(waca.tier1, meta, idx, stm, ply);
        }
        return evaluate(meta, idx, stm, ply);
    }

    inline int32_t evaluate_search(const PatternMeta* __restrict meta,
                                   const uint16_t* __restrict idx,
                                   uint64_t B, uint64_t W,
                                   int stm, int ply,
                                   int32_t alpha, int32_t beta,
                                   bool in_mpc_probe,
                                   GateMode gate_mode,
                                   int32_t m25_units,
                                   int32_t m3_units,
                                   uint64_t& stat_s25,
                                   uint64_t& stat_s3,
                                   int32_t tt_guard_units = 576) const noexcept {
        if (is_waca_v32) {
            return eval_waca_g2_v32_gated(*waca_v32, meta, idx, B, W, stm, ply, alpha, beta,
                                          in_mpc_probe, gate_mode, m25_units, m3_units,
                                          stat_s25, stat_s3, tt_guard_units);
        }
        if (is_waca_v31) {
            return eval_waca_g2_v31_gated(*waca_v31, meta, idx, B, W, stm, ply, alpha, beta,
                                          in_mpc_probe, gate_mode, m25_units, m3_units,
                                          stat_s25, stat_s3, tt_guard_units);
        }
        if (is_waca) {
            return eval_waca_g2_gated(waca, meta, idx, B, W, stm, ply, alpha, beta,
                                      in_mpc_probe, gate_mode, m25_units, m3_units,
                                      stat_s25, stat_s3, tt_guard_units);
        }
        return evaluate(meta, idx, stm, ply);
    }
};

inline bool save_raw_p2c(const std::string& path,
                         const RawNTuple64Weights& raw,
                         float score_scale = P2C_DEFAULT_SCALE,
                         uint32_t flags = 1U) {
    const auto& pt = get_pattern_tables();
    const size_t w_bytes = static_cast<size_t>(pt.total_raw_entries) * sizeof(int16_t);
    std::vector<uint8_t> payload;
    payload.reserve(NUM_PLIES * w_bytes);
    for (int ply = 0; ply < NUM_PLIES; ++ply) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(raw.w_ply[ply].data());
        payload.insert(payload.end(), p, p + w_bytes);
    }

    P2CFileHeader hdr{};
    std::memcpy(hdr.magic, P2C_MAGIC, 8);
    hdr.version            = P2C_FORMAT_VERSION;
    hdr.eval_mode          = static_cast<uint32_t>(EvalMode::RAW_NTUPLE_64);
    hdr.score_scale        = score_scale;
    hdr.flags              = flags;
    hdr.key_raw_entries    = static_cast<uint32_t>(pt.key_raw_entries);
    hdr.nonkey_raw_entries = static_cast<uint32_t>(pt.nonkey_raw_entries);
    hdr.total_raw_entries  = static_cast<uint32_t>(pt.total_raw_entries);
    hdr.num_plies          = static_cast<uint32_t>(NUM_PLIES);
    hdr.num_blocks         = static_cast<uint32_t>(NUM_BLOCKS);
    hdr.reserved           = 0U;
    hdr.payload_bytes      = static_cast<uint64_t>(payload.size());
    hdr.checksum_fnv1a     = fnv1a_64_update(fnv1a_64_init(), payload.data(), payload.size());

    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) return false;
    ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    ofs.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    return static_cast<bool>(ofs);
}

inline bool save_p2c64(const std::string& path,
                       const P2C64Weights& w,
                       EvalMode mode,
                       float score_scale = P2C_DEFAULT_SCALE,
                       uint32_t flags = 1U) {
    if (mode == EvalMode::RAW_NTUPLE_64) return false;
    const auto& pt = get_pattern_tables();
    const size_t kb_bytes = static_cast<size_t>(pt.key_raw_entries) * sizeof(BlockEntry16);
    const size_t nk_bytes = static_cast<size_t>(pt.nonkey_raw_entries) * sizeof(int16_t);
    const size_t sc_bytes = sizeof(SCReLUPlyParams);
    const size_t ml_bytes = 1536 * sizeof(int16_t);

    std::vector<uint8_t> payload;
    payload.reserve(NUM_BLOCKS * kb_bytes + NUM_PLIES * (nk_bytes + sc_bytes + ml_bytes));

    auto append = [&](const void* src, size_t n) {
        const uint8_t* p = static_cast<const uint8_t*>(src);
        payload.insert(payload.end(), p, p + n);
    };
    for (int b = 0; b < NUM_BLOCKS; ++b) append(w.key_blocks[b].data(), kb_bytes);
    for (int ply = 0; ply < NUM_PLIES; ++ply) append(w.nonkey_w[ply].data(), nk_bytes);
    for (int ply = 0; ply < NUM_PLIES; ++ply) {
        append(w.screlu[ply].mu,   12 * sizeof(int16_t));
        append(w.screlu[ply].beta, 12 * sizeof(int16_t));
    }
    for (int ply = 0; ply < NUM_PLIES; ++ply) {
        append(w.macro[ply].t_cw, 512 * sizeof(int16_t));
        append(w.macro[ply].t_ea, 512 * sizeof(int16_t));
        append(w.macro[ply].t_cd, 512 * sizeof(int16_t));
    }

    P2CFileHeader hdr{};
    std::memcpy(hdr.magic, P2C_MAGIC, 8);
    hdr.version            = P2C_FORMAT_VERSION;
    hdr.eval_mode          = static_cast<uint32_t>(mode);
    hdr.score_scale        = score_scale;
    hdr.flags              = flags;
    hdr.key_raw_entries    = static_cast<uint32_t>(pt.key_raw_entries);
    hdr.nonkey_raw_entries = static_cast<uint32_t>(pt.nonkey_raw_entries);
    hdr.total_raw_entries  = static_cast<uint32_t>(pt.total_raw_entries);
    hdr.num_plies          = static_cast<uint32_t>(NUM_PLIES);
    hdr.num_blocks         = static_cast<uint32_t>(NUM_BLOCKS);
    hdr.reserved           = 0U;
    hdr.payload_bytes      = static_cast<uint64_t>(payload.size());
    hdr.checksum_fnv1a     = fnv1a_64_update(fnv1a_64_init(), payload.data(), payload.size());

    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) return false;
    ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    ofs.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    return static_cast<bool>(ofs);
}

inline bool save_waca_g2_p2c(const std::string& path,
                             const WacaG2Weights& w,
                             float score_scale = P2C_DEFAULT_SCALE,
                             uint32_t flags = 1U) {
    const auto& pt = get_pattern_tables();
    const int H = std::clamp<int>(w.mlp_hidden, 1, WACA_MAX_HID);
    const size_t t1_bytes  = static_cast<size_t>(pt.total_raw_entries) * sizeof(int16_t);
    const size_t emb_bytes = static_cast<size_t>(pt.key_raw_entries) * sizeof(WacaKeyEmb4);

    std::vector<uint8_t> payload;
    payload.reserve(NUM_PLIES * t1_bytes + WACA_EO_BLOCKS * emb_bytes + 65536);

    auto append = [&](const void* src, size_t n) {
        const uint8_t* p = static_cast<const uint8_t*>(src);
        payload.insert(payload.end(), p, p + n);
    };

    // 1) Tier-1 untouched 64-pattern x 60-ply PTM weights (46.92 MiB)
    for (int ply = 0; ply < NUM_PLIES; ++ply) append(w.tier1.w_ply[ply].data(), t1_bytes);
    // 2) Stage 2.5 linear weights & bias
    for (int ply = 0; ply < NUM_PLIES; ++ply) append(w.s25[ply].w, WACA_F25 * sizeof(int16_t));
    for (int ply = 0; ply < NUM_PLIES; ++ply) append(&w.s25[ply].b, sizeof(int32_t));
    // 3) Stage 3 physically separated 20-block EO Key-14 embeddings (7.01 MiB)
    for (int b = 0; b < WACA_EO_BLOCKS; ++b) append(w.h_emb[b].data(), emb_bytes);
    // 4) Stage 3 60-ply Key-14 SCReLU head (mu, beta)
    for (int ply = 0; ply < NUM_PLIES; ++ply) {
        append(w.screlu[ply].mu,   12 * sizeof(int16_t));
        append(w.screlu[ply].beta, 12 * sizeof(int16_t));
    }
    // 5) Stage 3 60-ply 24->H->1 integer SCReLU MLP
    for (int ply = 0; ply < NUM_PLIES; ++ply) {
        for (int j = 0; j < H; ++j) append(w.mlp[ply].w1[j], WACA_F3 * sizeof(int8_t));
    }
    for (int ply = 0; ply < NUM_PLIES; ++ply) append(w.mlp[ply].b1, static_cast<size_t>(H) * sizeof(int32_t));
    for (int ply = 0; ply < NUM_PLIES; ++ply) append(w.mlp[ply].w2, static_cast<size_t>(H) * sizeof(int16_t));
    for (int ply = 0; ply < NUM_PLIES; ++ply) append(&w.mlp[ply].b2, sizeof(int32_t));

    P2CFileHeader hdr{};
    std::memcpy(hdr.magic, P2C_MAGIC_V3, 8);
    hdr.version            = P2C_FORMAT_VERSION_V3;
    hdr.eval_mode          = static_cast<uint32_t>(EvalMode::WACA_G2);
    hdr.score_scale        = score_scale;
    hdr.flags              = flags;
    hdr.key_raw_entries    = static_cast<uint32_t>(pt.key_raw_entries);
    hdr.nonkey_raw_entries = static_cast<uint32_t>(pt.nonkey_raw_entries);
    hdr.total_raw_entries  = static_cast<uint32_t>(pt.total_raw_entries);
    hdr.num_plies          = static_cast<uint32_t>(NUM_PLIES);
    hdr.num_blocks         = static_cast<uint32_t>(WACA_EO_BLOCKS);
    hdr.reserved           = static_cast<uint32_t>(H);
    hdr.payload_bytes      = static_cast<uint64_t>(payload.size());
    hdr.checksum_fnv1a     = fnv1a_64_update(fnv1a_64_init(), payload.data(), payload.size());

    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) return false;
    ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    ofs.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    return static_cast<bool>(ofs);
}

inline bool save_waca_g2_v31_p2c(const std::string& path,
                                 const WacaG2V31Weights& w,
                                 float score_scale = P2C_DEFAULT_SCALE,
                                 uint32_t flags = 1U) {
    const auto& pt = get_pattern_tables();
    const int H = std::clamp<int>(w.mlp_hidden, 1, WACA_V31_MAX_HID);
    const size_t t1_bytes  = static_cast<size_t>(pt.total_raw_entries) * sizeof(int16_t);
    const size_t emb_bytes = static_cast<size_t>(pt.key_raw_entries) * sizeof(WacaKeyEmb4);

    std::vector<uint8_t> payload;
    payload.reserve(NUM_PLIES * t1_bytes + WACA_V31_EO_BLOCKS * emb_bytes + 81920);

    auto append = [&](const void* src, size_t n) {
        const uint8_t* p = static_cast<const uint8_t*>(src);
        payload.insert(payload.end(), p, p + n);
    };

    // 1) Tier-1 untouched 64-pattern x 60-ply PTM weights (46.92 MiB)
    for (int ply = 0; ply < NUM_PLIES; ++ply) append(w.tier1.w_ply[ply].data(), t1_bytes);
    // 2) Stage 2.5 slim 8-dim linear weights & bias
    for (int ply = 0; ply < NUM_PLIES; ++ply) append(w.s25[ply].w, WACA_V31_F25 * sizeof(int16_t));
    for (int ply = 0; ply < NUM_PLIES; ++ply) append(&w.s25[ply].b, sizeof(int32_t));
    // 3) Stage 3 physically separated 30-block EO Key-14 embeddings (10.51 MiB)
    for (int b = 0; b < WACA_V31_EO_BLOCKS; ++b) append(w.h_emb[b].data(), emb_bytes);
    // 4) Stage 3 60-ply Key-14 SCReLU head (mu, beta)
    for (int ply = 0; ply < NUM_PLIES; ++ply) {
        append(w.screlu[ply].mu,   12 * sizeof(int16_t));
        append(w.screlu[ply].beta, 12 * sizeof(int16_t));
    }
    // 5) Stage 3 60-ply 32->H->1 integer SCReLU MLP
    for (int ply = 0; ply < NUM_PLIES; ++ply) {
        for (int j = 0; j < H; ++j) append(w.mlp[ply].w1[j], WACA_V31_F3 * sizeof(int8_t));
    }
    for (int ply = 0; ply < NUM_PLIES; ++ply) append(w.mlp[ply].b1, static_cast<size_t>(H) * sizeof(int32_t));
    for (int ply = 0; ply < NUM_PLIES; ++ply) append(w.mlp[ply].w2, static_cast<size_t>(H) * sizeof(int16_t));
    for (int ply = 0; ply < NUM_PLIES; ++ply) append(&w.mlp[ply].b2, sizeof(int32_t));

    P2CFileHeader hdr{};
    std::memcpy(hdr.magic, P2C_MAGIC_V31, 8);
    hdr.version            = P2C_FORMAT_VERSION_V31;
    hdr.eval_mode          = static_cast<uint32_t>(EvalMode::WACA_G2_V31);
    hdr.score_scale        = score_scale;
    hdr.flags              = flags;
    hdr.key_raw_entries    = static_cast<uint32_t>(pt.key_raw_entries);
    hdr.nonkey_raw_entries = static_cast<uint32_t>(pt.nonkey_raw_entries);
    hdr.total_raw_entries  = static_cast<uint32_t>(pt.total_raw_entries);
    hdr.num_plies          = static_cast<uint32_t>(NUM_PLIES);
    hdr.num_blocks         = static_cast<uint32_t>(WACA_V31_EO_BLOCKS);
    hdr.reserved           = static_cast<uint32_t>(H);
    hdr.payload_bytes      = static_cast<uint64_t>(payload.size());
    hdr.checksum_fnv1a     = fnv1a_64_update(fnv1a_64_init(), payload.data(), payload.size());

    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) return false;
    ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    ofs.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    return static_cast<bool>(ofs);
}

inline bool save_waca_g2_v32_p2c(const std::string& path,
                                 const WacaG2V32Weights& w,
                                 float score_scale = P2C_DEFAULT_SCALE,
                                 uint32_t flags = 1U) {
    const auto& pt = get_pattern_tables();
    const int H = std::clamp<int>(w.mlp_hidden, 1, WACA_V32_MAX_HID);
    const size_t t1_bytes  = static_cast<size_t>(pt.total_raw_entries) * sizeof(int16_t);
    const size_t emb_bytes = static_cast<size_t>(pt.key_raw_entries) * sizeof(WacaKeyEmb4);

    std::vector<uint8_t> payload;
    payload.reserve(NUM_PLIES * t1_bytes + WACA_V32_EO_BLOCKS * emb_bytes + 81920);

    auto append = [&](const void* src, size_t n) {
        const uint8_t* p = static_cast<const uint8_t*>(src);
        payload.insert(payload.end(), p, p + n);
    };

    // 1) Tier-1 untouched 64-pattern x 60-ply PTM weights (46.92 MiB)
    for (int ply = 0; ply < NUM_PLIES; ++ply) append(w.tier1.w_ply[ply].data(), t1_bytes);
    // 2) Stage 2.5 slim 8-dim linear weights & bias
    for (int ply = 0; ply < NUM_PLIES; ++ply) append(w.s25[ply].w, WACA_V32_F25 * sizeof(int16_t));
    for (int ply = 0; ply < NUM_PLIES; ++ply) append(&w.s25[ply].b, sizeof(int32_t));
    // 3) Stage 3 physically separated 30-block EO Key-14 embeddings (10.51 MiB)
    for (int b = 0; b < WACA_V32_EO_BLOCKS; ++b) append(w.h_emb[b].data(), emb_bytes);
    // 4) Stage 3 60-ply Key-14 SCReLU head (mu, beta)
    for (int ply = 0; ply < NUM_PLIES; ++ply) {
        append(w.screlu[ply].mu,   12 * sizeof(int16_t));
        append(w.screlu[ply].beta, 12 * sizeof(int16_t));
    }
    // 5) Stage 3 60-ply 28->H->1 integer SCReLU MLP
    for (int ply = 0; ply < NUM_PLIES; ++ply) {
        for (int j = 0; j < H; ++j) append(w.mlp[ply].w1[j], WACA_V32_F3 * sizeof(int8_t));
    }
    for (int ply = 0; ply < NUM_PLIES; ++ply) append(w.mlp[ply].b1, static_cast<size_t>(H) * sizeof(int32_t));
    for (int ply = 0; ply < NUM_PLIES; ++ply) append(w.mlp[ply].w2, static_cast<size_t>(H) * sizeof(int16_t));
    for (int ply = 0; ply < NUM_PLIES; ++ply) append(&w.mlp[ply].b2, sizeof(int32_t));

    P2CFileHeader hdr{};
    std::memcpy(hdr.magic, P2C_MAGIC_V32, 8);
    hdr.version            = P2C_FORMAT_VERSION_V32;
    hdr.eval_mode          = static_cast<uint32_t>(EvalMode::WACA_G2_V32);
    hdr.score_scale        = score_scale;
    hdr.flags              = flags;
    hdr.key_raw_entries    = static_cast<uint32_t>(pt.key_raw_entries);
    hdr.nonkey_raw_entries = static_cast<uint32_t>(pt.nonkey_raw_entries);
    hdr.total_raw_entries  = static_cast<uint32_t>(pt.total_raw_entries);
    hdr.num_plies          = static_cast<uint32_t>(NUM_PLIES);
    hdr.num_blocks         = static_cast<uint32_t>(WACA_V32_EO_BLOCKS);
    hdr.reserved           = static_cast<uint32_t>(H);
    hdr.payload_bytes      = static_cast<uint64_t>(payload.size());
    hdr.checksum_fnv1a     = fnv1a_64_update(fnv1a_64_init(), payload.data(), payload.size());

    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) return false;
    ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    ofs.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    return static_cast<bool>(ofs);
}

inline bool load_p2c_model(const std::string& path, LoadedModel& out, std::string* err_msg = nullptr) {
    auto fail = [&](const std::string& msg) -> bool {
        if (err_msg) *err_msg = msg;
        return false;
    };
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return fail("Cannot open file: " + path);

    P2CFileHeader hdr{};
    if (!ifs.read(reinterpret_cast<char*>(&hdr), sizeof(hdr))) {
        return fail("Failed to read 64B header from: " + path);
    }

    const bool is_v2  = (std::memcmp(hdr.magic, P2C_MAGIC, 8) == 0 && hdr.version == P2C_FORMAT_VERSION);
    const bool is_v3  = (std::memcmp(hdr.magic, P2C_MAGIC_V3, 8) == 0 && hdr.version == P2C_FORMAT_VERSION_V3);
    const bool is_v31 = (std::memcmp(hdr.magic, P2C_MAGIC_V31, 8) == 0 && hdr.version == P2C_FORMAT_VERSION_V31);
    const bool is_v32 = (std::memcmp(hdr.magic, P2C_MAGIC_V32, 8) == 0 && hdr.version == P2C_FORMAT_VERSION_V32);
    if (!is_v2 && !is_v3 && !is_v31 && !is_v32) {
        return fail("Invalid magic/version (expected P2C64V20 v2, P2C64V30 v3, P2C64V31 v4, or P2C64V32 v5) in: " + path);
    }

    const auto& pt = get_pattern_tables();
    if (hdr.key_raw_entries    != static_cast<uint32_t>(pt.key_raw_entries) ||
        hdr.nonkey_raw_entries != static_cast<uint32_t>(pt.nonkey_raw_entries) ||
        hdr.total_raw_entries  != static_cast<uint32_t>(pt.total_raw_entries) ||
        hdr.num_plies          != static_cast<uint32_t>(NUM_PLIES)) {
        return fail("Header table dimensions mismatch P2C-64 spec");
    }

    if (is_v2) {
        if (hdr.eval_mode > static_cast<uint32_t>(EvalMode::CAND_C_PTM_HYBRID)) {
            return fail("Invalid v2.0 eval_mode in header");
        }
        if (hdr.num_blocks != static_cast<uint32_t>(NUM_BLOCKS)) {
            return fail("Header num_blocks mismatch P2C-64 v2.0 spec");
        }
    } else if (is_v3) {
        if (hdr.eval_mode != static_cast<uint32_t>(EvalMode::WACA_G2)) {
            return fail("Invalid v3.0 eval_mode in header (expected WACA_G2=6)");
        }
        if (hdr.num_blocks != static_cast<uint32_t>(WACA_EO_BLOCKS)) {
            return fail("Header num_blocks mismatch P2C-64 v3.0 spec (expected 20 EO blocks)");
        }
        if (hdr.reserved < 1U || hdr.reserved > static_cast<uint32_t>(WACA_MAX_HID)) {
            return fail("Invalid v3.0 mlp_hidden in header (expected 1..24)");
        }
    } else if (is_v31) {
        if (hdr.eval_mode != static_cast<uint32_t>(EvalMode::WACA_G2_V31)) {
            return fail("Invalid v3.1 eval_mode in header (expected WACA_G2_V31=7)");
        }
        if (hdr.num_blocks != static_cast<uint32_t>(WACA_V31_EO_BLOCKS)) {
            return fail("Header num_blocks mismatch P2C-64 v3.1 spec (expected 30 EO blocks)");
        }
        if (hdr.reserved < 1U || hdr.reserved > static_cast<uint32_t>(WACA_V31_MAX_HID)) {
            return fail("Invalid v3.1 mlp_hidden in header (expected 1..24)");
        }
    } else {
        if (hdr.eval_mode != static_cast<uint32_t>(EvalMode::WACA_G2_V32)) {
            return fail("Invalid v3.2 eval_mode in header (expected WACA_G2_V32=8)");
        }
        if (hdr.num_blocks != static_cast<uint32_t>(WACA_V32_EO_BLOCKS)) {
            return fail("Header num_blocks mismatch P2C-64 v3.2 spec (expected 30 EO blocks)");
        }
        if (hdr.reserved < 1U || hdr.reserved > static_cast<uint32_t>(WACA_V32_MAX_HID)) {
            return fail("Invalid v3.2 mlp_hidden in header (expected 1..24)");
        }
    }

    std::vector<uint8_t> payload(static_cast<size_t>(hdr.payload_bytes));
    if (!ifs.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(hdr.payload_bytes))) {
        return fail("Truncated payload in: " + path);
    }
    const uint64_t chk = fnv1a_64_update(fnv1a_64_init(), payload.data(), payload.size());
    if (chk != hdr.checksum_fnv1a) {
        return fail("FNV-1a checksum mismatch in: " + path);
    }

    out.mode        = static_cast<EvalMode>(hdr.eval_mode);
    out.score_scale = hdr.score_scale;
    out.flags       = hdr.flags;
    out.is_raw      = (out.mode == EvalMode::RAW_NTUPLE_64);
    out.is_waca     = (out.mode == EvalMode::WACA_G2);
    out.is_waca_v31 = (out.mode == EvalMode::WACA_G2_V31);
    out.is_waca_v32 = (out.mode == EvalMode::WACA_G2_V32);

    size_t off = 0;
    auto read_chunk = [&](void* dst, size_t n) -> bool {
        if (off + n > payload.size()) return false;
        std::memcpy(dst, payload.data() + off, n);
        off += n;
        return true;
    };

    if (out.is_raw) {
        const size_t w_bytes = static_cast<size_t>(pt.total_raw_entries) * sizeof(int16_t);
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(out.raw.w_ply[ply].data(), w_bytes)) return fail("RAW w_ply read error");
        }
    } else if (out.is_waca_v32) {
        out.waca_v32 = std::make_unique<WacaG2V32Weights>();
        auto& wv = *out.waca_v32;
        const int H = static_cast<int>(hdr.reserved);
        wv.mlp_hidden = H;
        const size_t t1_bytes  = static_cast<size_t>(pt.total_raw_entries) * sizeof(int16_t);
        const size_t emb_bytes = static_cast<size_t>(pt.key_raw_entries) * sizeof(WacaKeyEmb4);
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(wv.tier1.w_ply[ply].data(), t1_bytes)) return fail("WACA v3.2 tier1 read error");
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(wv.s25[ply].w, WACA_V32_F25 * sizeof(int16_t))) return fail("WACA v3.2 s25 w read error");
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(&wv.s25[ply].b, sizeof(int32_t))) return fail("WACA v3.2 s25 b read error");
        }
        for (int b = 0; b < WACA_V32_EO_BLOCKS; ++b) {
            if (!read_chunk(wv.h_emb[b].data(), emb_bytes)) return fail("WACA v3.2 h_emb read error");
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(wv.screlu[ply].mu,   12 * sizeof(int16_t))) return fail("WACA v3.2 screlu mu read error");
            if (!read_chunk(wv.screlu[ply].beta, 12 * sizeof(int16_t))) return fail("WACA v3.2 screlu beta read error");
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            wv.mlp[ply] = WacaV32MlpPly{};
            for (int j = 0; j < H; ++j) {
                if (!read_chunk(wv.mlp[ply].w1[j], WACA_V32_F3 * sizeof(int8_t))) return fail("WACA v3.2 mlp w1 read error");
            }
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(wv.mlp[ply].b1, static_cast<size_t>(H) * sizeof(int32_t))) return fail("WACA v3.2 mlp b1 read error");
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(wv.mlp[ply].w2, static_cast<size_t>(H) * sizeof(int16_t))) return fail("WACA v3.2 mlp w2 read error");
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(&wv.mlp[ply].b2, sizeof(int32_t))) return fail("WACA v3.2 mlp b2 read error");
        }
    } else if (out.is_waca_v31) {
        out.waca_v31 = std::make_unique<WacaG2V31Weights>();
        auto& wv = *out.waca_v31;
        const int H = static_cast<int>(hdr.reserved);
        wv.mlp_hidden = H;
        const size_t t1_bytes  = static_cast<size_t>(pt.total_raw_entries) * sizeof(int16_t);
        const size_t emb_bytes = static_cast<size_t>(pt.key_raw_entries) * sizeof(WacaKeyEmb4);
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(wv.tier1.w_ply[ply].data(), t1_bytes)) return fail("WACA v3.1 tier1 read error");
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(wv.s25[ply].w, WACA_V31_F25 * sizeof(int16_t))) return fail("WACA v3.1 s25 w read error");
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(&wv.s25[ply].b, sizeof(int32_t))) return fail("WACA v3.1 s25 b read error");
        }
        for (int b = 0; b < WACA_V31_EO_BLOCKS; ++b) {
            if (!read_chunk(wv.h_emb[b].data(), emb_bytes)) return fail("WACA v3.1 h_emb read error");
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(wv.screlu[ply].mu,   12 * sizeof(int16_t))) return fail("WACA v3.1 screlu mu read error");
            if (!read_chunk(wv.screlu[ply].beta, 12 * sizeof(int16_t))) return fail("WACA v3.1 screlu beta read error");
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            wv.mlp[ply] = WacaV31MlpPly{};
            for (int j = 0; j < H; ++j) {
                if (!read_chunk(wv.mlp[ply].w1[j], WACA_V31_F3 * sizeof(int8_t))) return fail("WACA v3.1 mlp w1 read error");
            }
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(wv.mlp[ply].b1, static_cast<size_t>(H) * sizeof(int32_t))) return fail("WACA v3.1 mlp b1 read error");
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(wv.mlp[ply].w2, static_cast<size_t>(H) * sizeof(int16_t))) return fail("WACA v3.1 mlp w2 read error");
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(&wv.mlp[ply].b2, sizeof(int32_t))) return fail("WACA v3.1 mlp b2 read error");
        }
    } else if (out.is_waca) {
        const int H = static_cast<int>(hdr.reserved);
        out.waca.mlp_hidden = H;
        const size_t t1_bytes  = static_cast<size_t>(pt.total_raw_entries) * sizeof(int16_t);
        const size_t emb_bytes = static_cast<size_t>(pt.key_raw_entries) * sizeof(WacaKeyEmb4);
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(out.waca.tier1.w_ply[ply].data(), t1_bytes)) return fail("WACA tier1 read error");
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(out.waca.s25[ply].w, WACA_F25 * sizeof(int16_t))) return fail("WACA s25 w read error");
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(&out.waca.s25[ply].b, sizeof(int32_t))) return fail("WACA s25 b read error");
        }
        for (int b = 0; b < WACA_EO_BLOCKS; ++b) {
            if (!read_chunk(out.waca.h_emb[b].data(), emb_bytes)) return fail("WACA h_emb read error");
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(out.waca.screlu[ply].mu,   12 * sizeof(int16_t))) return fail("WACA screlu mu read error");
            if (!read_chunk(out.waca.screlu[ply].beta, 12 * sizeof(int16_t))) return fail("WACA screlu beta read error");
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            out.waca.mlp[ply] = WacaMlpPly{};
            for (int j = 0; j < H; ++j) {
                if (!read_chunk(out.waca.mlp[ply].w1[j], WACA_F3 * sizeof(int8_t))) return fail("WACA mlp w1 read error");
            }
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(out.waca.mlp[ply].b1, static_cast<size_t>(H) * sizeof(int32_t))) return fail("WACA mlp b1 read error");
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(out.waca.mlp[ply].w2, static_cast<size_t>(H) * sizeof(int16_t))) return fail("WACA mlp w2 read error");
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(&out.waca.mlp[ply].b2, sizeof(int32_t))) return fail("WACA mlp b2 read error");
        }
    } else {
        const size_t kb_bytes = static_cast<size_t>(pt.key_raw_entries) * sizeof(BlockEntry16);
        const size_t nk_bytes = static_cast<size_t>(pt.nonkey_raw_entries) * sizeof(int16_t);
        for (int b = 0; b < NUM_BLOCKS; ++b) {
            if (!read_chunk(out.p2c.key_blocks[b].data(), kb_bytes)) return fail("P2C key_blocks read error");
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(out.p2c.nonkey_w[ply].data(), nk_bytes)) return fail("P2C nonkey_w read error");
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(out.p2c.screlu[ply].mu,   12 * sizeof(int16_t))) return fail("P2C screlu mu read error");
            if (!read_chunk(out.p2c.screlu[ply].beta, 12 * sizeof(int16_t))) return fail("P2C screlu beta read error");
        }
        for (int ply = 0; ply < NUM_PLIES; ++ply) {
            if (!read_chunk(out.p2c.macro[ply].t_cw, 512 * sizeof(int16_t))) return fail("P2C t_cw read error");
            if (!read_chunk(out.p2c.macro[ply].t_ea, 512 * sizeof(int16_t))) return fail("P2C t_ea read error");
            if (!read_chunk(out.p2c.macro[ply].t_cd, 512 * sizeof(int16_t))) return fail("P2C t_cd read error");
        }
    }
    if (off != payload.size()) {
        return fail("Trailing unread bytes in payload");
    }
    return true;
}

} // namespace p2c64
