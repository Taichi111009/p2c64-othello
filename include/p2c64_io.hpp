#pragma once

#include "p2c64_eval.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace p2c64 {

// ============================================================================
// P2C-64 v2.0 Binary Weight File Format (.p2c)
//   - Fixed 64-byte little-endian header with magic="P2C64V20", version=2, FNV-1a checksum
//   - Full PTM layout:
//       RAW_NTUPLE_64 : 46.92 MiB (60 plies x 410,022 x int16_t)
//       P2C-64 v2.0   : 50.61 MiB (Key-14 14.02 MiB + NonKey-50 36.41 MiB
//                                  + 60-ply SCReLU 2.81 KiB + 60-ply Macro-LUT 180.0 KiB)
// ============================================================================
static constexpr char     P2C_MAGIC[8]       = {'P','2','C','6','4','V','2','0'};
static constexpr uint32_t P2C_FORMAT_VERSION = 2U;
static constexpr float    P2C_DEFAULT_SCALE  = 128.0f;

#pragma pack(push, 1)
struct P2CFileHeader {
    char     magic[8];
    uint32_t version;            // 2
    uint32_t eval_mode;          // EvalMode enum value [0..5]
    float    score_scale;        // Integer eval units per 1.0 disc
    uint32_t flags;              // Bit 0 = D4 symmetrized
    uint32_t key_raw_entries;    // 91,854
    uint32_t nonkey_raw_entries; // 318,168
    uint32_t total_raw_entries;  // 410,022
    uint32_t num_plies;          // 60
    uint32_t num_blocks;         // 10
    uint32_t reserved;           // 0
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
    }
    return "UNKNOWN";
}

struct LoadedModel {
    EvalMode            mode        = EvalMode::LINEAR_64;
    float               score_scale = P2C_DEFAULT_SCALE;
    uint32_t            flags       = 0;
    RawNTuple64Weights  raw;
    P2C64Weights        p2c;
    bool                is_raw      = false;

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
        }
        return 0;
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
    if (std::memcmp(hdr.magic, P2C_MAGIC, 8) != 0) {
        return fail("Invalid magic bytes (expected P2C64V20) in: " + path);
    }
    if (hdr.version != P2C_FORMAT_VERSION) {
        return fail("Unsupported .p2c version (expected 2) in: " + path);
    }
    if (hdr.eval_mode > static_cast<uint32_t>(EvalMode::CAND_C_PTM_HYBRID)) {
        return fail("Invalid eval_mode in header");
    }
    const auto& pt = get_pattern_tables();
    if (hdr.key_raw_entries    != static_cast<uint32_t>(pt.key_raw_entries) ||
        hdr.nonkey_raw_entries != static_cast<uint32_t>(pt.nonkey_raw_entries) ||
        hdr.total_raw_entries  != static_cast<uint32_t>(pt.total_raw_entries) ||
        hdr.num_plies          != static_cast<uint32_t>(NUM_PLIES) ||
        hdr.num_blocks         != static_cast<uint32_t>(NUM_BLOCKS)) {
        return fail("Header table dimensions mismatch P2C-64 v2.0 spec");
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
