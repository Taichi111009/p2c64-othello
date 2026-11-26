#pragma once

#include <cstdint>
#include <cstddef>

namespace p2c64 {

// ============================================================================
// Packed Training/Validation Sample (exact 20 bytes / position)
//   25.51M positions * 20 B = ~510.2 MB (fits easily in Colab / workstation RAM)
// ============================================================================
#pragma pack(push, 1)
struct PackedSample {
    uint64_t black;  // Bitboard of Black discs (sq = r*8 + c, a1 = 0, h8 = 63)
    uint64_t white;  // Bitboard of White discs
    int8_t   score;  // Ground-truth final disc difference from STM perspective [-64, +64]
    int8_t   stm;    // Side to move (+1 = Black, -1 = White)
    uint8_t  ply;    // Number of moves played: popcount(black | white) - 4 in [0, 59]
    uint8_t  flags;  // Bit 0: 1 if current STM has no legal move (pass position)
};
#pragma pack(pop)

static_assert(sizeof(PackedSample) == 20, "PackedSample must be strictly 20 bytes");

} // namespace p2c64
