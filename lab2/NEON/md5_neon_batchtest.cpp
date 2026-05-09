/**
 * md5_neon_advanced.cpp
 * 包含 2路、4路、8路 NEON SIMD 并行版本
 */

#include "md5_neon.h"
#include <arm_neon.h>
#include <cstring>
#include <iostream>
#include <string>

using namespace std;

// 外部声明原始预处理函数
extern Byte *StringProcess(string input, int *n_byte);

// ============================================================
// NEON 工具宏
// ============================================================
#define NEON_ROTL(v, n) \
    vorrq_u32(vshlq_n_u32((v), (n)), vshrq_n_u32((v), 32-(n)))

#define NEON_F(x,y,z) vorrq_u32(vandq_u32((x),(y)), vandq_u32(vmvnq_u32(x),(z)))
#define NEON_G(x,y,z) vorrq_u32(vandq_u32((x),(z)), vandq_u32((y),vmvnq_u32(z)))
#define NEON_H(x,y,z) veorq_u32(veorq_u32((x),(y)),(z))
#define NEON_I(x,y,z) veorq_u32((y), vorrq_u32((x),vmvnq_u32(z)))

#define NEON_FF(a,b,c,d,x,s,ac) { \
    (a) = vaddq_u32(vaddq_u32((a), NEON_F((b),(c),(d))), vaddq_u32((x), vdupq_n_u32(ac))); \
    (a) = NEON_ROTL((a), (s)); \
    (a) = vaddq_u32((a), (b)); \
}
#define NEON_GG(a,b,c,d,x,s,ac) { \
    (a) = vaddq_u32(vaddq_u32((a), NEON_G((b),(c),(d))), vaddq_u32((x), vdupq_n_u32(ac))); \
    (a) = NEON_ROTL((a), (s)); \
    (a) = vaddq_u32((a), (b)); \
}
#define NEON_HH(a,b,c,d,x,s,ac) { \
    (a) = vaddq_u32(vaddq_u32((a), NEON_H((b),(c),(d))), vaddq_u32((x), vdupq_n_u32(ac))); \
    (a) = NEON_ROTL((a), (s)); \
    (a) = vaddq_u32((a), (b)); \
}
#define NEON_II(a,b,c,d,x,s,ac) { \
    (a) = vaddq_u32(vaddq_u32((a), NEON_I((b),(c),(d))), vaddq_u32((x), vdupq_n_u32(ac))); \
    (a) = NEON_ROTL((a), (s)); \
    (a) = vaddq_u32((a), (b)); \
}

// 辅助函数：提取向量到状态数组并进行字节序转换
void ExtractAndSwap(uint32x4_t va, uint32x4_t vb, uint32x4_t vc, uint32x4_t vd, bit32 states[][4], int count) {
    uint32_t buf_a[4], buf_b[4], buf_c[4], buf_d[4];
    vst1q_u32(buf_a, va); vst1q_u32(buf_b, vb);
    vst1q_u32(buf_c, vc); vst1q_u32(buf_d, vd);
    for (int i = 0; i < count; i++) {
        uint32_t raw[4] = {buf_a[i], buf_b[i], buf_c[i], buf_d[i]};
        for (int j = 0; j < 4; j++) {
            uint32_t v = raw[j];
            states[i][j] = ((v & 0xff) << 24) | ((v & 0xff00) << 8) | ((v & 0xff0000) >> 8) | ((v & 0xff000000) >> 24);
        }
    }
}

// ============================================================
// 1. MD5Hash_SIMD2：一次处理 2 个口令（利用率测试）
// ============================================================
void MD5Hash_SIMD2(string inputs[2], bit32 states[2][4]) {
    Byte *msgs[2]; int lengths[2], max_blocks = 0;
    for (int i = 0; i < 2; i++) {
        msgs[i] = StringProcess(inputs[i], &lengths[i]);
        int nb = lengths[i] / 64; if (nb > max_blocks) max_blocks = nb;
    }
    for (int i = 0; i < 2; i++) {
        int nb = lengths[i] / 64;
        if (nb < max_blocks) {
            Byte *ext = new Byte[max_blocks * 64](); memcpy(ext, msgs[i], lengths[i]);
            delete[] msgs[i]; msgs[i] = ext;
        }
    }
    uint32x4_t va = vdupq_n_u32(0x67452301), vb = vdupq_n_u32(0xefcdab89);
    uint32x4_t vc = vdupq_n_u32(0x98badcfe), vd = vdupq_n_u32(0x10325476);
    for (int blk = 0; blk < max_blocks; blk++) {
        uint32x4_t vx[16];
        for (int j = 0; j < 16; j++) {
            uint32_t w[4] = {0, 0, 0, 0}; // 仅填充前两个通道
            for (int k = 0; k < 2; k++) {
                int off = blk * 64 + j * 4;
                w[k] = (uint32_t)msgs[k][off] | (uint32_t)msgs[k][off+1] << 8 | (uint32_t)msgs[k][off+2] << 16 | (uint32_t)msgs[k][off+3] << 24;
            }
            vx[j] = vld1q_u32(w);
        }
        uint32x4_t va0 = va, vb0 = vb, vc0 = vc, vd0 = vd;
        /* Round 1 */
        NEON_FF(va,vb,vc,vd,vx[0],7,0xd76aa478); NEON_FF(vd,va,vb,vc,vx[1],12,0xe8c7b756); NEON_FF(vc,vd,va,vb,vx[2],17,0x242070db); NEON_FF(vb,vc,vd,va,vx[3],22,0xc1bdceee);
        NEON_FF(va,vb,vc,vd,vx[4],7,0xf57c0faf); NEON_FF(vd,va,vb,vc,vx[5],12,0x4787c62a); NEON_FF(vc,vd,va,vb,vx[6],17,0xa8304613); NEON_FF(vb,vc,vd,va,vx[7],22,0xfd469501);
        NEON_FF(va,vb,vc,vd,vx[8],7,0x698098d8); NEON_FF(vd,va,vb,vc,vx[9],12,0x8b44f7af); NEON_FF(vc,vd,va,vb,vx[10],17,0xffff5bb1); NEON_FF(vb,vc,vd,va,vx[11],22,0x895cd7be);
        NEON_FF(va,vb,vc,vd,vx[12],7,0x6b901122); NEON_FF(vd,va,vb,vc,vx[13],12,0xfd987193); NEON_FF(vc,vd,va,vb,vx[14],17,0xa679438e); NEON_FF(vb,vc,vd,va,vx[15],22,0x49b40821);
        /* Round 2 */
        NEON_GG(va,vb,vc,vd,vx[1],5,0xf61e2562); NEON_GG(vd,va,vb,vc,vx[6],9,0xc040b340); NEON_GG(vc,vd,va,vb,vx[11],14,0x265e5a51); NEON_GG(vb,vc,vd,va,vx[0],20,0xe9b6c7aa);
        NEON_GG(va,vb,vc,vd,vx[5],5,0xd62f105d); NEON_GG(vd,va,vb,vc,vx[10],9,0x02441453); NEON_GG(vc,vd,va,vb,vx[15],14,0xd8a1e681); NEON_GG(vb,vc,vd,va,vx[4],20,0xe7d3fbc8);
        NEON_GG(va,vb,vc,vd,vx[9],5,0x21e1cde6); NEON_GG(vd,va,vb,vc,vx[14],9,0xc33707d6); NEON_GG(vc,vd,va,vb,vx[3],14,0xf4d50d87); NEON_GG(vb,vc,vd,va,vx[8],20,0x455a14ed);
        NEON_GG(va,vb,vc,vd,vx[13],5,0xa9e3e905); NEON_GG(vd,va,vb,vc,vx[2],9,0xfcefa3f8); NEON_GG(vc,vd,va,vb,vx[7],14,0x676f02d9); NEON_GG(vb,vc,vd,va,vx[12],20,0x8d2a4c8a);
        /* Round 3 */
        NEON_HH(va,vb,vc,vd,vx[5],4,0xfffa3942); NEON_HH(vd,va,vb,vc,vx[8],11,0x8771f681); NEON_HH(vc,vd,va,vb,vx[11],16,0x6d9d6122); NEON_HH(vb,vc,vd,va,vx[14],23,0xfde5380c);
        NEON_HH(va,vb,vc,vd,vx[1],4,0xa4beea44); NEON_HH(vd,va,vb,vc,vx[4],11,0x4bdecfa9); NEON_HH(vc,vd,va,vb,vx[7],16,0xf6bb4b60); NEON_HH(vb,vc,vd,va,vx[10],23,0xbebfbc70);
        NEON_HH(va,vb,vc,vd,vx[13],4,0x289b7ec6); NEON_HH(vd,va,vb,vc,vx[0],11,0xeaa127fa); NEON_HH(vc,vd,va,vb,vx[3],16,0xd4ef3085); NEON_HH(vb,vc,vd,va,vx[6],23,0x04881d05);
        NEON_HH(va,vb,vc,vd,vx[9],4,0xd9d4d039); NEON_HH(vd,va,vb,vc,vx[12],11,0xe6db99e5); NEON_HH(vc,vd,va,vb,vx[15],16,0x1fa27cf8); NEON_HH(vb,vc,vd,va,vx[2],23,0xc4ac5665);
        /* Round 4 */
        NEON_II(va,vb,vc,vd,vx[0],6,0xf4292244); NEON_II(vd,va,vb,vc,vx[7],10,0x432aff97); NEON_II(vc,vd,va,vb,vx[14],15,0xab9423a7); NEON_II(vb,vc,vd,va,vx[5],21,0xfc93a039);
        NEON_II(va,vb,vc,vd,vx[12],6,0x655b59c3); NEON_II(vd,va,vb,vc,vx[3],10,0x8f0ccc92); NEON_II(vc,vd,va,vb,vx[10],15,0xffeff47d); NEON_II(vb,vc,vd,va,vx[1],21,0x85845dd1);
        NEON_II(va,vb,vc,vd,vx[8],6,0x6fa87e4f); NEON_II(vd,va,vb,vc,vx[15],10,0xfe2ce6e0); NEON_II(vc,vd,va,vb,vx[6],15,0xa3014314); NEON_II(vb,vc,vd,va,vx[13],21,0x4e0811a1);
        NEON_II(va,vb,vc,vd,vx[4],6,0xf7537e82); NEON_II(vd,va,vb,vc,vx[11],10,0xbd3af235); NEON_II(vc,vd,va,vb,vx[2],15,0x2ad7d2bb); NEON_II(vb,vc,vd,va,vx[9],21,0xeb86d391);

        va = vaddq_u32(va, va0); vb = vaddq_u32(vb, vb0); vc = vaddq_u32(vc, vc0); vd = vaddq_u32(vd, vd0);
    }
    ExtractAndSwap(va, vb, vc, vd, states, 2);
    for (int i = 0; i < 2; i++) delete[] msgs[i];
}

// ============================================================
// 2. MD5Hash_SIMD4：一次处理 4 个口令（标准版本）
// ============================================================
void MD5Hash_SIMD4(string inputs[4], bit32 states[4][4]) {
    Byte *msgs[4]; int lengths[4], max_blocks = 0;
    for (int i = 0; i < 4; i++) {
        msgs[i] = StringProcess(inputs[i], &lengths[i]);
        int nb = lengths[i] / 64; if (nb > max_blocks) max_blocks = nb;
    }
    for (int i = 0; i < 4; i++) {
        int nb = lengths[i] / 64;
        if (nb < max_blocks) {
            Byte *ext = new Byte[max_blocks * 64](); memcpy(ext, msgs[i], lengths[i]);
            delete[] msgs[i]; msgs[i] = ext;
        }
    }
    uint32x4_t va = vdupq_n_u32(0x67452301), vb = vdupq_n_u32(0xefcdab89);
    uint32x4_t vc = vdupq_n_u32(0x98badcfe), vd = vdupq_n_u32(0x10325476);
    for (int blk = 0; blk < max_blocks; blk++) {
        uint32x4_t vx[16];
        for (int j = 0; j < 16; j++) {
            uint32_t w[4];
            for (int k = 0; k < 4; k++) {
                int off = blk * 64 + j * 4;
                w[k] = (uint32_t)msgs[k][off] | (uint32_t)msgs[k][off+1] << 8 | (uint32_t)msgs[k][off+2] << 16 | (uint32_t)msgs[k][off+3] << 24;
            }
            vx[j] = vld1q_u32(w);
        }
        uint32x4_t va0 = va, vb0 = vb, vc0 = vc, vd0 = vd;
        /* Round 1 */
        NEON_FF(va,vb,vc,vd,vx[0],7,0xd76aa478); NEON_FF(vd,va,vb,vc,vx[1],12,0xe8c7b756); NEON_FF(vc,vd,va,vb,vx[2],17,0x242070db); NEON_FF(vb,vc,vd,va,vx[3],22,0xc1bdceee);
        NEON_FF(va,vb,vc,vd,vx[4],7,0xf57c0faf); NEON_FF(vd,va,vb,vc,vx[5],12,0x4787c62a); NEON_FF(vc,vd,va,vb,vx[6],17,0xa8304613); NEON_FF(vb,vc,vd,va,vx[7],22,0xfd469501);
        NEON_FF(va,vb,vc,vd,vx[8],7,0x698098d8); NEON_FF(vd,va,vb,vc,vx[9],12,0x8b44f7af); NEON_FF(vc,vd,va,vb,vx[10],17,0xffff5bb1); NEON_FF(vb,vc,vd,va,vx[11],22,0x895cd7be);
        NEON_FF(va,vb,vc,vd,vx[12],7,0x6b901122); NEON_FF(vd,va,vb,vc,vx[13],12,0xfd987193); NEON_FF(vc,vd,va,vb,vx[14],17,0xa679438e); NEON_FF(vb,vc,vd,va,vx[15],22,0x49b40821);
        /* Round 2 */
        NEON_GG(va,vb,vc,vd,vx[1],5,0xf61e2562); NEON_GG(vd,va,vb,vc,vx[6],9,0xc040b340); NEON_GG(vc,vd,va,vb,vx[11],14,0x265e5a51); NEON_GG(vb,vc,vd,va,vx[0],20,0xe9b6c7aa);
        NEON_GG(va,vb,vc,vd,vx[5],5,0xd62f105d); NEON_GG(vd,va,vb,vc,vx[10],9,0x02441453); NEON_GG(vc,vd,va,vb,vx[15],14,0xd8a1e681); NEON_GG(vb,vc,vd,va,vx[4],20,0xe7d3fbc8);
        NEON_GG(va,vb,vc,vd,vx[9],5,0x21e1cde6); NEON_GG(vd,va,vb,vc,vx[14],9,0xc33707d6); NEON_GG(vc,vd,va,vb,vx[3],14,0xf4d50d87); NEON_GG(vb,vc,vd,va,vx[8],20,0x455a14ed);
        NEON_GG(va,vb,vc,vd,vx[13],5,0xa9e3e905); NEON_GG(vd,va,vb,vc,vx[2],9,0xfcefa3f8); NEON_GG(vc,vd,va,vb,vx[7],14,0x676f02d9); NEON_GG(vb,vc,vd,va,vx[12],20,0x8d2a4c8a);
        /* Round 3 */
        NEON_HH(va,vb,vc,vd,vx[5],4,0xfffa3942); NEON_HH(vd,va,vb,vc,vx[8],11,0x8771f681); NEON_HH(vc,vd,va,vb,vx[11],16,0x6d9d6122); NEON_HH(vb,vc,vd,va,vx[14],23,0xfde5380c);
        NEON_HH(va,vb,vc,vd,vx[1],4,0xa4beea44); NEON_HH(vd,va,vb,vc,vx[4],11,0x4bdecfa9); NEON_HH(vc,vd,va,vb,vx[7],16,0xf6bb4b60); NEON_HH(vb,vc,vd,va,vx[10],23,0xbebfbc70);
        NEON_HH(va,vb,vc,vd,vx[13],4,0x289b7ec6); NEON_HH(vd,va,vb,vc,vx[0],11,0xeaa127fa); NEON_HH(vc,vd,va,vb,vx[3],16,0xd4ef3085); NEON_HH(vb,vc,vd,va,vx[6],23,0x04881d05);
        NEON_HH(va,vb,vc,vd,vx[9],4,0xd9d4d039); NEON_HH(vd,va,vb,vc,vx[12],11,0xe6db99e5); NEON_HH(vc,vd,va,vb,vx[15],16,0x1fa27cf8); NEON_HH(vb,vc,vd,va,vx[2],23,0xc4ac5665);
        /* Round 4 */
        NEON_II(va,vb,vc,vd,vx[0],6,0xf4292244); NEON_II(vd,va,vb,vc,vx[7],10,0x432aff97); NEON_II(vc,vd,va,vb,vx[14],15,0xab9423a7); NEON_II(vb,vc,vd,va,vx[5],21,0xfc93a039);
        NEON_II(va,vb,vc,vd,vx[12],6,0x655b59c3); NEON_II(vd,va,vb,vc,vx[3],10,0x8f0ccc92); NEON_II(vc,vd,va,vb,vx[10],15,0xffeff47d); NEON_II(vb,vc,vd,va,vx[1],21,0x85845dd1);
        NEON_II(va,vb,vc,vd,vx[8],6,0x6fa87e4f); NEON_II(vd,va,vb,vc,vx[15],10,0xfe2ce6e0); NEON_II(vc,vd,va,vb,vx[6],15,0xa3014314); NEON_II(vb,vc,vd,va,vx[13],21,0x4e0811a1);
        NEON_II(va,vb,vc,vd,vx[4],6,0xf7537e82); NEON_II(vd,va,vb,vc,vx[11],10,0xbd3af235); NEON_II(vc,vd,va,vb,vx[2],15,0x2ad7d2bb); NEON_II(vb,vc,vd,va,vx[9],21,0xeb86d391);
        
        va = vaddq_u32(va, va0); vb = vaddq_u32(vb, vb0); vc = vaddq_u32(vc, vc0); vd = vaddq_u32(vd, vd0);
    }
    ExtractAndSwap(va, vb, vc, vd, states, 4);
    for (int i = 0; i < 4; i++) delete[] msgs[i];
}

// ============================================================
// 3. MD5Hash_SIMD8：一次处理 8 个口令（双倍循环展开）
// ============================================================
void MD5Hash_SIMD8(string inputs[8], bit32 states[8][4]) {
    Byte *msgs[8]; int lengths[8], max_blocks = 0;
    for (int i = 0; i < 8; i++) {
        msgs[i] = StringProcess(inputs[i], &lengths[i]);
        int nb = lengths[i] / 64; if (nb > max_blocks) max_blocks = nb;
    }
    for (int i = 0; i < 8; i++) {
        int nb = lengths[i] / 64;
        if (nb < max_blocks) {
            Byte *ext = new Byte[max_blocks * 64](); memcpy(ext, msgs[i], lengths[i]);
            delete[] msgs[i]; msgs[i] = ext;
        }
    }
    // 定义两组寄存器
    uint32x4_t va1 = vdupq_n_u32(0x67452301), vb1 = vdupq_n_u32(0xefcdab89), vc1 = vdupq_n_u32(0x98badcfe), vd1 = vdupq_n_u32(0x10325476);
    uint32x4_t va2 = vdupq_n_u32(0x67452301), vb2 = vdupq_n_u32(0xefcdab89), vc2 = vdupq_n_u32(0x98badcfe), vd2 = vdupq_n_u32(0x10325476);

    for (int blk = 0; blk < max_blocks; blk++) {
        uint32x4_t vx1[16], vx2[16];
        for (int j = 0; j < 16; j++) {
            uint32_t w1[4], w2[4];
            for (int k = 0; k < 4; k++) {
                int off = blk * 64 + j * 4;
                w1[k] = (uint32_t)msgs[k][off] | (uint32_t)msgs[k][off+1] << 8 | (uint32_t)msgs[k][off+2] << 16 | (uint32_t)msgs[k][off+3] << 24;
                w2[k] = (uint32_t)msgs[k+4][off] | (uint32_t)msgs[k+4][off+1] << 8 | (uint32_t)msgs[k+4][off+2] << 16 | (uint32_t)msgs[k+4][off+3] << 24;
            }
            vx1[j] = vld1q_u32(w1); vx2[j] = vld1q_u32(w2);
        }
        uint32x4_t va1_0 = va1, vb1_0 = vb1, vc1_0 = vc1, vd1_0 = vd1;
        uint32x4_t va2_0 = va2, vb2_0 = vb2, vc2_0 = vc2, vd2_0 = vd2;

        // 核心优化：手动展开轮函数，两组指令并行发射
        #define DUAL_FF(j, s, ac) { NEON_FF(va1,vb1,vc1,vd1, vx1[j], s, ac); NEON_FF(va2,vb2,vc2,vd2, vx2[j], s, ac); }
        #define DUAL_GG(j, s, ac) { NEON_GG(va1,vb1,vc1,vd1, vx1[j], s, ac); NEON_GG(va2,vb2,vc2,vd2, vx2[j], s, ac); }
        #define DUAL_HH(j, s, ac) { NEON_HH(va1,vb1,vc1,vd1, vx1[j], s, ac); NEON_HH(va2,vb2,vc2,vd2, vx2[j], s, ac); }
        #define DUAL_II(j, s, ac) { NEON_II(va1,vb1,vc1,vd1, vx1[j], s, ac); NEON_II(va2,vb2,vc2,vd2, vx2[j], s, ac); }

        /* Round 1 */
        DUAL_FF(0,7,0xd76aa478); DUAL_FF(1,12,0xe8c7b756); DUAL_FF(2,17,0x242070db); DUAL_FF(3,22,0xc1bdceee);
        DUAL_FF(4,7,0xf57c0faf); DUAL_FF(5,12,0x4787c62a); DUAL_FF(6,17,0xa8304613); DUAL_FF(7,22,0xfd469501);
        DUAL_FF(8,7,0x698098d8); DUAL_FF(9,12,0x8b44f7af); DUAL_FF(10,17,0xffff5bb1); DUAL_FF(11,22,0x895cd7be);
        DUAL_FF(12,7,0x6b901122); DUAL_FF(13,12,0xfd987193); DUAL_FF(14,17,0xa679438e); DUAL_FF(15,22,0x49b40821);
        /* Round 2 */
        DUAL_GG(1,5,0xf61e2562); DUAL_GG(6,9,0xc040b340); DUAL_GG(11,14,0x265e5a51); DUAL_GG(0,20,0xe9b6c7aa);
        DUAL_GG(5,5,0xd62f105d); DUAL_GG(10,9,0x02441453); DUAL_GG(15,14,0xd8a1e681); DUAL_GG(4,20,0xe7d3fbc8);
        DUAL_GG(9,5,0x21e1cde6); DUAL_GG(14,9,0xc33707d6); DUAL_GG(3,14,0xf4d50d87); DUAL_GG(8,20,0x455a14ed);
        DUAL_GG(13,5,0xa9e3e905); DUAL_GG(2,9,0xfcefa3f8); DUAL_GG(7,14,0x676f02d9); DUAL_GG(12,20,0x8d2a4c8a);
        /* Round 3 */
        DUAL_HH(5,4,0xfffa3942); DUAL_HH(8,11,0x8771f681); DUAL_HH(11,16,0x6d9d6122); DUAL_HH(14,23,0xfde5380c);
        DUAL_HH(1,4,0xa4beea44); DUAL_HH(4,11,0x4bdecfa9); DUAL_HH(7,16,0xf6bb4b60); DUAL_HH(10,23,0xbebfbc70);
        DUAL_HH(13,4,0x289b7ec6); DUAL_HH(0,11,0xeaa127fa); DUAL_HH(3,16,0xd4ef3085); DUAL_HH(6,23,0x04881d05);
        DUAL_HH(9,4,0xd9d4d039); DUAL_HH(12,11,0xe6db99e5); DUAL_HH(15,16,0x1fa27cf8); DUAL_HH(2,23,0xc4ac5665);
        /* Round 4 */
        DUAL_II(0,6,0xf4292244); DUAL_II(7,10,0x432aff97); DUAL_II(14,15,0xab9423a7); DUAL_II(5,21,0xfc93a039);
        DUAL_II(12,6,0x655b59c3); DUAL_II(3,10,0x8f0ccc92); DUAL_II(10,15,0xffeff47d); DUAL_II(1,21,0x85845dd1);
        DUAL_II(8,6,0x6fa87e4f); DUAL_II(15,10,0xfe2ce6e0); DUAL_II(6,15,0xa3014314); DUAL_II(13,21,0x4e0811a1);
        DUAL_II(4,6,0xf7537e82); DUAL_II(11,10,0xbd3af235); DUAL_II(2,15,0x2ad7d2bb); DUAL_II(9,21,0xeb86d391);

        va1 = vaddq_u32(va1, va1_0); vb1 = vaddq_u32(vb1, vb1_0); vc1 = vaddq_u32(vc1, vc1_0); vd1 = vaddq_u32(vd1, vd1_0);
        va2 = vaddq_u32(va2, va2_0); vb2 = vaddq_u32(vb2, vb2_0); vc2 = vaddq_u32(vc2, vc2_0); vd2 = vaddq_u32(vd2, vd2_0);
    }
    ExtractAndSwap(va1, vb1, vc1, vd1, states, 4);
    ExtractAndSwap(va2, vb2, vc2, vd2, &states[4], 4);
    for (int i = 0; i < 8; i++) delete[] msgs[i];
}