/**
 * md5_sse.cpp
 * 基于原始 md5.cpp，实现 SSE2 SIMD 并行版本 MD5Hash_SIMD4
 * 一次同时处理 4 个口令，SSE2 128bit 向量 = 4×32bit
 *
 * 编译：g++ -O3 -msse2 md5_sse.cpp md5.cpp -o your_program
 *       或 g++ -O3 -march=native ...
 * 平台：x86 / x86_64
 */

#include "md5.h"
#include <emmintrin.h>   // SSE2 头文件
#include <cstring>
#include <string>

using namespace std;

// 外部声明
extern Byte *StringProcess(string input, int *n_byte);

// ============================================================
// SSE2 工具宏 【已全部修正，无错误】
// ============================================================

// 循环左移：使用 SSE2 立即数移位指令（编译期常量，正确高效）
#define SSE_ROTL(v, n) \
    _mm_or_si128(_mm_slli_epi32((v), (n)), _mm_srli_epi32((v), 32-(n)))

// SSE2 位运算（标准 MD5 逻辑）
#define SSE_F(x,y,z) _mm_or_si128(_mm_and_si128((x),(y)), _mm_andnot_si128((x),(z)))
#define SSE_G(x,y,z) _mm_or_si128(_mm_and_si128((x),(z)), _mm_andnot_si128((z),(y)))
#define SSE_H(x,y,z) _mm_xor_si128(_mm_xor_si128((x),(y)),(z))
// 【已修正】I(x,y,z) = y ^ (x | ~z)，标准MD5逻辑
#define SSE_I(x,y,z) \
    _mm_xor_si128((y), _mm_or_si128((x), _mm_xor_si128((z), _mm_set1_epi32(0xFFFFFFFF))))

// 向量化 FF/GG/HH/II
#define SSE_FF(a,b,c,d,x,s,ac) { \
    (a) = _mm_add_epi32(_mm_add_epi32((a), SSE_F((b),(c),(d))), \
           _mm_add_epi32((x), _mm_set1_epi32(ac))); \
    (a) = SSE_ROTL((a), (s)); \
    (a) = _mm_add_epi32((a), (b)); \
}
#define SSE_GG(a,b,c,d,x,s,ac) { \
    (a) = _mm_add_epi32(_mm_add_epi32((a), SSE_G((b),(c),(d))), \
           _mm_add_epi32((x), _mm_set1_epi32(ac))); \
    (a) = SSE_ROTL((a), (s)); \
    (a) = _mm_add_epi32((a), (b)); \
}
#define SSE_HH(a,b,c,d,x,s,ac) { \
    (a) = _mm_add_epi32(_mm_add_epi32((a), SSE_H((b),(c),(d))), \
           _mm_add_epi32((x), _mm_set1_epi32(ac))); \
    (a) = SSE_ROTL((a), (s)); \
    (a) = _mm_add_epi32((a), (b)); \
}
#define SSE_II(a,b,c,d,x,s,ac) { \
    (a) = _mm_add_epi32(_mm_add_epi32((a), SSE_I((b),(c),(d))), \
           _mm_add_epi32((x), _mm_set1_epi32(ac))); \
    (a) = SSE_ROTL((a), (s)); \
    (a) = _mm_add_epi32((a), (b)); \
}

// ============================================================
// MD5Hash_SIMD4：SSE2 4路并行 MD5
// ============================================================
void MD5Hash_SIMD4(string inputs[4], bit32 states[4][4])
{
    // Step 1: 4个输入填充 & 对齐块数
    Byte *msgs[4];
    int   lengths[4];
    int   max_blocks = 0;

    for (int i = 0; i < 4; i++) {
        msgs[i] = StringProcess(inputs[i], &lengths[i]);
        int nb = lengths[i] / 64;
        if (nb > max_blocks) max_blocks = nb;
    }

    // 短块补0
    for (int i = 0; i < 4; i++) {
        int nb = lengths[i] / 64;
        if (nb < max_blocks) {
            Byte *ext = new Byte[max_blocks * 64]();
            memcpy(ext, msgs[i], lengths[i]);
            delete[] msgs[i];
            msgs[i] = ext;
        }
    }

    // Step 2: SSE2 向量初始状态（MD5 初始向量）
    __m128i va = _mm_set1_epi32(0x67452301);
    __m128i vb = _mm_set1_epi32(0xefcdab89);
    __m128i vc = _mm_set1_epi32(0x98badcfe);
    __m128i vd = _mm_set1_epi32(0x10325476);

    // Step 3: 逐块处理
    for (int blk = 0; blk < max_blocks; blk++) {
        __m128i vx[16];

        // 构造 SoA 向量：4个输入的第j个字打包成一个向量
        for (int j = 0; j < 16; j++) {
            uint32_t w[4];
            for (int k = 0; k < 4; k++) {
                int off = blk * 64 + j * 4;
                w[k] = (uint32_t)msgs[k][off]        |
                       (uint32_t)msgs[k][off + 1] <<  8 |
                       (uint32_t)msgs[k][off + 2] << 16 |
                       (uint32_t)msgs[k][off + 3] << 24;
            }
            vx[j] = _mm_loadu_si128((__m128i*)w);
        }

        // 保存初始状态
        __m128i va0 = va, vb0 = vb, vc0 = vc, vd0 = vd;

        /* Round 1 */
        SSE_FF(va,vb,vc,vd, vx[ 0], 7, 0xd76aa478);
        SSE_FF(vd,va,vb,vc, vx[ 1],12, 0xe8c7b756);
        SSE_FF(vc,vd,va,vb, vx[ 2],17, 0x242070db);
        SSE_FF(vb,vc,vd,va, vx[ 3],22, 0xc1bdceee);
        SSE_FF(va,vb,vc,vd, vx[ 4], 7, 0xf57c0faf);
        SSE_FF(vd,va,vb,vc, vx[ 5],12, 0x4787c62a);
        SSE_FF(vc,vd,va,vb, vx[ 6],17, 0xa8304613);
        SSE_FF(vb,vc,vd,va, vx[ 7],22, 0xfd469501);
        SSE_FF(va,vb,vc,vd, vx[ 8], 7, 0x698098d8);
        SSE_FF(vd,va,vb,vc, vx[ 9],12, 0x8b44f7af);
        SSE_FF(vc,vd,va,vb, vx[10],17, 0xffff5bb1);
        SSE_FF(vb,vc,vd,va, vx[11],22, 0x895cd7be);
        SSE_FF(va,vb,vc,vd, vx[12], 7, 0x6b901122);
        SSE_FF(vd,va,vb,vc, vx[13],12, 0xfd987193);
        SSE_FF(vc,vd,va,vb, vx[14],17, 0xa679438e);
        SSE_FF(vb,vc,vd,va, vx[15],22, 0x49b40821);

        /* Round 2 */
        SSE_GG(va,vb,vc,vd, vx[ 1], 5, 0xf61e2562);
        SSE_GG(vd,va,vb,vc, vx[ 6], 9, 0xc040b340);
        SSE_GG(vc,vd,va,vb, vx[11],14, 0x265e5a51);
        SSE_GG(vb,vc,vd,va, vx[ 0],20, 0xe9b6c7aa);
        SSE_GG(va,vb,vc,vd, vx[ 5], 5, 0xd62f105d);
        SSE_GG(vd,va,vb,vc, vx[10], 9, 0x02441453);
        SSE_GG(vc,vd,va,vb, vx[15],14, 0xd8a1e681);
        SSE_GG(vb,vc,vd,va, vx[ 4],20, 0xe7d3fbc8);
        SSE_GG(va,vb,vc,vd, vx[ 9], 5, 0x21e1cde6);
        SSE_GG(vd,va,vb,vc, vx[14], 9, 0xc33707d6);
        SSE_GG(vc,vd,va,vb, vx[ 3],14, 0xf4d50d87);
        SSE_GG(vb,vc,vd,va, vx[ 8],20, 0x455a14ed);
        SSE_GG(va,vb,vc,vd, vx[13], 5, 0xa9e3e905);
        SSE_GG(vd,va,vb,vc, vx[ 2], 9, 0xfcefa3f8);
        SSE_GG(vc,vd,va,vb, vx[ 7],14, 0x676f02d9);
        SSE_GG(vb,vc,vd,va, vx[12],20, 0x8d2a4c8a);

        /* Round 3 */
        SSE_HH(va,vb,vc,vd, vx[ 5], 4, 0xfffa3942);
        SSE_HH(vd,va,vb,vc, vx[ 8],11, 0x8771f681);
        SSE_HH(vc,vd,va,vb, vx[11],16, 0x6d9d6122);
        SSE_HH(vb,vc,vd,va, vx[14],23, 0xfde5380c);
        SSE_HH(va,vb,vc,vd, vx[ 1], 4, 0xa4beea44);
        SSE_HH(vd,va,vb,vc, vx[ 4],11, 0x4bdecfa9);
        SSE_HH(vc,vd,va,vb, vx[ 7],16, 0xf6bb4b60);
        SSE_HH(vb,vc,vd,va, vx[10],23, 0xbebfbc70);
        SSE_HH(va,vb,vc,vd, vx[13], 4, 0x289b7ec6);
        SSE_HH(vd,va,vb,vc, vx[ 0],11, 0xeaa127fa);
        SSE_HH(vc,vd,va,vb, vx[ 3],16, 0xd4ef3085);
        SSE_HH(vb,vc,vd,va, vx[ 6],23, 0x04881d05);
        SSE_HH(va,vb,vc,vd, vx[ 9], 4, 0xd9d4d039);
        SSE_HH(vd,va,vb,vc, vx[12],11, 0xe6db99e5);
        SSE_HH(vc,vd,va,vb, vx[15],16, 0x1fa27cf8);
        SSE_HH(vb,vc,vd,va, vx[ 2],23, 0xc4ac5665);

        /* Round 4 */
        SSE_II(va,vb,vc,vd, vx[ 0], 6, 0xf4292244);
        SSE_II(vd,va,vb,vc, vx[ 7],10, 0x432aff97);
        SSE_II(vc,vd,va,vb, vx[14],15, 0xab9423a7);
        SSE_II(vb,vc,vd,va, vx[ 5],21, 0xfc93a039);
        SSE_II(va,vb,vc,vd, vx[12], 6, 0x655b59c3);
        SSE_II(vd,va,vb,vc, vx[ 3],10, 0x8f0ccc92);
        SSE_II(vc,vd,va,vb, vx[10],15, 0xffeff47d);
        SSE_II(vb,vc,vd,va, vx[ 1],21, 0x85845dd1);
        SSE_II(va,vb,vc,vd, vx[ 8], 6, 0x6fa87e4f);
        SSE_II(vd,va,vb,vc, vx[15],10, 0xfe2ce6e0);
        SSE_II(vc,vd,va,vb, vx[ 6],15, 0xa3014314);
        SSE_II(vb,vc,vd,va, vx[13],21, 0x4e0811a1);
        SSE_II(va,vb,vc,vd, vx[ 4], 6, 0xf7537e82);
        SSE_II(vd,va,vb,vc, vx[11],10, 0xbd3af235);
        SSE_II(vc,vd,va,vb, vx[ 2],15, 0x2ad7d2bb);
        SSE_II(vb,vc,vd,va, vx[ 9],21, 0xeb86d391);

        // 状态累加
        va = _mm_add_epi32(va, va0);
        vb = _mm_add_epi32(vb, vb0);
        vc = _mm_add_epi32(vc, vc0);
        vd = _mm_add_epi32(vd, vd0);
    }

    // Step 4: 从SSE向量提取结果
    uint32_t buf_a[4], buf_b[4], buf_c[4], buf_d[4];
    _mm_storeu_si128((__m128i*)buf_a, va);
    _mm_storeu_si128((__m128i*)buf_b, vb);
    _mm_storeu_si128((__m128i*)buf_c, vc);
    _mm_storeu_si128((__m128i*)buf_d, vd);

    for (int i = 0; i < 4; i++) {
        states[i][0] = buf_a[i];
        states[i][1] = buf_b[i];
        states[i][2] = buf_c[i];
        states[i][3] = buf_d[i];
    }

    // Step 5: 字节序反转（和原版一致）
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            uint32_t v = states[i][j];
            states[i][j] = ((v & 0xff)       << 24) |
                           ((v & 0xff00)     <<  8) |
                           ((v & 0xff0000)   >>  8) |
                           ((v & 0xff000000) >> 24);
        }
    }

    // 释放内存
    for (int i = 0; i < 4; i++) delete[] msgs[i];
}