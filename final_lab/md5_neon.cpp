/**
 * md5_neon.cpp
 *
 * 基于原始 md5.cpp，增加 NEON SIMD 并行版本 MD5Hash_SIMD4
 * 一次同时处理 4 个口令，每条 NEON 指令等价于 4 条串行指令
 *
 * 编译：g++ -O2 -march=armv8-a md5_neon.cpp md5.cpp -o your_program
 *        或 g++ -O2 -march=native ...
 */

#include "md5_neon.h"
#include <arm_neon.h>
#include <cstring>

using namespace std;

// StringProcess 在 md5.cpp 中已定义，直接用外部声明即可
extern Byte *StringProcess(string input, int *n_byte);

// ============================================================
// NEON 工具宏 
// C++语法：（带参数的）宏定义：#define + 宏名(参数列表) + 替换文本
// ============================================================

// 循环左移 Rotate Left
// NEON 没有直接的循环移位指令，用 左移/右移 组合实现
// vshlq_n_u32(v, n)：向量内每个元素逻辑左移 n 位（n 必须是编译期常量）
// vshrq_n_u32(v, n)：向量内每个元素逻辑右移 n 位
#define NEON_ROTL(v, n) \
    vorrq_u32(vshlq_n_u32((v), (n)), vshrq_n_u32((v), 32-(n)))

// 向量化 F/G/H/I
//V（Vector） + [操作缩写] + Q（Quadword）
// vandq_u32  : 按位 AND     （对应原始的 &）
// vorrq_u32  : 按位 OR      （对应原始的 |）
// veorq_u32  : 按位 XOR     （对应原始的 ^）异或 exclusive or
// vmvnq_u32  : 按位 NOT     （对应原始的 ~）取反 move not
#define NEON_F(x,y,z) vorrq_u32(vandq_u32((x),(y)), vandq_u32(vmvnq_u32(x),(z)))
#define NEON_G(x,y,z) vorrq_u32(vandq_u32((x),(z)), vandq_u32((y),vmvnq_u32(z)))
#define NEON_H(x,y,z) veorq_u32(veorq_u32((x),(y)),(z))
#define NEON_I(x,y,z) veorq_u32((y), vorrq_u32((x),vmvnq_u32(z)))

// 向量化 FF/GG/HH/II
// 对比原始串行宏：
//   FF(a,b,c,d,x,s,ac) { a += F(b,c,d) + x + ac; a = ROTL(a,s); a += b; }
// 区别只是把标量换成向量，操作符换成 NEON intrinsic
// vdupq_n_u32(ac)：将标量常数 ac 广播到向量的 4 个通道
// vaddq_u32：向量加法
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

// ============================================================
// MD5Hash_SIMD4：同时处理 4 个口令
// ============================================================
void MD5Hash_SIMD4(string inputs[4], bit32 states[4][4])
{
    // ----------------------------------------------------------
    // Step 1: 对 4 个口令分别做 padding（复用原始 StringProcess）
    // ----------------------------------------------------------
    Byte *msgs[4];
    int   lengths[4];
    int   max_blocks = 0;

    for (int i = 0; i < 4; i++) {
        msgs[i] = StringProcess(inputs[i], &lengths[i]);//填充
        int nb = lengths[i] / 64;//计算块数
        if (nb > max_blocks) max_blocks = nb;//找出最长的那个
    }

    // 若某口令的 block 数小于 max_blocks，扩展并给短的口令后面拼上全 0 的空块，
    //  使得 4 个口令的块数一致，方便 SIMD 处理
    for (int i = 0; i < 4; i++) {
        int nb = lengths[i] / 64;
        if (nb < max_blocks) {
            Byte *ext = new Byte[max_blocks * 64]();
            memcpy(ext, msgs[i], lengths[i]);
            delete[] msgs[i];
            msgs[i] = ext;
        }
    }

    // ----------------------------------------------------------
    // Step 2: 初始化 4 路向量状态
    // 每个 uint32x4_t 存 4 个口令的同一状态变量
    // va = [a_口令0, a_口令1, a_口令2, a_口令3]
    // vb = [b_口令0, b_口令1, b_口令2, b_口令3]  ...以此类推
    // vdupq_n_u32(x)：将标量 x 复制到向量的 4 个通道
    // ----------------------------------------------------------
    uint32x4_t va = vdupq_n_u32(0x67452301);
    uint32x4_t vb = vdupq_n_u32(0xefcdab89);
    uint32x4_t vc = vdupq_n_u32(0x98badcfe);
    uint32x4_t vd = vdupq_n_u32(0x10325476);

    // ----------------------------------------------------------
    // Step 3: 逐 block 处理（与串行版本结构完全对应）
    // ----------------------------------------------------------
    for (int blk = 0; blk < max_blocks; blk++) {

        // 构建 vx[16]：数据布局转换 AoS -> SoA
        //
        // 串行：每个口令有自己的 x[0..15]（16个 uint32_t）
        // SIMD：vx[j] 把 4 个口令的第 j 个字打包进一个向量
        //
        //   vx[0] = [x0_口令0, x0_口令1, x0_口令2, x0_口令3]
        //   vx[1] = [x1_口令0, x1_口令1, x1_口令2, x1_口令3]
        //   ...
        uint32x4_t vx[16];
        for (int j = 0; j < 16; j++) {// 遍历 MD5 块中的 16个 32位字
            uint32_t w[4];// 临时存放 4 个口令相同位置的数据
            for (int k = 0; k < 4; k++) {// 遍历 4 个口令
                int off = blk * 64 + j * 4;
                // 小端序组合：把 4 个 Byte 拼成 1 个 uint32
                w[k] = (uint32_t)msgs[k][off    ]        |
                       (uint32_t)msgs[k][off + 1] <<  8  |
                       (uint32_t)msgs[k][off + 2] << 16  |
                       (uint32_t)msgs[k][off + 3] << 24;
            }
            // vld1q_u32：从内存连续加载 4 个 uint32_t 到向量寄存器
            vx[j] = vld1q_u32(w);
        }

        // 保存本块开始时的状态，用于最后累加
        uint32x4_t va0 = va, vb0 = vb, vc0 = vc, vd0 = vd;

        /* Round 1 - FF */
        NEON_FF(va,vb,vc,vd, vx[ 0], s11, 0xd76aa478);
        NEON_FF(vd,va,vb,vc, vx[ 1], s12, 0xe8c7b756);
        NEON_FF(vc,vd,va,vb, vx[ 2], s13, 0x242070db);
        NEON_FF(vb,vc,vd,va, vx[ 3], s14, 0xc1bdceee);
        NEON_FF(va,vb,vc,vd, vx[ 4], s11, 0xf57c0faf);
        NEON_FF(vd,va,vb,vc, vx[ 5], s12, 0x4787c62a);
        NEON_FF(vc,vd,va,vb, vx[ 6], s13, 0xa8304613);
        NEON_FF(vb,vc,vd,va, vx[ 7], s14, 0xfd469501);
        NEON_FF(va,vb,vc,vd, vx[ 8], s11, 0x698098d8);
        NEON_FF(vd,va,vb,vc, vx[ 9], s12, 0x8b44f7af);
        NEON_FF(vc,vd,va,vb, vx[10], s13, 0xffff5bb1);
        NEON_FF(vb,vc,vd,va, vx[11], s14, 0x895cd7be);
        NEON_FF(va,vb,vc,vd, vx[12], s11, 0x6b901122);
        NEON_FF(vd,va,vb,vc, vx[13], s12, 0xfd987193);
        NEON_FF(vc,vd,va,vb, vx[14], s13, 0xa679438e);
        NEON_FF(vb,vc,vd,va, vx[15], s14, 0x49b40821);

        /* Round 2 - GG */
        NEON_GG(va,vb,vc,vd, vx[ 1], s21, 0xf61e2562);
        NEON_GG(vd,va,vb,vc, vx[ 6], s22, 0xc040b340);
        NEON_GG(vc,vd,va,vb, vx[11], s23, 0x265e5a51);
        NEON_GG(vb,vc,vd,va, vx[ 0], s24, 0xe9b6c7aa);
        NEON_GG(va,vb,vc,vd, vx[ 5], s21, 0xd62f105d);
        NEON_GG(vd,va,vb,vc, vx[10], s22, 0x02441453);
        NEON_GG(vc,vd,va,vb, vx[15], s23, 0xd8a1e681);
        NEON_GG(vb,vc,vd,va, vx[ 4], s24, 0xe7d3fbc8);
        NEON_GG(va,vb,vc,vd, vx[ 9], s21, 0x21e1cde6);
        NEON_GG(vd,va,vb,vc, vx[14], s22, 0xc33707d6);
        NEON_GG(vc,vd,va,vb, vx[ 3], s23, 0xf4d50d87);
        NEON_GG(vb,vc,vd,va, vx[ 8], s24, 0x455a14ed);
        NEON_GG(va,vb,vc,vd, vx[13], s21, 0xa9e3e905);
        NEON_GG(vd,va,vb,vc, vx[ 2], s22, 0xfcefa3f8);
        NEON_GG(vc,vd,va,vb, vx[ 7], s23, 0x676f02d9);
        NEON_GG(vb,vc,vd,va, vx[12], s24, 0x8d2a4c8a);

        /* Round 3 - HH */
        NEON_HH(va,vb,vc,vd, vx[ 5], s31, 0xfffa3942);
        NEON_HH(vd,va,vb,vc, vx[ 8], s32, 0x8771f681);
        NEON_HH(vc,vd,va,vb, vx[11], s33, 0x6d9d6122);
        NEON_HH(vb,vc,vd,va, vx[14], s34, 0xfde5380c);
        NEON_HH(va,vb,vc,vd, vx[ 1], s31, 0xa4beea44);
        NEON_HH(vd,va,vb,vc, vx[ 4], s32, 0x4bdecfa9);
        NEON_HH(vc,vd,va,vb, vx[ 7], s33, 0xf6bb4b60);
        NEON_HH(vb,vc,vd,va, vx[10], s34, 0xbebfbc70);
        NEON_HH(va,vb,vc,vd, vx[13], s31, 0x289b7ec6);
        NEON_HH(vd,va,vb,vc, vx[ 0], s32, 0xeaa127fa);
        NEON_HH(vc,vd,va,vb, vx[ 3], s33, 0xd4ef3085);
        NEON_HH(vb,vc,vd,va, vx[ 6], s34, 0x04881d05);
        NEON_HH(va,vb,vc,vd, vx[ 9], s31, 0xd9d4d039);
        NEON_HH(vd,va,vb,vc, vx[12], s32, 0xe6db99e5);
        NEON_HH(vc,vd,va,vb, vx[15], s33, 0x1fa27cf8);
        NEON_HH(vb,vc,vd,va, vx[ 2], s34, 0xc4ac5665);

        /* Round 4 - II */
        NEON_II(va,vb,vc,vd, vx[ 0], s41, 0xf4292244);
        NEON_II(vd,va,vb,vc, vx[ 7], s42, 0x432aff97);
        NEON_II(vc,vd,va,vb, vx[14], s43, 0xab9423a7);
        NEON_II(vb,vc,vd,va, vx[ 5], s44, 0xfc93a039);
        NEON_II(va,vb,vc,vd, vx[12], s41, 0x655b59c3);
        NEON_II(vd,va,vb,vc, vx[ 3], s42, 0x8f0ccc92);
        NEON_II(vc,vd,va,vb, vx[10], s43, 0xffeff47d);
        NEON_II(vb,vc,vd,va, vx[ 1], s44, 0x85845dd1);
        NEON_II(va,vb,vc,vd, vx[ 8], s41, 0x6fa87e4f);
        NEON_II(vd,va,vb,vc, vx[15], s42, 0xfe2ce6e0);
        NEON_II(vc,vd,va,vb, vx[ 6], s43, 0xa3014314);
        NEON_II(vb,vc,vd,va, vx[13], s44, 0x4e0811a1);
        NEON_II(va,vb,vc,vd, vx[ 4], s41, 0xf7537e82);
        NEON_II(vd,va,vb,vc, vx[11], s42, 0xbd3af235);
        NEON_II(vc,vd,va,vb, vx[ 2], s43, 0x2ad7d2bb);
        NEON_II(vb,vc,vd,va, vx[ 9], s44, 0xeb86d391);

        // 累加（与串行版本的 state[0]+=a 对应）
        va = vaddq_u32(va, va0);
        vb = vaddq_u32(vb, vb0);
        vc = vaddq_u32(vc, vc0);
        vd = vaddq_u32(vd, vd0);
    }

    // ----------------------------------------------------------
    // Step 4: 提取向量结果，写回 states[4][4]
    // vst1q_u32：将向量寄存器的 4 个 uint32_t 连续写到内存
    // ----------------------------------------------------------
    uint32_t buf_a[4], buf_b[4], buf_c[4], buf_d[4];
    vst1q_u32(buf_a, va);
    vst1q_u32(buf_b, vb);
    vst1q_u32(buf_c, vc);
    vst1q_u32(buf_d, vd);//

    for (int i = 0; i < 4; i++) {
        states[i][0] = buf_a[i];
        states[i][1] = buf_b[i];
        states[i][2] = buf_c[i];
        states[i][3] = buf_d[i];
    }

    // ----------------------------------------------------------
    // Step 5: 字节序转换（与原始 MD5Hash 末尾处理完全一致）
    // ----------------------------------------------------------
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            uint32_t v = states[i][j];
            states[i][j] = ((v & 0xff)      << 24) |
                           ((v & 0xff00)     <<  8) |
                           ((v & 0xff0000)   >>  8) |
                           ((v & 0xff000000) >> 24);
        }
    }

    for (int i = 0; i < 4; i++) delete[] msgs[i];
}