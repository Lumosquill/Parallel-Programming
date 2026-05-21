#pragma once
#include "md5.h"  // 复用原始的 Byte, bit32, StringProcess

// 一次并行处理 4 个口令的 MD5（NEON 128位，4×32bit 并行）
// 输入：inputs[4]    — 4个字符串
// 输出：states[4][4] — states[i][0..3] 是第 i 个口令的 MD5 结果
//                      格式与原始 MD5Hash 的 state[] 完全一致
void MD5Hash_SIMD4(std::string inputs[4], bit32 states[4][4]);