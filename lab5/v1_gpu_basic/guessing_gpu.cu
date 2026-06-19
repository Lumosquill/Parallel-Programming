// ============================================================
// v1_gpu_basic/guessing_gpu.cu — 基础要求: 单PT GPU加速
// 基于 lab3/v6_openmp_final/guessing.cpp
// 将 Generate() 内层循环替换为 CUDA kernel
// ============================================================

#include "PCFG.h"
#include <cuda_runtime.h>

// ── CUDA 错误检查 ──
#define CUDA_CHK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

// ============================================================
// GPU Kernel: 每个线程一个 guess (prefix + value → output)
// 参考: 孙沐赟 generate_guesses_kernel + 许洋 fastGenerateKernel
// ============================================================
__global__ void gen_kernel(
    const char* d_all_vals,     // 所有 value 字符串拼接 (扁平化)
    const int*  d_offsets,      // 每个 value 在 d_all_vals 中的起始偏移 [num+1]
    const char* d_prefix,       // 前缀字符串
    int         prefix_len,     // 前缀长度
    char*       d_output,       // 输出: 固定槽位 d_output + idx * slot_len
    int         slot_len,       // 槽位宽度 (GPU_SLOT_LEN)
    int         num_vals        // value 总数
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vals) return;

    // 定位 value
    int v_start = d_offsets[idx];
    int v_len   = d_offsets[idx + 1] - v_start;

    // 定位输出槽位
    char* out = d_output + idx * slot_len;

    // 拷贝前缀
    for (int i = 0; i < prefix_len; i++) out[i] = d_prefix[i];
    // 拷贝 value
    for (int i = 0; i < v_len; i++) out[prefix_len + i] = d_all_vals[v_start + i];
    // 终止符
    out[prefix_len + v_len] = '\0';
}

// ============================================================
// GPU 工作区: static 局部变量, grow-only 重分配 (孙沐赟模式)
// ============================================================
static char*  d_all_vals  = nullptr;
static int*   d_offsets   = nullptr;
static char*  d_prefix    = nullptr;
static char*  d_output    = nullptr;
static size_t cap_vals    = 0;
static size_t cap_offsets = 0;
static size_t cap_prefix  = 0;
static size_t cap_output  = 0;

static void ensure_buf(size_t vals_bytes, int num_vals, size_t pref_bytes, size_t out_bytes) {
    if (vals_bytes > cap_vals) {
        if (d_all_vals) cudaFree(d_all_vals);
        CUDA_CHK(cudaMalloc(&d_all_vals, vals_bytes));
        cap_vals = vals_bytes;
    }
    if ((size_t)(num_vals + 1) > cap_offsets) {
        if (d_offsets) cudaFree(d_offsets);
        CUDA_CHK(cudaMalloc(&d_offsets, (num_vals + 1) * sizeof(int)));
        cap_offsets = num_vals + 1;
    }
    if (pref_bytes > cap_prefix) {
        if (d_prefix) cudaFree(d_prefix);
        CUDA_CHK(cudaMalloc(&d_prefix, pref_bytes + 1));
        cap_prefix = pref_bytes + 1;
    }
    if (out_bytes > cap_output) {
        if (d_output) cudaFree(d_output);
        CUDA_CHK(cudaMalloc(&d_output, out_bytes));
        cap_output = out_bytes;
    }
}

static void free_buf() {
    if (d_all_vals) { cudaFree(d_all_vals); d_all_vals = nullptr; cap_vals = 0; }
    if (d_offsets)  { cudaFree(d_offsets);  d_offsets  = nullptr; cap_offsets = 0; }
    if (d_prefix)   { cudaFree(d_prefix);   d_prefix   = nullptr; cap_prefix = 0; }
    if (d_output)   { cudaFree(d_output);   d_output   = nullptr; cap_output = 0; }
}

// ============================================================
// 原版: CalProb (一字不改)
// ============================================================
void PriorityQueue::CalProb(PT &pt) {
    pt.prob = pt.preterm_prob;
    int index = 0;
    for (int idx : pt.curr_indices) {
        if (pt.content[index].type == 1)
            pt.prob *= (float)m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx]
                     / m.letters[m.FindLetter(pt.content[index])].total_freq;
        else if (pt.content[index].type == 2)
            pt.prob *= (float)m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx]
                     / m.digits[m.FindDigit(pt.content[index])].total_freq;
        else if (pt.content[index].type == 3)
            pt.prob *= (float)m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx]
                     / m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        index++;
    }
}

// ============================================================
// 原版: init (一字不改)
// ============================================================
void PriorityQueue::init() {
    for (PT pt : m.ordered_pts) {
        for (segment seg : pt.content) {
            if (seg.type == 1)
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            else if (seg.type == 2)
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            else if (seg.type == 3)
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

// ============================================================
// 原版: PT::NewPTs (一字不改)
// ============================================================
vector<PT> PT::NewPTs() {
    vector<PT> res;
    if (content.size() == 1) return res;
    int init_pivot = pivot;
    for (int i = pivot; i < (int)curr_indices.size() - 1; i++) {
        curr_indices[i]++;
        if (curr_indices[i] < max_indices[i]) {
            pivot = i;
            res.emplace_back(*this);
        }
        curr_indices[i]--;
    }
    pivot = init_pivot;
    return res;
}

// ============================================================
// 辅助: 构建前缀 (原版逻辑)
// ============================================================
static string build_prefix(const PT& pt, model& m) {
    string prefix;
    int seg_idx = 0;
    for (int idx : pt.curr_indices) {
        if (pt.content[seg_idx].type == 1)
            prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
        else if (pt.content[seg_idx].type == 2)
            prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
        else if (pt.content[seg_idx].type == 3)
            prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
        seg_idx++;
        if (seg_idx == (int)pt.content.size() - 1) break;
    }
    return prefix;
}

// ============================================================
// 原版 CPU Generate (保留, 用于比对和小任务)
// ============================================================
void PriorityQueue::Generate(PT pt) {
    CalProb(pt);

    if (pt.content.size() == 1) {
        segment* a;
        if (pt.content[0].type == 1)      a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
        else if (pt.content[0].type == 3) a = &m.symbols[m.FindSymbol(pt.content[0])];

        int n = pt.max_indices[0];
        for (int i = 0; i < n; i++)
            guesses.emplace_back(a->ordered_values[i]);
        total_guesses += n;
    } else {
        string prefix = build_prefix(pt, m);
        int last = pt.content.size() - 1;
        segment* a;
        if (pt.content[last].type == 1)      a = &m.letters[m.FindLetter(pt.content[last])];
        else if (pt.content[last].type == 2) a = &m.digits[m.FindDigit(pt.content[last])];
        else if (pt.content[last].type == 3) a = &m.symbols[m.FindSymbol(pt.content[last])];

        int n = pt.max_indices[last];
        for (int i = 0; i < n; i++)
            guesses.emplace_back(prefix + a->ordered_values[i]);
        total_guesses += n;
    }
}

// ============================================================
// ★ GPU 加速版 Generate (基础要求核心)
//   将原版内层 for 循环替换为 GPU kernel
// ============================================================
void PriorityQueue::GenerateGPU(PT pt) {
    CalProb(pt);

    string prefix;
    segment* last_seg;
    int num_vals;

    if (pt.content.size() == 1) {
        if (pt.content[0].type == 1)      last_seg = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2) last_seg = &m.digits[m.FindDigit(pt.content[0])];
        else if (pt.content[0].type == 3) last_seg = &m.symbols[m.FindSymbol(pt.content[0])];
        prefix = "";
        num_vals = pt.max_indices[0];
    } else {
        prefix = build_prefix(pt, m);
        int last = pt.content.size() - 1;
        if (pt.content[last].type == 1)      last_seg = &m.letters[m.FindLetter(pt.content[last])];
        else if (pt.content[last].type == 2) last_seg = &m.digits[m.FindDigit(pt.content[last])];
        else if (pt.content[last].type == 3) last_seg = &m.symbols[m.FindSymbol(pt.content[last])];
        num_vals = pt.max_indices[last];
    }

    // ── 阈值判断: 小任务走 CPU, 大任务走 GPU ──
    if (num_vals < GPU_THRESHOLD) {
        for (int i = 0; i < num_vals; i++)
            guesses.emplace_back(prefix + last_seg->ordered_values[i]);
        total_guesses += num_vals;
        return;
    }

    // ── 字符串扁平化 (所有学长都用这个) ──
    vector<char> flat_vals;
    vector<int>  offsets(num_vals + 1);
    int max_val_len = 0;

    offsets[0] = 0;
    for (int i = 0; i < num_vals; i++) {
        const string& v = last_seg->ordered_values[i];
        int len = v.size();
        flat_vals.insert(flat_vals.end(), v.begin(), v.end());
        offsets[i + 1] = offsets[i] + len;
        if (len > max_val_len) max_val_len = len;
    }

    int slot_len = prefix.size() + max_val_len + 1;
    if (slot_len > GPU_SLOT_LEN) slot_len = GPU_SLOT_LEN;  // 截断保护

    // ── GPU 内存: 条件重分配 (孙沐赟 grow-only) ──
    ensure_buf(flat_vals.size(), num_vals, prefix.size(),
               (size_t)num_vals * slot_len);

    // ── H2D 传输 ──
    CUDA_CHK(cudaMemcpy(d_all_vals, flat_vals.data(), flat_vals.size(), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_offsets,  offsets.data(),  (num_vals+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_prefix,   prefix.data(),    prefix.size(), cudaMemcpyHostToDevice));

    // ── 启动 kernel ──
    int grid = (num_vals + GPU_THREADS - 1) / GPU_THREADS;
    gen_kernel<<<grid, GPU_THREADS>>>(d_all_vals, d_offsets, d_prefix, (int)prefix.size(),
                                       d_output, slot_len, num_vals);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    // ── D2H 传输 + 重建 guesses ──
    vector<char> h_out(num_vals * slot_len);
    CUDA_CHK(cudaMemcpy(h_out.data(), d_output, num_vals * slot_len, cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_vals; i++)
        guesses.emplace_back(&h_out[i * slot_len]);

    total_guesses += num_vals;
}

// ============================================================
// PopNext: 使用 GPU 加速版 (基础版)
// ============================================================
void PriorityQueue::PopNext() {
    GenerateGPU(priority.front());

    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts) {
        CalProb(pt);
        // 二分插入 (刘迪乘: 概率降序)
        auto it = lower_bound(priority.begin(), priority.end(), pt,
            [](const PT& a, const PT& b) { return a.prob > b.prob; });
        priority.insert(it, pt);
    }
    priority.erase(priority.begin());
}

// ============================================================
// GPU 资源清理 (main 结束时调用)
// ============================================================
void gpu_cleanup() {
    free_buf();
}
