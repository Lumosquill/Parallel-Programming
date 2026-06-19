// ============================================================
// v2_gpu_batch/guessing_gpu.cu — 进阶1: 多PT批处理
// 基于 v1 + 林盛森/许洋 Block-Per-PT 映射 + 孙沐赟数据打包
// ============================================================

#include "PCFG.h"
#include <cuda_runtime.h>

#define CUDA_CHK(c) do { \
    cudaError_t e = (c); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

// ============================================================
// Kernel 1: 单PT (同v1, 用于小batch fallback)
// ============================================================
__global__ void gen_kernel_single(
    const char* d_vals, const int* d_off, const char* d_pref,
    int pref_len, char* d_out, int slot, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int vs = d_off[i], vl = d_off[i+1] - vs;
    char* o = d_out + i * slot;
    for (int j = 0; j < pref_len; j++) o[j] = d_pref[j];
    for (int j = 0; j < vl; j++) o[pref_len + j] = d_vals[vs + j];
    o[pref_len + vl] = '\0';
}

// ============================================================
// Kernel 2: 多PT批处理 — Block-Per-PT 映射 (许洋v3 / 林盛森)
// gridDim.x = num_pts, blockDim.x = GPU_THREADS
// 每 block 处理一个 PT, 线程 stride 遍历该 PT 的 values
// ============================================================
__global__ void gen_kernel_batch(
    const char* d_all_vals,     // 所有PT的values拼接
    const int*  d_val_off,      // 每个value的偏移 [total_vals+1]
    const char* d_prefixes,     // 所有PT前缀 (定长 GPU_SLOT_LEN)
    const int*  d_pref_lens,    // 每个前缀长度 [num_pts]
    const int*  d_pt_start,     // 每个PT在value数组中的起始 [num_pts]
    const int*  d_pt_count,     // 每个PT的value数量 [num_pts]
    const int*  d_pt_out_off,   // 每个PT在输出中的起始偏移 [num_pts]
    char*       d_output,       // 输出 (定长槽位)
    int*        d_out_lens,     // 每个输出的实际长度
    int         slot            // GPU_SLOT_LEN
) {
    int pt_id = blockIdx.x;
    int tid   = threadIdx.x;

    int v_start  = d_pt_start[pt_id];   // 此PT的第一个value索引
    int v_count  = d_pt_count[pt_id];   // 此PT的value数
    int out_base = d_pt_out_off[pt_id]; // 此PT在输出中的起始槽位
    int pref_len = d_pref_lens[pt_id];  // 此PT的前缀长度

    // 前缀基地址 (定长槽位)
    const char* pref = d_prefixes + pt_id * GPU_SLOT_LEN;

    // 线程 stride 遍历此PT的所有 values
    for (int vi = tid; vi < v_count; vi += blockDim.x) {
        int global_vi = v_start + vi;
        int vs = d_val_off[global_vi];
        int vl = d_val_off[global_vi + 1] - vs;
        int out_idx = out_base + vi;

        char* o = d_output + out_idx * slot;
        for (int j = 0; j < pref_len; j++) o[j] = pref[j];
        for (int j = 0; j < vl; j++) o[pref_len + j] = d_all_vals[vs + j];
        o[pref_len + vl] = '\0';
        d_out_lens[out_idx] = pref_len + vl;
    }
}

// ============================================================
// GPU 工作区: static grow-only (同v1, 加批处理数组)
// ============================================================
static char* d_all_vals   = nullptr;
static int*  d_val_off    = nullptr;
static char* d_prefixes   = nullptr;
static int*  d_pref_lens  = nullptr;
static int*  d_pt_start   = nullptr;
static int*  d_pt_count   = nullptr;
static int*  d_pt_out_off = nullptr;
static char* d_output     = nullptr;
static int*  d_out_lens   = nullptr;
static size_t cap_vals=0, cap_offs=0, cap_prefs=0, cap_pts=0, cap_out=0;

static void ensure_buf(size_t vb, int nv, size_t pb, int np, size_t ob, int no) {
    if (vb > cap_vals) {
        if (d_all_vals) cudaFree(d_all_vals);
        CUDA_CHK(cudaMalloc(&d_all_vals, vb)); cap_vals = vb;
    }
    if ((size_t)(nv+1) > cap_offs) {
        if (d_val_off) cudaFree(d_val_off);
        CUDA_CHK(cudaMalloc(&d_val_off, (nv+1)*sizeof(int))); cap_offs = nv+1;
    }
    if (pb > cap_prefs) {
        if (d_prefixes) cudaFree(d_prefixes);
        CUDA_CHK(cudaMalloc(&d_prefixes, pb)); cap_prefs = pb;
    }
    if ((size_t)np > cap_pts) {
        if (d_pref_lens)  cudaFree(d_pref_lens);
        if (d_pt_start)   cudaFree(d_pt_start);
        if (d_pt_count)   cudaFree(d_pt_count);
        if (d_pt_out_off) cudaFree(d_pt_out_off);
        CUDA_CHK(cudaMalloc(&d_pref_lens,  np*sizeof(int)));
        CUDA_CHK(cudaMalloc(&d_pt_start,   np*sizeof(int)));
        CUDA_CHK(cudaMalloc(&d_pt_count,   np*sizeof(int)));
        CUDA_CHK(cudaMalloc(&d_pt_out_off, np*sizeof(int)));
        cap_pts = np;
    }
    if (ob > cap_out) {
        if (d_output)   cudaFree(d_output);
        if (d_out_lens) cudaFree(d_out_lens);
        CUDA_CHK(cudaMalloc(&d_output,   ob));
        CUDA_CHK(cudaMalloc(&d_out_lens, no*sizeof(int)));
        cap_out = ob;
    }
}

static void free_buf() {
    if (d_all_vals)  { cudaFree(d_all_vals);  d_all_vals=nullptr;  cap_vals=0; }
    if (d_val_off)   { cudaFree(d_val_off);   d_val_off=nullptr;   cap_offs=0; }
    if (d_prefixes)  { cudaFree(d_prefixes);  d_prefixes=nullptr;  cap_prefs=0; }
    if (d_pref_lens) { cudaFree(d_pref_lens); d_pref_lens=nullptr; }
    if (d_pt_start)  { cudaFree(d_pt_start);  d_pt_start=nullptr;  }
    if (d_pt_count)  { cudaFree(d_pt_count);  d_pt_count=nullptr;  }
    if (d_pt_out_off){ cudaFree(d_pt_out_off);d_pt_out_off=nullptr;}
    if (d_output)    { cudaFree(d_output);    d_output=nullptr;    cap_out=0; }
    if (d_out_lens)  { cudaFree(d_out_lens);  d_out_lens=nullptr;  }
    cap_pts = 0;
}

// ============================================================
// CalProb / init / NewPTs / Generate / GenerateGPU (同v1, 省略重复)
// 完整实现见 v1, 此处仅保留 GPU batch 新增代码
// ============================================================
// [这些函数从 v1 原样复制, 篇幅原因此处略去, 实际文件会包含]
// ============================================================

void PriorityQueue::CalProb(PT &pt) {
    pt.prob = pt.preterm_prob; int index = 0;
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
        CalProb(pt); priority.emplace_back(pt);
    }
}

vector<PT> PT::NewPTs() {
    vector<PT> res;
    if (content.size() == 1) return res;
    int init_pivot = pivot;
    for (int i = pivot; i < (int)curr_indices.size() - 1; i++) {
        curr_indices[i]++;
        if (curr_indices[i] < max_indices[i]) { pivot = i; res.emplace_back(*this); }
        curr_indices[i]--;
    }
    pivot = init_pivot; return res;
}

static string build_prefix(const PT& pt, model& m) {
    string prefix; int seg_idx = 0;
    for (int idx : pt.curr_indices) {
        if (pt.content[seg_idx].type == 1)
            prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
        else if (pt.content[seg_idx].type == 2)
            prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
        else if (pt.content[seg_idx].type == 3)
            prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
        seg_idx++; if (seg_idx == (int)pt.content.size() - 1) break;
    }
    return prefix;
}

void PriorityQueue::Generate(PT pt) {
    CalProb(pt);
    string prefix; segment* last_seg; int num_vals;
    if (pt.content.size() == 1) {
        if (pt.content[0].type == 1)      last_seg = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2) last_seg = &m.digits[m.FindDigit(pt.content[0])];
        else                              last_seg = &m.symbols[m.FindSymbol(pt.content[0])];
        prefix = ""; num_vals = pt.max_indices[0];
    } else {
        prefix = build_prefix(pt, m);
        int last = pt.content.size() - 1;
        if (pt.content[last].type == 1)      last_seg = &m.letters[m.FindLetter(pt.content[last])];
        else if (pt.content[last].type == 2) last_seg = &m.digits[m.FindDigit(pt.content[last])];
        else                                 last_seg = &m.symbols[m.FindSymbol(pt.content[last])];
        num_vals = pt.max_indices[last];
    }
    for (int i = 0; i < num_vals; i++)
        guesses.emplace_back(prefix + last_seg->ordered_values[i]);
    total_guesses += num_vals;
}

void PriorityQueue::GenerateGPU(PT pt) {
    CalProb(pt);
    string prefix; segment* last_seg; int num_vals;
    if (pt.content.size() == 1) {
        if (pt.content[0].type == 1)      last_seg = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2) last_seg = &m.digits[m.FindDigit(pt.content[0])];
        else                              last_seg = &m.symbols[m.FindSymbol(pt.content[0])];
        prefix = ""; num_vals = pt.max_indices[0];
    } else {
        prefix = build_prefix(pt, m);
        int last = pt.content.size() - 1;
        if (pt.content[last].type == 1)      last_seg = &m.letters[m.FindLetter(pt.content[last])];
        else if (pt.content[last].type == 2) last_seg = &m.digits[m.FindDigit(pt.content[last])];
        else                                 last_seg = &m.symbols[m.FindSymbol(pt.content[last])];
        num_vals = pt.max_indices[last];
    }
    if (num_vals < GPU_THRESHOLD) {
        for (int i = 0; i < num_vals; i++)
            guesses.emplace_back(prefix + last_seg->ordered_values[i]);
        total_guesses += num_vals; return;
    }
    // 单PT GPU (同v1逻辑, 略)
    // 但如果被 PopNextBatch 调用, 小任务应走CPU
    for (int i = 0; i < num_vals; i++)
        guesses.emplace_back(prefix + last_seg->ordered_values[i]);
    total_guesses += num_vals;
}

// ============================================================
// ★ 进阶1核心: 多PT批处理
//   攒够 batch_size 个PT → 一次kernel全部处理
//   参考: 孙沐赟 PopNextParallel + 许洋 Block-PT映射
// ============================================================
void PriorityQueue::PopNextBatch(int batch_size) {
    if (priority.empty()) return;

    // ── 第1步: 收集 batch_size 个 PT ──
    int n = min(batch_size, (int)priority.size());
    vector<PT>  batch_pts;
    vector<string> prefixes;
    vector<segment*> last_segs;
    vector<int> num_vals_list;

    for (int i = 0; i < n; i++) {
        PT pt = priority[i];
        CalProb(pt);

        string prefix; segment* last_seg; int num_vals;
        if (pt.content.size() == 1) {
            if (pt.content[0].type == 1)      last_seg = &m.letters[m.FindLetter(pt.content[0])];
            else if (pt.content[0].type == 2) last_seg = &m.digits[m.FindDigit(pt.content[0])];
            else                              last_seg = &m.symbols[m.FindSymbol(pt.content[0])];
            prefix = ""; num_vals = pt.max_indices[0];
        } else {
            prefix = build_prefix(pt, m);
            int last = pt.content.size() - 1;
            if (pt.content[last].type == 1)      last_seg = &m.letters[m.FindLetter(pt.content[last])];
            else if (pt.content[last].type == 2) last_seg = &m.digits[m.FindDigit(pt.content[last])];
            else                                 last_seg = &m.symbols[m.FindSymbol(pt.content[last])];
            num_vals = pt.max_indices[last];
        }

        // 小任务: 立即 CPU 处理, 不占 GPU 批次
        if (num_vals < GPU_THRESHOLD) {
            for (int j = 0; j < num_vals; j++)
                guesses.emplace_back(prefix + last_seg->ordered_values[j]);
            total_guesses += num_vals;
            // 仍然收集 new PTs (在下面统一处理)
        } else {
            batch_pts.push_back(pt);
            prefixes.push_back(prefix);
            last_segs.push_back(last_seg);
            num_vals_list.push_back(num_vals);
        }
    }

    // ── 第2步: 扁平化所有 PT 的 values ──
    int total_vals = 0;
    for (int nv : num_vals_list) total_vals += nv;

    if (total_vals > 0) {
        vector<char> flat_vals;
        vector<int>  val_off(total_vals + 1);
        vector<char> flat_prefs(batch_pts.size() * GPU_SLOT_LEN, 0);
        vector<int>  pref_lens(batch_pts.size());
        vector<int>  pt_start(batch_pts.size());
        vector<int>  pt_count(batch_pts.size());
        vector<int>  pt_out_off(batch_pts.size());

        val_off[0] = 0;
        int vi = 0, max_slot = 0;
        for (int p = 0; p < (int)batch_pts.size(); p++) {
            pt_start[p] = vi;
            pt_count[p] = num_vals_list[p];
            pref_lens[p] = prefixes[p].size();
            // 定长前缀
            memcpy(&flat_prefs[p * GPU_SLOT_LEN], prefixes[p].data(), prefixes[p].size());

            int out0 = (p == 0) ? 0 : pt_out_off[p-1] + num_vals_list[p-1];
            pt_out_off[p] = out0;

            // 计算槽位宽度
            int slot = prefixes[p].size() + 1;
            for (int j = 0; j < num_vals_list[p]; j++) {
                int len = last_segs[p]->ordered_values[j].size();
                if (prefixes[p].size() + len + 1 > slot)
                    slot = prefixes[p].size() + len + 1;
            }
            if (slot > GPU_SLOT_LEN) slot = GPU_SLOT_LEN;
            if (slot > max_slot) max_slot = slot;

            for (int j = 0; j < num_vals_list[p]; j++) {
                const string& v = last_segs[p]->ordered_values[j];
                flat_vals.insert(flat_vals.end(), v.begin(), v.end());
                val_off[vi + 1] = val_off[vi] + v.size();
                vi++;
            }
        }

        // ── 第3步: GPU 分配 + 传输 + 启动 ──
        ensure_buf(flat_vals.size(), total_vals,
                   batch_pts.size() * GPU_SLOT_LEN, batch_pts.size(),
                   (size_t)total_vals * max_slot, total_vals);

        CUDA_CHK(cudaMemcpy(d_all_vals,   flat_vals.data(), flat_vals.size(), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(d_val_off,    val_off.data(),  (total_vals+1)*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(d_prefixes,   flat_prefs.data(), batch_pts.size()*GPU_SLOT_LEN, cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(d_pref_lens,  pref_lens.data(), batch_pts.size()*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(d_pt_start,   pt_start.data(),  batch_pts.size()*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(d_pt_count,   pt_count.data(),  batch_pts.size()*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(d_pt_out_off, pt_out_off.data(),batch_pts.size()*sizeof(int), cudaMemcpyHostToDevice));

        // ★ Block-Per-PT: grid = num_pts, block = GPU_THREADS
        gen_kernel_batch<<<batch_pts.size(), GPU_THREADS>>>(
            d_all_vals, d_val_off, d_prefixes, d_pref_lens,
            d_pt_start, d_pt_count, d_pt_out_off, d_output, d_out_lens, max_slot);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());

        // ── 第4步: D2H + 重建 guesses ──
        vector<char> h_out(total_vals * max_slot);
        vector<int>  h_lens(total_vals);
        CUDA_CHK(cudaMemcpy(h_out.data(), d_output,   total_vals * max_slot, cudaMemcpyDeviceToHost));
        CUDA_CHK(cudaMemcpy(h_lens.data(),d_out_lens, total_vals * sizeof(int), cudaMemcpyDeviceToHost));

        for (int i = 0; i < total_vals; i++) {
            int len = h_lens[i] > 0 ? h_lens[i] : strlen(&h_out[i * max_slot]);
            guesses.emplace_back(&h_out[i * max_slot], len);
        }
        total_guesses += total_vals;
    }

    // ── 第5步: 所有 PT 生成 NewPTs, 二分插入 ──
    for (int i = 0; i < n; i++) {
        vector<PT> new_pts = priority[i].NewPTs();
        for (PT& np : new_pts) {
            CalProb(np);
            auto it = lower_bound(priority.begin(), priority.end(), np,
                [](const PT& a, const PT& b) { return a.prob > b.prob; });
            priority.insert(it, np);
        }
    }
    priority.erase(priority.begin(), priority.begin() + n);
}

void PriorityQueue::PopNext() { PopNextBatch(1); }

void gpu_cleanup() { free_buf(); }
