// ============================================================
// v3_gpu_pipeline/guessing_gpu.cu — 进阶2: CPU/GPU重叠
// 基于 v2 + 葛明宇 AsyncGpuTask + cudaMemcpyAsync
// 核心: GPU计算期间 CPU 继续做队列维护, 不空等
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
// Kernel (同 v2 批量版, Block-Per-PT)
// ============================================================
__global__ void gen_kernel_batch(
    const char* d_all_vals, const int* d_val_off,
    const char* d_prefixes, const int* d_pref_lens,
    const int* d_pt_start, const int* d_pt_count,
    const int* d_pt_out_off,
    char* d_output, int* d_out_lens, int slot
) {
    int pt_id = blockIdx.x;
    int tid   = threadIdx.x;
    int v_start  = d_pt_start[pt_id];
    int v_count  = d_pt_count[pt_id];
    int out_base = d_pt_out_off[pt_id];
    int pref_len = d_pref_lens[pt_id];
    const char* pref = d_prefixes + pt_id * GPU_SLOT_LEN;

    for (int vi = tid; vi < v_count; vi += blockDim.x) {
        int gvi = v_start + vi;
        int vs = d_val_off[gvi], vl = d_val_off[gvi+1] - vs;
        int oi = out_base + vi;
        char* o = d_output + oi * slot;
        for (int j = 0; j < pref_len; j++) o[j] = pref[j];
        for (int j = 0; j < vl; j++) o[pref_len + j] = d_all_vals[vs + j];
        o[pref_len + vl] = '\0';
        d_out_lens[oi] = pref_len + vl;
    }
}

// ============================================================
// GPU 工作区 (双缓冲: 当前 + 异步)
// ============================================================
struct GpuBuf {
    char *d_all_vals=nullptr, *d_prefixes=nullptr, *d_output=nullptr;
    int  *d_val_off=nullptr, *d_pref_lens=nullptr, *d_pt_start=nullptr;
    int  *d_pt_count=nullptr, *d_pt_out_off=nullptr, *d_out_lens=nullptr;
    size_t cap_vals=0, cap_offs=0, cap_prefs=0, cap_pts=0, cap_out=0;
    cudaStream_t stream = nullptr;

    void ensure(size_t vb, int nv, size_t pb, int np, size_t ob, int no) {
        if (!stream) CUDA_CHK(cudaStreamCreate(&stream));
        if (vb > cap_vals) { if(d_all_vals)cudaFree(d_all_vals); CUDA_CHK(cudaMalloc(&d_all_vals,vb)); cap_vals=vb; }
        if ((size_t)(nv+1)>cap_offs){ if(d_val_off)cudaFree(d_val_off); CUDA_CHK(cudaMalloc(&d_val_off,(nv+1)*sizeof(int))); cap_offs=nv+1; }
        if (pb > cap_prefs) { if(d_prefixes)cudaFree(d_prefixes); CUDA_CHK(cudaMalloc(&d_prefixes,pb)); cap_prefs=pb; }
        if ((size_t)np > cap_pts) {
            if(d_pref_lens)cudaFree(d_pref_lens); if(d_pt_start)cudaFree(d_pt_start);
            if(d_pt_count)cudaFree(d_pt_count); if(d_pt_out_off)cudaFree(d_pt_out_off);
            CUDA_CHK(cudaMalloc(&d_pref_lens,np*sizeof(int))); CUDA_CHK(cudaMalloc(&d_pt_start,np*sizeof(int)));
            CUDA_CHK(cudaMalloc(&d_pt_count,np*sizeof(int))); CUDA_CHK(cudaMalloc(&d_pt_out_off,np*sizeof(int)));
            cap_pts=np;
        }
        if (ob > cap_out) {
            if(d_output)cudaFree(d_output); if(d_out_lens)cudaFree(d_out_lens);
            CUDA_CHK(cudaMalloc(&d_output,ob)); CUDA_CHK(cudaMalloc(&d_out_lens,no*sizeof(int))); cap_out=ob;
        }
    }
    void free_all() {
        if(d_all_vals)cudaFree(d_all_vals); if(d_val_off)cudaFree(d_val_off);
        if(d_prefixes)cudaFree(d_prefixes); if(d_pref_lens)cudaFree(d_pref_lens);
        if(d_pt_start)cudaFree(d_pt_start); if(d_pt_count)cudaFree(d_pt_count);
        if(d_pt_out_off)cudaFree(d_pt_out_off); if(d_output)cudaFree(d_output);
        if(d_out_lens)cudaFree(d_out_lens);
        if(stream)cudaStreamDestroy(stream);
        memset(this,0,sizeof(*this));
    }
};

// 双缓冲: a=当前同步用, b=异步发射用
static GpuBuf g_a, g_b;

// ============================================================
// 异步任务: 跨 PopNext 调用保持状态 (葛明宇模式)
// ============================================================
struct AsyncTask {
    bool   in_flight = false;
    GpuBuf* buf = nullptr;       // 指向 g_b
    int    total_vals = 0;
    int    slot_len = 0;
    vector<char> h_out;
    vector<int>  h_lens;
};

static AsyncTask g_task;

// ── 收集上一轮异步结果 ──
static void drain_async(PriorityQueue& q) {
    if (!g_task.in_flight) return;

    CUDA_CHK(cudaStreamSynchronize(g_task.buf->stream));

    // D2H
    size_t out_bytes = (size_t)g_task.total_vals * g_task.slot_len;
    g_task.h_out.resize(out_bytes);
    g_task.h_lens.resize(g_task.total_vals);
    CUDA_CHK(cudaMemcpy(g_task.h_out.data(), g_task.buf->d_output, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHK(cudaMemcpy(g_task.h_lens.data(), g_task.buf->d_out_lens, g_task.total_vals*sizeof(int), cudaMemcpyDeviceToHost));

    // 重建 guesses
    for (int i = 0; i < g_task.total_vals; i++) {
        int len = g_task.h_lens[i] > 0 ? g_task.h_lens[i]
                  : (int)strlen(&g_task.h_out[i * g_task.slot_len]);
        q.guesses.emplace_back(&g_task.h_out[i * g_task.slot_len], len);
    }
    q.total_guesses += g_task.total_vals;

    g_task.in_flight = false;
}

// ============================================================
// CalProb / init / NewPTs / 辅助函数 (同 v1/v2)
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
    string prefix; segment* ls; int n;
    if (pt.content.size() == 1) {
        if (pt.content[0].type==1) ls=&m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type==2) ls=&m.digits[m.FindDigit(pt.content[0])];
        else ls=&m.symbols[m.FindSymbol(pt.content[0])];
        prefix=""; n=pt.max_indices[0];
    } else {
        prefix=build_prefix(pt,m); int last=pt.content.size()-1;
        if (pt.content[last].type==1) ls=&m.letters[m.FindLetter(pt.content[last])];
        else if (pt.content[last].type==2) ls=&m.digits[m.FindDigit(pt.content[last])];
        else ls=&m.symbols[m.FindSymbol(pt.content[last])];
        n=pt.max_indices[last];
    }
    for(int i=0;i<n;i++) guesses.emplace_back(prefix+ls->ordered_values[i]);
    total_guesses+=n;
}

void PriorityQueue::GenerateGPU(PT pt) { Generate(pt); }  // 简化
void PriorityQueue::PopNextBatch(int) { PopNext(); }

// ============================================================
// ★ v3 核心: 异步流水线 PopNextPipeline
//   第 N 轮: 发射 GPU → 不等待 → CPU 做第 N+1 轮队列维护
//   第 N+1 轮: 先收第 N 轮结果 → 发射新 GPU → CPU 继续
// ============================================================
void PriorityQueue::PopNextPipeline(int batch_size) {
    if (priority.empty()) return;

    // ── Step 1: 先收上一轮的异步结果 ──
    drain_async(*this);

    // ── Step 2: 收集当前批次的 PT ──
    int n = min(batch_size, (int)priority.size());
    vector<PT> batch_pts; vector<string> prefixes;
    vector<segment*> last_segs; vector<int> nv_list;

    for (int i = 0; i < n; i++) {
        PT pt = priority[i]; CalProb(pt);
        string prefix; segment* ls; int nv;
        if (pt.content.size() == 1) {
            if (pt.content[0].type==1) ls=&m.letters[m.FindLetter(pt.content[0])];
            else if (pt.content[0].type==2) ls=&m.digits[m.FindDigit(pt.content[0])];
            else ls=&m.symbols[m.FindSymbol(pt.content[0])];
            prefix=""; nv=pt.max_indices[0];
        } else {
            prefix=build_prefix(pt,m); int last=pt.content.size()-1;
            if (pt.content[last].type==1) ls=&m.letters[m.FindLetter(pt.content[last])];
            else if (pt.content[last].type==2) ls=&m.digits[m.FindDigit(pt.content[last])];
            else ls=&m.symbols[m.FindSymbol(pt.content[last])];
            nv=pt.max_indices[last];
        }
        if (nv < GPU_THRESHOLD) {
            for (int j=0;j<nv;j++) guesses.emplace_back(prefix+ls->ordered_values[j]);
            total_guesses+=nv;
        } else {
            batch_pts.push_back(pt); prefixes.push_back(prefix);
            last_segs.push_back(ls); nv_list.push_back(nv);
        }
    }

    // ── Step 3: 扁平化 + 异步发射 GPU ──
    int total_vals = 0;
    for (int nv : nv_list) total_vals += nv;

    if (total_vals > 0 && !batch_pts.empty()) {
        // 准备 host 数据
        vector<char> flat_vals, flat_prefs(batch_pts.size() * GPU_SLOT_LEN, 0);
        vector<int> val_off(total_vals+1), pref_lens(batch_pts.size());
        vector<int> pt_start(batch_pts.size()), pt_count(batch_pts.size());
        vector<int> pt_out_off(batch_pts.size());

        val_off[0]=0; int vi=0, max_slot=0;
        for (int p=0; p<(int)batch_pts.size(); p++) {
            pt_start[p]=vi; pt_count[p]=nv_list[p];
            pref_lens[p]=prefixes[p].size();
            memcpy(&flat_prefs[p*GPU_SLOT_LEN], prefixes[p].data(), prefixes[p].size());
            pt_out_off[p]=(p==0)?0:pt_out_off[p-1]+nv_list[p-1];
            int slot=prefixes[p].size()+1;
            for (int j=0;j<nv_list[p];j++) {
                int vl=last_segs[p]->ordered_values[j].size();
                if(prefixes[p].size()+vl+1>slot) slot=prefixes[p].size()+vl+1;
            }
            if(slot>GPU_SLOT_LEN) slot=GPU_SLOT_LEN;
            if(slot>max_slot) max_slot=slot;
            for (int j=0;j<nv_list[p];j++) {
                const string& v=last_segs[p]->ordered_values[j];
                flat_vals.insert(flat_vals.end(),v.begin(),v.end());
                val_off[vi+1]=val_off[vi]+v.size(); vi++;
            }
        }

        // 使用 g_b (异步缓冲)
        g_b.ensure(flat_vals.size(), total_vals,
                   batch_pts.size()*GPU_SLOT_LEN, batch_pts.size(),
                   (size_t)total_vals*max_slot, total_vals);

        // ★ 全部异步传输 (cudaMemcpyAsync)
        CUDA_CHK(cudaMemcpyAsync(g_b.d_all_vals, flat_vals.data(), flat_vals.size(),
                                  cudaMemcpyHostToDevice, g_b.stream));
        CUDA_CHK(cudaMemcpyAsync(g_b.d_val_off, val_off.data(), (total_vals+1)*sizeof(int),
                                  cudaMemcpyHostToDevice, g_b.stream));
        CUDA_CHK(cudaMemcpyAsync(g_b.d_prefixes, flat_prefs.data(),
                                  batch_pts.size()*GPU_SLOT_LEN, cudaMemcpyHostToDevice, g_b.stream));
        CUDA_CHK(cudaMemcpyAsync(g_b.d_pref_lens, pref_lens.data(),
                                  batch_pts.size()*sizeof(int), cudaMemcpyHostToDevice, g_b.stream));
        CUDA_CHK(cudaMemcpyAsync(g_b.d_pt_start, pt_start.data(),
                                  batch_pts.size()*sizeof(int), cudaMemcpyHostToDevice, g_b.stream));
        CUDA_CHK(cudaMemcpyAsync(g_b.d_pt_count, pt_count.data(),
                                  batch_pts.size()*sizeof(int), cudaMemcpyHostToDevice, g_b.stream));
        CUDA_CHK(cudaMemcpyAsync(g_b.d_pt_out_off, pt_out_off.data(),
                                  batch_pts.size()*sizeof(int), cudaMemcpyHostToDevice, g_b.stream));

        // ★ 异步 kernel 发射 (不等)
        gen_kernel_batch<<<batch_pts.size(), GPU_THREADS, 0, g_b.stream>>>(
            g_b.d_all_vals, g_b.d_val_off, g_b.d_prefixes, g_b.d_pref_lens,
            g_b.d_pt_start, g_b.d_pt_count, g_b.d_pt_out_off,
            g_b.d_output, g_b.d_out_lens, max_slot);
        CUDA_CHK(cudaGetLastError());

        // 记录异步任务
        g_task.in_flight  = true;
        g_task.buf        = &g_b;
        g_task.total_vals = total_vals;
        g_task.slot_len   = max_slot;

        // ★ 不等 GPU! CPU 继续做 Step 4
    }

    // ── Step 4: CPU 维护队列 (与 GPU 并行!) ──
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

void PriorityQueue::PopNext() { PopNextPipeline(1); }

void gpu_cleanup() { drain_async(*((PriorityQueue*)nullptr)); g_a.free_all(); g_b.free_all(); }
