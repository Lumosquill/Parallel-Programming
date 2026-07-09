#include "PCFG.h"

// ==================== PriorityQueue::CalProb ====================
void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;
    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        else if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        else if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index++;
    }
}

// ==================== PriorityQueue::init ====================
void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
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
    // 按概率降序排列（确定性：等概率时按 preterminal 顺序，所有进程一致）
    sort(priority.begin(), priority.end(),
         [](const PT& a, const PT& b) { return a.prob > b.prob; });
}

// ==================== PT::NewPTs (不变) ====================
vector<PT> PT::NewPTs()
{
    vector<PT> res;
    if (content.size() == 1) return res;
    int init_pivot = pivot;
    for (int i = pivot; i < (int)curr_indices.size() - 1; i++)
    {
        curr_indices[i]++;
        if (curr_indices[i] < max_indices[i])
        {
            pivot = i;
            res.emplace_back(*this);
        }
        curr_indices[i]--;
    }
    pivot = init_pivot;
    return res;
}

// ==================== 串行 Generate（保留用于正确性对比） ====================
void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        segment *a;
        if (pt.content[0].type == 1)      a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
        else                              a = &m.symbols[m.FindSymbol(pt.content[0])];

        for (int i = 0; i < pt.max_indices[0]; i++)
            guesses.emplace_back(a->ordered_values[i]);
        total_guesses += pt.max_indices[0];
    }
    else
    {
        string prefix;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
                prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2)
                prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 3)
                prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx++;
            if (seg_idx == (int)pt.content.size() - 1) break;
        }

        segment *a;
        int last = pt.content.size() - 1;
        if (pt.content[last].type == 1)      a = &m.letters[m.FindLetter(pt.content[last])];
        else if (pt.content[last].type == 2) a = &m.digits[m.FindDigit(pt.content[last])];
        else                                 a = &m.symbols[m.FindSymbol(pt.content[last])];

        int total = pt.max_indices[last];
        for (int i = 0; i < total; i++)
            guesses.emplace_back(prefix + a->ordered_values[i]);
        total_guesses += total;
    }
}

// ==================== ★ MPI + OpenMP 双层混合 Generate ====================
// 三层并行架构：
//   Layer 1 (MPI):   进程间 block 分布，处理不同的 value 区间
//   Layer 2 (OpenMP): 进程内 parallel for，多线程填充
//   Layer 3 (SIMD):   在 main 中批量调用 MD5Hash_SIMD4
//
// 参考：
//   - 孙沐赟的 MPI 轮转分配 + time_guess/time_hash 分离
//   - 林盛森的 MPI+GPU 批处理架构
//   - 葛明宇的阈值自适应策略

void PriorityQueue::Generate_mpi_omp(PT pt)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    CalProb(pt);

    // ── 定位最后一个 segment 及其取值列表 ──
    segment *a = nullptr;
    string prefix;
    int total;

    if (pt.content.size() == 1)
    {
        if (pt.content[0].type == 1)      a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
        else                              a = &m.symbols[m.FindSymbol(pt.content[0])];
        prefix = "";
        total = pt.max_indices[0];
    }
    else
    {
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
                prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2)
                prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 3)
                prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx++;
            if (seg_idx == (int)pt.content.size() - 1) break;
        }
        int last = pt.content.size() - 1;
        if (pt.content[last].type == 1)      a = &m.letters[m.FindLetter(pt.content[last])];
        else if (pt.content[last].type == 2) a = &m.digits[m.FindDigit(pt.content[last])];
        else                                 a = &m.symbols[m.FindSymbol(pt.content[last])];
        total = pt.max_indices[last];
    }

    // ── Layer 1: MPI block 分布 ──
    int base = total / size;
    int rem  = total % size;
    int start_idx, end_idx;
    if (rank < rem) {
        start_idx = rank * (base + 1);
        end_idx   = start_idx + base + 1;
    } else {
        start_idx = rem * (base + 1) + (rank - rem) * base;
        end_idx   = start_idx + base;
    }
    int local_n = end_idx - start_idx;

    // ── Layer 2: OpenMP 进程内并行 ──
    // ★ local_n==0 也必须继续执行（参与 MPI_Allreduce），不能 return
    if (local_n > 0 && local_n < OMP_THRESHOLD)
    {
        // 小任务串行
        for (int i = start_idx; i < end_idx; i++)
            guesses.emplace_back(prefix + a->ordered_values[i]);
    }
    else if (local_n > 0)
    {
        // 大任务：预分配 + OpenMP parallel for（无锁设计）
        size_t old = guesses.size();
        guesses.resize(old + local_n);

        #pragma omp parallel for schedule(guided, OMP_CHUNK_SIZE)
        for (int i = 0; i < local_n; i++)
            guesses[old + i] = prefix + a->ordered_values[start_idx + i];
    }

    // ── MPI 汇总计数（★ 所有进程必须到达此处，否则死锁）──
    int global_total = 0;
    MPI_Allreduce(&local_n, &global_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    total_guesses = global_total;
}

// ==================== PopNext：排序 vector（与 Lab4 一致，保证确定性） ====================
void PriorityQueue::PopNext()
{
    // 取队首（概率最高的 PT）
    PT best_pt = priority.front();

    // MPI + OpenMP 双层 Generate
    Generate_mpi_omp(best_pt);

    // 生成新 PT
    vector<PT> new_pts = best_pt.NewPTs();
    for (PT &pt : new_pts)
    {
        CalProb(pt);
        // 线性扫描插入（保持按概率降序，确定性）
        bool inserted = false;
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                    { priority.emplace(iter + 1, pt); inserted = true; break; }
            }
            if (iter == priority.end() - 1) { priority.emplace_back(pt); inserted = true; break; }
            if (iter == priority.begin() && iter->prob < pt.prob)
                { priority.emplace(iter, pt); inserted = true; break; }
        }
        if (!inserted) priority.emplace_back(pt);
    }

    // 移除已处理的 PT
    priority.erase(priority.begin());
}
