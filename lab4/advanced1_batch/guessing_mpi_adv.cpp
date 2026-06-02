#include "PCFG.h"
#include <mpi.h>
using namespace std;

void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;
    for (int idx : pt.curr_indices)
    {
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

void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)      pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            else if (seg.type == 2) pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            else if (seg.type == 3) pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

vector<PT> PT::NewPTs()
{
    vector<PT> res;
    if (content.size() == 1) return res;
    int init_pivot = pivot;
    for (int i = pivot; i < (int)curr_indices.size() - 1; i++)
    {
        curr_indices[i]++;
        if (curr_indices[i] < max_indices[i]) { pivot = i; res.emplace_back(*this); }
        curr_indices[i]--;
    }
    pivot = init_pivot;
    return res;
}

// 一个进程完整生成一个 PT（不分割）
static void gen_full_pt(PriorityQueue *q, PT &pt, vector<string> &out)
{
    q->CalProb(pt);
    if (pt.content.size() == 1)
    {
        segment *a = nullptr;
        if (pt.content[0].type == 1)          a = &q->m.letters[q->m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)     a = &q->m.digits[q->m.FindDigit(pt.content[0])];
        else if (pt.content[0].type == 3)     a = &q->m.symbols[q->m.FindSymbol(pt.content[0])];
        for (int i = 0; i < (int)pt.max_indices[0]; i++)
            out.emplace_back(a->ordered_values[i]);
    }
    else
    {
        string prefix;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
                prefix += q->m.letters[q->m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2)
                prefix += q->m.digits[q->m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 3)
                prefix += q->m.symbols[q->m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx++;
            if (seg_idx == (int)pt.content.size() - 1) break;
        }
        segment *a = nullptr;
        int last = pt.content.size() - 1;
        if (pt.content[last].type == 1)          a = &q->m.letters[q->m.FindLetter(pt.content[last])];
        else if (pt.content[last].type == 2)     a = &q->m.digits[q->m.FindDigit(pt.content[last])];
        else if (pt.content[last].type == 3)     a = &q->m.symbols[q->m.FindSymbol(pt.content[last])];
        for (int i = 0; i < (int)pt.max_indices[last]; i++)
            out.emplace_back(prefix + a->ordered_values[i]);
    }
}

// ==== PopNextBatch_mpi：全进程同步，零 PT 分发通信 ====
void PriorityQueue::PopNextBatch_mpi(int batch_size)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int actual = min(batch_size, (int)priority.size());
    if (actual == 0) return;

    // 所有进程从自己本地队列取同一批 PT（队列同步，无需通信）
    vector<PT> batch(priority.begin(), priority.begin() + actual);

    // 按块状分 PT 给各进程（纯本地计算，各进程只生成自己那份）
    int base = actual / size, rem = actual % size;
    int start, end;
    if (rank < rem) { start = rank * (base + 1);      end = start + base + 1; }
    else           { start = rem * (base + 1) + (rank - rem) * base; end = start + base; }

    int local_cnt = 0;
    for (int i = start; i < end; i++)
    {
        vector<string> tmp;
        gen_full_pt(this, batch[i], tmp);
        guesses.insert(guesses.end(), tmp.begin(), tmp.end());
        local_cnt += (int)tmp.size();
    }

    // 唯一一次 MPI 通信
    int global_cnt = 0;
    MPI_Allreduce(&local_cnt, &global_cnt, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    total_guesses += global_cnt;

    // 全进程同步管理队列
    vector<PT> all_new;
    for (int i = 0; i < actual; i++)
    {
        vector<PT> np = priority[i].NewPTs();
        for (PT &p : np) { CalProb(p); all_new.push_back(p); }
    }
    priority.erase(priority.begin(), priority.begin() + actual);
    for (PT &p : all_new)
    {
        bool ins = false;
        for (auto it = priority.begin(); it != priority.end(); it++)
            if (p.prob > it->prob) { priority.emplace(it, p); ins = true; break; }
        if (!ins) priority.emplace_back(p);
    }
}
