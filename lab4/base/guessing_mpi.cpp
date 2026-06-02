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
}

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

// ====== MPI 并行 Generate（块状负载均衡） ======
void PriorityQueue::Generate_mpi(PT pt)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    CalProb(pt);

    if (pt.content.size() == 1)
    {
        segment *a = nullptr;
        if (pt.content[0].type == 1)          a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)     a = &m.digits[m.FindDigit(pt.content[0])];
        else if (pt.content[0].type == 3)     a = &m.symbols[m.FindSymbol(pt.content[0])];

        int total = pt.max_indices[0];
        int base = total / size;
        int rem  = total % size;
        int start_idx, end_idx;
        if (rank < rem) { start_idx = rank * (base + 1);        end_idx = start_idx + base + 1; }
        else           { start_idx = rem * (base + 1) + (rank - rem) * base; end_idx = start_idx + base; }

        for (int i = start_idx; i < end_idx; i++)
            guesses.push_back(a->ordered_values[i]);

        int local_count = end_idx - start_idx;
        int global_total = 0;
        MPI_Allreduce(&local_count, &global_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        total_guesses = global_total;
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

        segment *a = nullptr;
        int last = pt.content.size() - 1;
        if (pt.content[last].type == 1)          a = &m.letters[m.FindLetter(pt.content[last])];
        else if (pt.content[last].type == 2)     a = &m.digits[m.FindDigit(pt.content[last])];
        else if (pt.content[last].type == 3)     a = &m.symbols[m.FindSymbol(pt.content[last])];

        int total = pt.max_indices[last];
        int base = total / size;
        int rem  = total % size;
        int start_idx, end_idx;
        if (rank < rem) { start_idx = rank * (base + 1);        end_idx = start_idx + base + 1; }
        else           { start_idx = rem * (base + 1) + (rank - rem) * base; end_idx = start_idx + base; }

        for (int i = start_idx; i < end_idx; i++)
            guesses.push_back(prefix + a->ordered_values[i]);

        int local_count = end_idx - start_idx;
        int global_total = 0;
        MPI_Allreduce(&local_count, &global_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        total_guesses = global_total;
    }
}

// ====== PopNext（全进程同步管理优先队列） ======
void PriorityQueue::PopNext()
{
    Generate_mpi(priority.front());
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        CalProb(pt);
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1) { priority.emplace_back(pt); break; }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }
    priority.erase(priority.begin());
}
