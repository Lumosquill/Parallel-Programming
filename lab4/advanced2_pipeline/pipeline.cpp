#include "PCFG.h"
#include <algorithm>
using namespace std;

static bool heap_cmp(const PT &a, const PT &b) { return a.prob < b.prob; }

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
    make_heap(priority.begin(), priority.end(), heap_cmp);
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

void PriorityQueue::Generate_one(PT &pt, vector<string> &out)
{
    CalProb(pt);
    if (pt.content.size() == 1)
    {
        segment *a = nullptr;
        if (pt.content[0].type == 1)          a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)     a = &m.digits[m.FindDigit(pt.content[0])];
        else if (pt.content[0].type == 3)     a = &m.symbols[m.FindSymbol(pt.content[0])];
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
        for (int i = 0; i < (int)pt.max_indices[last]; i++)
            out.emplace_back(prefix + a->ordered_values[i]);
    }
}

// Pipeline 生产者（堆优化 + batch 限大小）
void PriorityQueue::PopNext_pipeline(PasswordQueue &pq, long long limit)
{
    const int BATCH_MAX = 50000;  // 每批最多 5 万条
    while (!priority.empty() && total_guesses < limit)
    {
        pop_heap(priority.begin(), priority.end(), heap_cmp);
        PT top_pt = priority.back();
        priority.pop_back();

        vector<string> full;
        Generate_one(top_pt, full);
        total_guesses += (long long)full.size();

        // 大 PT 拆成多个小 batch 推入队列
        for (size_t i = 0; i < full.size(); i += BATCH_MAX)
        {
            size_t end = min(i + BATCH_MAX, full.size());
            vector<string> chunk(full.begin() + i, full.begin() + end);
            pq.push(chunk);
        }

        if (total_guesses >= limit) break;

        vector<PT> new_pts = top_pt.NewPTs();
        for (PT &np : new_pts)
        {
            CalProb(np);
            priority.push_back(np);
            push_heap(priority.begin(), priority.end(), heap_cmp);
        }
    }
    pq.set_done();
}
