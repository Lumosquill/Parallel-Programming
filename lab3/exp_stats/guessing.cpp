#include "PCFG.h"
#include <fstream>
#include <algorithm>
using namespace std;

// ── 串行版 + PT 分布统计 ──
void PriorityQueue::Generate(PT pt) {
    CalProb(pt);

    int total;

    if (pt.content.size() == 1) {
        segment *a;
        if (pt.content[0].type == 1)      a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
        else                              a = &m.symbols[m.FindSymbol(pt.content[0])];
        total = pt.max_indices[0];
        for (int i = 0; i < total; i++) {
            guesses.emplace_back(a->ordered_values[i]);
            total_guesses++;
        }
    } else {
        string prefix; int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1)
                prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2)
                prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 3)
                prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx++; if (seg_idx == pt.content.size() - 1) break;
        }
        segment *a; int last = pt.content.size() - 1;
        if (pt.content[last].type == 1)      a = &m.letters[m.FindLetter(pt.content[last])];
        else if (pt.content[last].type == 2) a = &m.digits[m.FindDigit(pt.content[last])];
        else                                 a = &m.symbols[m.FindSymbol(pt.content[last])];
        total = pt.max_indices[last];
        for (int i = 0; i < total; i++) {
            guesses.emplace_back(prefix + a->ordered_values[i]);
            total_guesses++;
        }
    }

    // ── 记录 PT 填充量（全局静态 vector） ──
    static vector<int> pt_sizes;
    pt_sizes.push_back(total);

    // 达到猜测上限时输出分布统计
    static int cumulative = 0;
    cumulative += total;
    if (cumulative >= 10000000) {
        sort(pt_sizes.begin(), pt_sizes.end());

        // 按区间统计
        int bins[4][2] = {{0, 2000}, {2000, 10000}, {10000, 100000}, {100000, 2147483647}};
        int counts[4] = {0, 0, 0, 0};
        long long sums[4] = {0, 0, 0, 0};

        for (int sz : pt_sizes) {
            for (int b = 0; b < 4; b++) {
                if (sz >= bins[b][0] && sz < bins[b][1]) {
                    counts[b]++; sums[b] += sz; break;
                }
            }
        }

        cout << "\n=== PT 分布统计 ===" << endl;
        cout << "总 PT 数: " << pt_sizes.size() << endl;
        cout << "区间\t\tPT数量\t占比\t猜测数\t占比" << endl;
        int total_pt = pt_sizes.size();
        long long total_sum = 0;
        for (int b = 0; b < 4; b++) total_sum += sums[b];
        for (int b = 0; b < 4; b++) {
            printf("[%d, %d)\t%d\t%.0f%%\t%lld\t%.0f%%\n",
                bins[b][0], bins[b][1], counts[b],
                100.0 * counts[b] / total_pt,
                sums[b], 100.0 * sums[b] / total_sum);
        }

        cumulative = 0;  // 只输出一次
    }
}

// ── 以下不变 ──
void PriorityQueue::CalProb(PT &pt) {
    pt.prob = pt.preterm_prob; int index = 0;
    for (int idx : pt.curr_indices) {
        if (pt.content[index].type == 1) {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        } else if (pt.content[index].type == 2) {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        } else if (pt.content[index].type == 3) {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
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

void PriorityQueue::PopNext() {
    Generate(priority.front());
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts) {
        CalProb(pt);
        for (auto iter = priority.begin(); iter != priority.end(); iter++) {
            if (iter != priority.end() - 1 && iter != priority.begin()) {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob) {
                    priority.emplace(iter + 1, pt); break;
                }
            }
            if (iter == priority.end() - 1) { priority.emplace_back(pt); break; }
            if (iter == priority.begin() && iter->prob < pt.prob) {
                priority.emplace(iter, pt); break;
            }
        }
    }
    priority.erase(priority.begin());
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
