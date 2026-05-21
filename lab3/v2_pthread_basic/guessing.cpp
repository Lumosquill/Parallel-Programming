#include "PCFG.h"
using namespace std;

void* PriorityQueue::fill_range(void* arg) {
    ThreadTask* task = (ThreadTask*)arg;
    vector<string> local;
    for (int i = task->start_index; i < task->end_index; i++)
        local.emplace_back(task->pre_terminal + task->a->ordered_values[i]);

    pthread_mutex_lock(task->mutex);
    task->guesses_out->insert(task->guesses_out->end(), local.begin(), local.end());
    *(task->total_guesses) += local.size();
    pthread_mutex_unlock(task->mutex);
    return NULL;
}

void PriorityQueue::Generate(PT pt) {
    CalProb(pt);
    if (pt.content.size() == 1) {
        segment *a;
        if (pt.content[0].type == 1)      a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
        else                              a = &m.symbols[m.FindSymbol(pt.content[0])];
        for (int i = 0; i < pt.max_indices[0]; i++) {
            guesses.emplace_back(a->ordered_values[i]); total_guesses++;
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
        for (int i = 0; i < pt.max_indices[last]; i++) {
            guesses.emplace_back(prefix + a->ordered_values[i]); total_guesses++;
        }
    }
}

void PriorityQueue::Generate_pthread(PT pt) {
    CalProb(pt);
    const int NUM = 8;
    const int WORKERS = NUM - 1;   // 7 个子线程，主线程做 t=0

    if (pt.content.size() == 1) {
        segment *a;
        if (pt.content[0].type == 1)      a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
        else                              a = &m.symbols[m.FindSymbol(pt.content[0])];

        int total = pt.max_indices[0];
        int chunk = (total + NUM - 1) / NUM;

        // ★ 主线程先干（t=0），干完再启子线程，避免并发写 guesses
        int main_end = min(chunk, total);
        for (int i = 0; i < main_end; i++)
            guesses.emplace_back(a->ordered_values[i]);
        total_guesses += main_end;

        pthread_t threads[WORKERS];
        ThreadTask tasks[WORKERS];

        for (int t = 0; t < WORKERS; t++) {
            int tid = t + 1;
            tasks[t].pre_terminal = "";
            tasks[t].a = a;
            tasks[t].guesses_out = &guesses;
            tasks[t].total_guesses = &total_guesses;
            tasks[t].mutex = &mutex;
            tasks[t].start_index = tid * chunk;
            tasks[t].end_index = (tid == NUM - 1) ? total : (tid + 1) * chunk;
            pthread_create(&threads[t], NULL, PriorityQueue::fill_range, &tasks[t]);
        }

        for (int t = 0; t < WORKERS; t++)
            pthread_join(threads[t], NULL);
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

        int total = pt.max_indices[last];
        int chunk = (total + NUM - 1) / NUM;

        // ★ 主线程先干，干完再启子线程
        int main_end = min(chunk, total);
        for (int i = 0; i < main_end; i++)
            guesses.emplace_back(prefix + a->ordered_values[i]);
        total_guesses += main_end;

        pthread_t threads[WORKERS];
        ThreadTask tasks[WORKERS];

        for (int t = 0; t < WORKERS; t++) {
            int tid = t + 1;
            tasks[t].pre_terminal = prefix;
            tasks[t].a = a;
            tasks[t].guesses_out = &guesses;
            tasks[t].total_guesses = &total_guesses;
            tasks[t].mutex = &mutex;
            tasks[t].start_index = tid * chunk;
            tasks[t].end_index = (tid == NUM - 1) ? total : (tid + 1) * chunk;
            pthread_create(&threads[t], NULL, PriorityQueue::fill_range, &tasks[t]);
        }

        for (int t = 0; t < WORKERS; t++)
            pthread_join(threads[t], NULL);
    }
}

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
    Generate_pthread(priority.front());
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
