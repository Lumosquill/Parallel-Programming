#include "PCFG.h"
using namespace std;

ThreadPool* g_pool = nullptr;

void InitThreadPool(int n) {
    if (g_pool == nullptr)
        g_pool = new ThreadPool(n);   // 创建 n 个工作线程（n = THREAD_NUM - 1）
}

void CleanupThreadPool() {
    if (g_pool != nullptr) { delete g_pool; g_pool = nullptr; }
}

// ── 直接执行一个任务（工作线程和主线程共用） ──
void ExecuteTask(const Task& task) {
    segment* a = task.a;
    vector<string>& g = *(task.guesses);
    size_t off = task.offset;
    for (int i = task.start_index; i < task.end_index; i++)
        g[off + (i - task.start_index)] = task.pre_terminal + a->ordered_values[i];
}

// ── 工作线程入口 ──
void* ThreadPool::WorkerThread(void* arg) {
    ThreadPool* pool = static_cast<ThreadPool*>(arg);

    while (true) {
        pthread_mutex_lock(&pool->queue_mutex);
        while (!pool->stop && pool->tasks.empty())
            pthread_cond_wait(&pool->condition, &pool->queue_mutex);

        if (pool->stop && pool->tasks.empty()) {
            pthread_mutex_unlock(&pool->queue_mutex);
            return NULL;
        }

        Task task = pool->tasks.front();
        pool->tasks.pop();
        pthread_mutex_unlock(&pool->queue_mutex);

        // ★ 直接写入 guesses 的 offset 位置，无锁
        ExecuteTask(task);

        pthread_mutex_lock(&pool->queue_mutex);
        pool->active_tasks--;
        if (pool->active_tasks == 0 && pool->tasks.empty())
            pthread_cond_signal(&pool->completion_cond);
        pthread_mutex_unlock(&pool->queue_mutex);
    }
}

// ── 最终版 Generate：线程池 + 预分配索引写入 + 主线程参与 + 阈值 ──
void PriorityQueue::Generate_pool(PT pt) {
    if (g_pool == nullptr) InitThreadPool(THREAD_NUM - 1);   // 懒初始化，兼容未显式 Init 的 main
    CalProb(pt);

    // ── 获取 segment 指针和前缀 ──
    segment *a;
    string prefix;
    int total;

    if (pt.content.size() == 1) {
        if (pt.content[0].type == 1)      a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
        else                              a = &m.symbols[m.FindSymbol(pt.content[0])];
        prefix = "";
        total = pt.max_indices[0];
    } else {
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1)
                prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2)
                prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 3)
                prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx++;
            if (seg_idx == pt.content.size() - 1) break;
        }
        int last = pt.content.size() - 1;
        if (pt.content[last].type == 1)      a = &m.letters[m.FindLetter(pt.content[last])];
        else if (pt.content[last].type == 2) a = &m.digits[m.FindDigit(pt.content[last])];
        else                                 a = &m.symbols[m.FindSymbol(pt.content[last])];
        total = pt.max_indices[last];
    }

    // ── 阈值：小任务串行 ──
    if (total < THRESHOLD) {
        for (int i = 0; i < total; i++)
            guesses.emplace_back(prefix + a->ordered_values[i]);
        total_guesses += total;
        return;
    }

    // ── 预分配空间 ──
    size_t old = guesses.size();
    guesses.resize(old + total);

    int chunk = (total + THREAD_NUM - 1) / THREAD_NUM;

    // ★ 为线程 t=1..THREAD_NUM-1 构建任务并入队
    for (int t = 1; t < THREAD_NUM; t++) {
        int start = t * chunk;
        int end   = (t == THREAD_NUM - 1) ? total : (t + 1) * chunk;
        if (start >= total) continue;  // 任务量不够分满所有线程

        Task task;
        task.pre_terminal = prefix;
        task.a = a;
        task.guesses = &guesses;
        task.offset = old + start;
        task.start_index = start;
        task.end_index = end;
        g_pool->Enqueue(task);
    }

    // ★ 主线程自己处理第一份（t=0），不闲置
    {
        int start = 0;
        int end   = min(chunk, total);
        size_t off = old + start;
        for (int i = start; i < end; i++)
            guesses[off + (i - start)] = prefix + a->ordered_values[i];
    }

    // 等待工作线程完成
    g_pool->WaitAll();

    total_guesses += total;
}

// ── 串行版（保留供正确性对比） ──
void PriorityQueue::Generate(PT pt) {
    CalProb(pt);
    if (pt.content.size() == 1) {
        segment *a;
        if (pt.content[0].type == 1)      a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
        else                              a = &m.symbols[m.FindSymbol(pt.content[0])];
        for (int i = 0; i < pt.max_indices[0]; i++) {
            guesses.emplace_back(a->ordered_values[i]);
            total_guesses++;
        }
    } else {
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
            if (seg_idx == pt.content.size() - 1) break;
        }
        segment *a;
        int last = pt.content.size() - 1;
        if (pt.content[last].type == 1)      a = &m.letters[m.FindLetter(pt.content[last])];
        else if (pt.content[last].type == 2) a = &m.digits[m.FindDigit(pt.content[last])];
        else                                 a = &m.symbols[m.FindSymbol(pt.content[last])];
        for (int i = 0; i < pt.max_indices[last]; i++) {
            guesses.emplace_back(prefix + a->ordered_values[i]);
            total_guesses++;
        }
    }
}

// ── 公共函数 ──
void PriorityQueue::CalProb(PT &pt) {
    pt.prob = pt.preterm_prob;
    int index = 0;
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
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

void PriorityQueue::PopNext() {
    Generate_pool(priority.front());

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
        if (curr_indices[i] < max_indices[i]) {
            pivot = i;
            res.emplace_back(*this);
        }
        curr_indices[i]--;
    }
    pivot = init_pivot;
    return res;
}
