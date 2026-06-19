// ============================================================
// v1_gpu_basic/PCFG.h — 基础要求: 单PT GPU加速
// 基于 lab3/v6_openmp_final 原版, segment/PT/model 一字未改
// ============================================================
#pragma once

#include <string>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <algorithm>
using namespace std;

// ──────────────── 原版数据结构 (不改) ────────────────

class segment {
public:
    int type;       // 1=字母 2=数字 3=特殊字符
    int length;
    segment(int type, int length) { this->type = type; this->length = length; };
    void PrintSeg();
    vector<string> ordered_values;
    vector<int> ordered_freqs;
    int total_freq = 0;
    unordered_map<string, int> values;
    unordered_map<int, int> freqs;
    void insert(string value);
    void order();
    void PrintValues();
};

class PT {
public:
    vector<segment> content;
    int pivot = 0;
    void insert(segment seg);
    void PrintPT();
    vector<PT> NewPTs();
    vector<int> curr_indices;
    vector<int> max_indices;
    float preterm_prob;
    float prob;
};

class model {
public:
    int preterm_id = -1, letters_id = -1, digits_id = -1, symbols_id = -1;
    int GetNextPretermID()  { preterm_id++;  return preterm_id; };
    int GetNextLettersID()  { letters_id++;  return letters_id; };
    int GetNextDigitsID()   { digits_id++;   return digits_id; };
    int GetNextSymbolsID()  { symbols_id++;  return symbols_id; };
    int total_preterm = 0;
    vector<PT> preterminals;
    int FindPT(PT pt);
    vector<segment> letters, digits, symbols;
    int FindLetter(segment seg);
    int FindDigit(segment seg);
    int FindSymbol(segment seg);
    unordered_map<int, int> preterm_freq, letters_freq, digits_freq, symbols_freq;
    vector<PT> ordered_pts;
    void train(string train_path);
    void store(string store_path);
    void load(string load_path);
    void parse(string pw);
    void order();
    void print();
};

// ──────────────── GPU 常量 ────────────────
#define GPU_THREADS     256
#define GPU_THRESHOLD   5000    // 单PT阈值: value数<此走CPU
#define GPU_SLOT_LEN    64      // 固定输出槽位(字节)

// ──────────────── PriorityQueue ────────────────
class PriorityQueue {
public:
    vector<PT> priority;
    model m;
    void CalProb(PT &pt);
    void init();

    // 原版CPU Generate (保留, 用于小任务和正确性比对)
    void Generate(PT pt);

    // ★ GPU加速版 (基础要求核心)
    void GenerateGPU(PT pt);

    // PopNext: 用GPU版替换原版
    void PopNext();

    int total_guesses = 0;
    vector<string> guesses;

    // 堆比较器
    static bool heap_cmp(const PT& a, const PT& b) { return a.prob < b.prob; }

private:
    static string build_prefix(const PT& pt, model& m);
    segment* get_last_seg(const PT& pt);
};
