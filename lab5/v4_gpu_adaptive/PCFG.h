// ============================================================
// v4_gpu_adaptive/PCFG.h — 进阶3: 按总value量触发批次
// 在 v3 pipeline 基础上, 将 batch 触发从固定 PT 个数改为累计 value 数
// ============================================================
#pragma once

#include <string>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <algorithm>
using namespace std;

class segment {
public:
    int type; int length;
    segment(int type, int length) { this->type = type; this->length = length; };
    void PrintSeg();
    vector<string> ordered_values; vector<int> ordered_freqs;
    int total_freq = 0;
    unordered_map<string, int> values; unordered_map<int, int> freqs;
    void insert(string value); void order(); void PrintValues();
};

class PT {
public:
    vector<segment> content; int pivot = 0;
    void insert(segment seg); void PrintPT(); vector<PT> NewPTs();
    vector<int> curr_indices; vector<int> max_indices;
    float preterm_prob; float prob;
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
    int FindLetter(segment seg); int FindDigit(segment seg); int FindSymbol(segment seg);
    unordered_map<int, int> preterm_freq, letters_freq, digits_freq, symbols_freq;
    vector<PT> ordered_pts;
    void train(string train_path); void store(string train_path); void load(string train_path);
    void parse(string pw); void order(); void print();
};

#define GPU_THREADS            256
#define GPU_THRESHOLD          5000     // 单PT阈值: <5000走CPU
#define GPU_SLOT_LEN           64
#define GPU_VALUE_TRIGGER      200000   // v4: 累计value数触发GPU批次

class PriorityQueue {
public:
    vector<PT> priority; model m;

    void CalProb(PT &pt); void init();
    void Generate(PT pt);

    void PopNextAdaptive();            // v4: 按value量触发 + 异步流水线
    void PopNext();

    int total_guesses = 0;
    vector<string> guesses;
    static bool heap_cmp(const PT& a, const PT& b) { return a.prob < b.prob; }
};
