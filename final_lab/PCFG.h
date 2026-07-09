#ifndef PCFG_H
#define PCFG_H

#include <string>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <omp.h>

using namespace std;

// ==================== 基础数据结构 ====================

class segment
{
public:
    int type, length;  // type: 1=letter, 2=digit, 3=symbol
    segment(int type, int length) : type(type), length(length) {}
    segment() : type(0), length(0) {}

    void PrintSeg();
    void insert(string value);
    void order();
    void PrintValues();

    vector<string> ordered_values;   // 按概率降序排列的value
    vector<int>    ordered_freqs;    // 对应频数
    int total_freq = 0;
    unordered_map<string, int> values;  // value → id
    unordered_map<int, int>    freqs;   // id → freq
};

class PT
{
public:
    vector<segment> content;
    int pivot = 0;
    vector<int> curr_indices;   // 每个segment当前指向的value下标
    vector<int> max_indices;    // 每个segment最多有多少个value
    float preterm_prob, prob;

    void insert(segment seg);
    void PrintPT();
    vector<PT> NewPTs();
};

class model
{
public:
    int preterm_id = -1, letters_id = -1, digits_id = -1, symbols_id = -1;
    int GetNextPretermID() { return ++preterm_id; }
    int GetNextLettersID() { return ++letters_id; }
    int GetNextDigitsID()  { return ++digits_id;  }
    int GetNextSymbolsID() { return ++symbols_id; }

    int total_preterm = 0;
    vector<PT> preterminals;
    vector<segment> letters, digits, symbols;
    vector<PT> ordered_pts;

    unordered_map<int, int> preterm_freq;
    unordered_map<int, int> letters_freq, digits_freq, symbols_freq;

    int FindPT(PT pt);
    int FindLetter(segment seg);
    int FindDigit(segment seg);
    int FindSymbol(segment seg);

    void train(string train_path);
    void store(string store_path);
    void load(string load_path);
    void parse(string pw);
    void order();
    void print();
};

// ==================== 优先队列 ====================

class PriorityQueue
{
public:
    vector<PT> priority;   // 按概率降序（排序 vector）
    model m;
    long long total_guesses = 0;
    vector<string> guesses;

    // 基础函数
    void CalProb(PT &pt);
    void init();

    // 串行 Generate（用于对比）
    void Generate(PT pt);

    // ★ MPI + OpenMP 双层混合 Generate
    void Generate_mpi_omp(PT pt);

    // PopNext
    void PopNext();
};

// ==================== 可配置参数 ====================
#define OMP_THRESHOLD  10000
#define OMP_CHUNK_SIZE 512
#define HASH_BATCH     4

#endif
