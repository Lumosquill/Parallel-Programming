#ifndef PCFG_H
#define PCFG_H

#include <string>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <mpi.h>

using namespace std;

class segment
{
public:
    int type, length;
    segment(int type, int length) : type(type), length(length) {}
    segment() : type(0), length(0) {}

    void PrintSeg();
    void insert(string value);
    void order();
    void PrintValues();

    vector<string> ordered_values;
    vector<int> ordered_freqs;
    int total_freq = 0;
    unordered_map<string, int> values;
    unordered_map<int, int> freqs;
};

class PT
{
public:
    vector<segment> content;
    int pivot = 0;
    vector<int> curr_indices;
    vector<int> max_indices;
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

class PriorityQueue
{
public:
    vector<PT> priority;
    model m;
    int total_guesses = 0;
    vector<string> guesses;

    void CalProb(PT &pt);
    void init();
    void Generate(PT pt);                // 串行
    void Generate_mpi(PT pt);            // 基础版（块状分割）
    void PopNext();                       // 基础版单 PT
    void PopNextBatch_mpi(int batch_size); // 进阶版多 PT 批处理
};

#endif
