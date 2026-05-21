#include <string>
#include <iostream>
#include <unordered_map>
#include <queue>
#include <pthread.h>
#include <vector>
using namespace std;

#define THREAD_NUM 8
#define THRESHOLD 10000

class segment;
class PriorityQueue;

// ── 任务参数 ──
struct Task {
    string pre_terminal;
    segment* a;
    vector<string>* guesses;    // ★ 直接指向全局 guesses
    size_t offset;              // ★ 写入起始位置
    int start_index;
    int end_index;
};

// ── 线程池（创建 THREAD_NUM-1 个工作线程，主线程分担一份） ──
class ThreadPool {
private:
    vector<pthread_t> workers;
    queue<Task> tasks;
    pthread_mutex_t queue_mutex;
    pthread_cond_t condition;
    pthread_cond_t completion_cond;
    int active_tasks;
    bool stop;

public:
    ThreadPool(size_t num_threads) : stop(false), active_tasks(0) {
        pthread_mutex_init(&queue_mutex, NULL);
        pthread_cond_init(&condition, NULL);
        pthread_cond_init(&completion_cond, NULL);
        workers.resize(num_threads);
        for (size_t i = 0; i < num_threads; i++)
            pthread_create(&workers[i], NULL, WorkerThread, this);
    }

    ~ThreadPool() {
        pthread_mutex_lock(&queue_mutex);
        stop = true;
        pthread_mutex_unlock(&queue_mutex);
        pthread_cond_broadcast(&condition);
        for (size_t i = 0; i < workers.size(); i++)
            pthread_join(workers[i], NULL);
        pthread_mutex_destroy(&queue_mutex);
        pthread_cond_destroy(&condition);
        pthread_cond_destroy(&completion_cond);
    }

    void Enqueue(const Task& task) {
        pthread_mutex_lock(&queue_mutex);
        tasks.push(task);
        active_tasks++;
        pthread_mutex_unlock(&queue_mutex);
        pthread_cond_signal(&condition);
    }

    void WaitAll() {
        pthread_mutex_lock(&queue_mutex);
        while (active_tasks > 0 || !tasks.empty())
            pthread_cond_wait(&completion_cond, &queue_mutex);
        pthread_mutex_unlock(&queue_mutex);
    }

private:
    static void* WorkerThread(void* arg);
};

extern ThreadPool* g_pool;
void InitThreadPool(int n);
void CleanupThreadPool();

// ── 内联执行（主线程干自己的任务，不经线程池） ──
void ExecuteTask(const Task& task);

// ── segment, PT, model（不变） ──
class segment
{
public:
    int type;
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

class PT
{
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

class model
{
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

class PriorityQueue
{
public:
    vector<PT> priority;
    model m;
    void CalProb(PT &pt);
    void init();
    void Generate(PT pt);
    void Generate_pool(PT pt);
    void PopNext();
    int total_guesses = 0;
    vector<string> guesses;
};
