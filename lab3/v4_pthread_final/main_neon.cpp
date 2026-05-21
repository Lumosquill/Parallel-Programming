#include "md5_neon.h"
#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
using namespace std;
using namespace chrono;

extern ThreadPool* g_pool;
extern void InitThreadPool(int n);
extern void CleanupThreadPool();

int main()
{
    InitThreadPool(THREAD_NUM - 1);  // 7 个子线程 + 主线程 = 8

    cout << "Testing MD5Hash correctness..." << endl;
    string test_pws[8] = {"123456", "password", "12345678", "qwerty", "123456789", "12345", "1234", "111111"};
    string test_hashes[8] = {
        "e10adc3949ba59abbe56e057f20f883e",
        "5f4dcc3b5aa765d61d8327deb882cf99",
        "25d55ad283aa400af464c76d713c07ad",
        "d8578edf8458ce06fbc5bb76a58c5ca4",
        "25f9e794323b453885f5181f1b624d0b",
        "827ccb0eea8a706c4c34a16891f84e7b",
        "81dc9bdb52d04dc20036dbd8313ed055",
        "96e79218965eb72c92a549dd5a330112"
    };
    for (int i = 0; i < 8; i++) {
        bit32 state[4];
        MD5Hash(test_pws[i], state);
        stringstream ss;
        for (int i1 = 0; i1 < 4; i1++) {
            ss << std::setw(8) << std::setfill('0') << hex << state[i1];
        }
        if (ss.str() != test_hashes[i]) {
            cout << "MD5Hash test failed for " << test_pws[i] << "!" << endl;
            CleanupThreadPool();
            return 1;
        }
    }
    cout << "MD5Hash test passed!" << endl;

    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "here" << endl;
    int curr_num = 0;
    auto start = system_clock::now();
    int history = 0;
    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();
        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            int generate_n = 10000000;
            if (history + q.total_guesses > 10000000)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                cout << "Hash time:" << time_hash << "seconds" << endl;
                cout << "Train time:" << time_train << "seconds" << endl;
                break;
            }
        }
        if (curr_num > 1000000) {
            auto start_hash = system_clock::now();

            size_t total     = q.guesses.size();
            size_t int_total = (total / 4) * 4;
            size_t left      = total - int_total;

            for (size_t i = 0; i < int_total; i += 4) {
                string batch[4] = { q.guesses[i], q.guesses[i+1],
                                    q.guesses[i+2], q.guesses[i+3] };
                bit32 states[4][4];
                MD5Hash_SIMD4(batch, states);
            }

            for (size_t i = 0; i < left; i++) {
                bit32 state[4];
                MD5Hash(q.guesses[int_total + i], state);
            }

            auto end_hash = system_clock::now();
            auto duration_hash = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration_hash.count()) * microseconds::period::num / microseconds::period::den;

            q.guesses.clear();
            history += int_total + left;
            curr_num = 0;
        }
    }

    CleanupThreadPool();
}
