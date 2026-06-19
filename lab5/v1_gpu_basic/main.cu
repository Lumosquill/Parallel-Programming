// ============================================================
// v1_gpu_basic/main.cu — 基础要求: 单PT GPU加速
// 基于 lab3/v6_openmp_final/main_neon.cpp
// ============================================================

#include "PCFG.h"
#include "../common/md5.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
using namespace std;
using namespace chrono;

extern void gpu_cleanup();

int main()
{
    // ── MD5 正确性测试 (原版一字不改) ──
    cout << "Testing MD5Hash correctness..." << endl;
    string test_pws[8] = {"123456", "password", "12345678", "qwerty",
                          "123456789", "12345", "1234", "111111"};
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
        for (int i1 = 0; i1 < 4; i1++)
            ss << setw(8) << setfill('0') << hex << state[i1];
        if (ss.str() != test_hashes[i]) {
            cout << "MD5Hash test failed for " << test_pws[i] << "!" << endl;
            return 1;
        }
    }
    cout << "MD5Hash test passed!" << endl;

    // ── 训练 ──
    double time_hash = 0, time_guess = 0, time_train = 0;
    PriorityQueue q;
    auto t0 = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto t1 = system_clock::now();
    time_train = double(duration_cast<microseconds>(t1 - t0).count())
               * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "Training done, starting guess generation..." << endl;

    // ── 主循环: GPU加速生成 ──
    int curr_num = 0, history = 0;
    auto t_start = system_clock::now();

    while (!q.priority.empty()) {
        q.PopNext();                      // ★ 调用 GPU 加速版
        q.total_guesses = q.guesses.size();

        if (q.total_guesses - curr_num >= 100000) {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;
        }

        // 达到 1000 万 → 停止并计时
        if (history + q.total_guesses > 10000000) {
            auto t2 = system_clock::now();
            time_guess = double(duration_cast<microseconds>(t2 - t_start).count())
                       * microseconds::period::num / microseconds::period::den;
            cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
            cout << "Hash time:" << time_hash << "seconds" << endl;
            cout << "Train time:" << time_train << "seconds" << endl;
            break;
        }

        // 每 100 万 guesses 做一次 MD5 哈希
        if (curr_num > 1000000) {
            auto th0 = system_clock::now();
            size_t total = q.guesses.size();
            for (size_t i = 0; i < total; i++) {
                bit32 state[4];
                MD5Hash(q.guesses[i], state);
            }
            auto th1 = system_clock::now();
            time_hash += double(duration_cast<microseconds>(th1 - th0).count())
                       * microseconds::period::num / microseconds::period::den;

            history += total;
            q.guesses.clear();
            curr_num = 0;
        }
    }

    gpu_cleanup();
    return 0;
}
