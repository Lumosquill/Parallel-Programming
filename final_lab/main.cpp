#include "PCFG.h"
#include "md5.h"
#include "md5_neon.h"
#include <fstream>
#include <unordered_set>

using namespace std;

// ================================================================
// 编译 (ARM):
//   mpic++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon.cpp \
//          -o ../main -O2 -fopenmp -std=c++11 -march=native
//
// 运行:
//   mpirun -np 4 ./main [MODE]
//     MODE=0: 纯 MPI
//     MODE=1: MPI + OpenMP
//     MODE=2: MPI + OpenMP + SIMD（默认）
// ================================================================

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ── 模式 ──
    int mode = 2;
    if (argc >= 2) mode = atoi(argv[1]);

    if (rank == 0) {
        const char* names[] = {"Pure MPI", "MPI+OpenMP", "MPI+OpenMP+SIMD"};
        cout << "============================================" << endl;
        cout << "  Mode:              " << names[mode] << endl;
        cout << "  MPI processes:     " << size << endl;
        #ifdef _OPENMP
        cout << "  OpenMP threads:    " << omp_get_max_threads() << endl;
        #endif
        cout << "  OMP_THRESHOLD:     " << OMP_THRESHOLD << endl;
        cout << "============================================" << endl;
    }

    PriorityQueue q;

    // ==================== 训练（全进程独立训练，参考 Lab4） ====================
    double time_train = 0;
    if (rank == 0) cout << "[Rank 0] Training..." << endl;
    double t0 = MPI_Wtime();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    time_train = MPI_Wtime() - t0;
    if (rank == 0) cout << "[Rank 0] Training: " << time_train << "s" << endl;
    MPI_Barrier(MPI_COMM_WORLD);

    // ==================== 初始化（全进程） ====================
    q.init();
    if (rank == 0)
        cout << "[Rank 0] Queue: " << q.priority.size() << " PTs" << endl;

    // ==================== 主循环 ====================
    double time_guess = 0, time_hash = 0;
    const long long LIMIT = 20000000;
    long long history = 0;

    vector<string> hash_buf;
    const int HASH_FLUSH = 100000;

    while (!q.priority.empty())
    {
        // ── 口令生成（MPI + OpenMP 双层）──
        double g0 = MPI_Wtime();
        q.PopNext();
        time_guess += MPI_Wtime() - g0;

        long long iter_total = q.total_guesses;  // 全局总和
        history += iter_total;

        // ── 攒入哈希缓冲区 ──
        for (const string& pw : q.guesses)
            hash_buf.push_back(pw);
        q.guesses.clear();

        // ── 批量 SIMD 哈希 ──
        if ((int)hash_buf.size() >= HASH_FLUSH)
        {
            double h0 = MPI_Wtime();
            if (mode == 2)
            {
                int aligned = ((int)hash_buf.size() / HASH_BATCH) * HASH_BATCH;

                #pragma omp parallel for schedule(static)
                for (int i = 0; i < aligned; i += HASH_BATCH)
                {
                    string batch[HASH_BATCH];
                    bit32 st[HASH_BATCH][4];
                    for (int j = 0; j < HASH_BATCH; j++)
                        batch[j] = hash_buf[i + j];
                    MD5Hash_SIMD4(batch, st);
                }

                #pragma omp parallel for schedule(static)
                for (int i = aligned; i < (int)hash_buf.size(); i++)
                {
                    bit32 s[4];
                    MD5Hash(hash_buf[i], s);
                }
            }
            else
            {
                for (const string& pw : hash_buf) {
                    bit32 s[4]; MD5Hash(pw, s);
                }
            }
            time_hash += MPI_Wtime() - h0;
            hash_buf.clear();
        }

        // ── 进度 ──
        if (rank == 0 && history % 2000000 < 100000)
            cout << "[Rank 0] Guesses: " << history
                 << " | Queue: " << q.priority.size() << endl;

        // ── 终止 ──
        if (history >= LIMIT) break;
    }

    // ── 冲刷哈希缓冲 ──
    if (!hash_buf.empty())
    {
        double h0 = MPI_Wtime();
        if (mode == 2)
        {
            int aligned = ((int)hash_buf.size() / HASH_BATCH) * HASH_BATCH;
            for (int i = 0; i < aligned; i += HASH_BATCH)
            {
                string batch[HASH_BATCH];
                bit32 st[HASH_BATCH][4];
                for (int j = 0; j < HASH_BATCH; j++)
                    batch[j] = hash_buf[i + j];
                MD5Hash_SIMD4(batch, st);
            }
            for (int i = aligned; i < (int)hash_buf.size(); i++) {
                bit32 s[4]; MD5Hash(hash_buf[i], s);
            }
        }
        else
        {
            for (const string& pw : hash_buf) {
                bit32 s[4]; MD5Hash(pw, s);
            }
        }
        time_hash += MPI_Wtime() - h0;
    }

    double total_time = MPI_Wtime() - t0;

    // ==================== 输出 ====================
    if (rank == 0)
    {
        cout << "\n"
             << "============================================" << endl
             << "           FINAL HYBRID RESULTS             " << endl
             << "============================================" << endl
             << "  Mode:              " << mode               << endl
             << "  MPI processes:     " << size               << endl;
        #ifdef _OPENMP
        cout << "  OpenMP threads:    " << omp_get_max_threads() << endl;
        #endif
        cout << "  Total guesses:     " << history            << endl
             << "--------------------------------------------" << endl
             << "  Train time:        " << time_train  << " s" << endl
             << "  Guess time:        " << time_guess  << " s" << endl
             << "  Hash  time:        " << time_hash   << " s" << endl
             << "  Total wall time:   " << total_time  << " s" << endl
             << "--------------------------------------------" << endl
             << "  Guess / Total:     " << time_guess / total_time * 100 << " %" << endl
             << "  Hash  / Total:     " << time_hash  / total_time * 100 << " %" << endl
             << "============================================" << endl;
    }

    MPI_Finalize();
    return 0;
}
