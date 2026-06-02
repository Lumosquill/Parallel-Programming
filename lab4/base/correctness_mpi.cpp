#include "PCFG.h"
#include "md5.h"
#include <mpi.h>
#include <unordered_set>
#include <fstream>
using namespace std;

// mpic++ correctness_mpi.cpp train.cpp guessing_mpi.cpp md5.cpp -o ../main -O2 -std=c++11

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    PriorityQueue q;

    // ======== 训练 ========
    if (rank == 0) cout << "[Rank 0] Training..." << endl;
    double t0 = MPI_Wtime();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    if (rank == 0) cout << "[Rank 0] Training: " << MPI_Wtime() - t0 << "s" << endl;
    MPI_Barrier(MPI_COMM_WORLD);

    // ======== 初始化 ========
    q.init();
    if (rank == 0) cout << "[Rank 0] Queue: " << q.priority.size() << " PTs" << endl;

    // ======== 加载测试集（每个进程独立加载以减少通信） ========
    unordered_set<string> test_set;
    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
    int test_count = 0;
    string pw;
    while (test_data >> pw && test_count < 1000000) {
        test_set.insert(pw);
        test_count++;
    }
    if (rank == 0) cout << "[Rank 0] Test set: " << test_set.size() << " passwords" << endl;

    // ======== 主循环（同基础版，增加破解统计） ========
    const long long LIMIT = 20000000;
    long long local_cracked = 0;
    int curr_num = 0;
    long long history = 0;

    while (!q.priority.empty())
    {
        q.PopNext();

        int lc = (int)q.guesses.size(), gc = 0;
        MPI_Allreduce(&lc, &gc, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        q.total_guesses = gc;

        if (q.total_guesses - curr_num >= 100000) {
            if (rank == 0) cout << "[Rank 0] Guesses: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;
            if (history + q.total_guesses >= LIMIT) break;
        }

        if (curr_num > 1000000) {
            for (const string &pw : q.guesses)
                if (test_set.count(pw)) local_cracked++;

            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }

    // ======== 汇总破解数 ========
    long long global_cracked = 0;
    MPI_Reduce(&local_cracked, &global_cracked, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
        cout << "\nCracked: " << global_cracked << endl;

    MPI_Finalize();
    return 0;
}
