#include "PCFG.h"
#include "md5.h"
#include <mpi.h>
using namespace std;

// mpic++ main_mpi.cpp train.cpp guessing_mpi.cpp md5.cpp -o ../main -O2 -std=c++11

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double time_train = 0, time_calc = 0;
    PriorityQueue q;

    // ======== 训练：全进程 ========
    if (rank == 0) cout << "[Rank 0] Training..." << endl;
    double t0 = MPI_Wtime();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    time_train = MPI_Wtime() - t0;
    if (rank == 0) cout << "[Rank 0] Training: " << time_train << "s" << endl;
    MPI_Barrier(MPI_COMM_WORLD);

    // ======== 初始化：全进程 ========
    q.init();
    if (rank == 0)
        cout << "[Rank 0] Queue: " << q.priority.size() << " PTs" << endl;

    // ======== 主循环 ========
    double time_guess = 0, time_hash = 0;
    double t_start = MPI_Wtime();
    const long long LIMIT = 20000000;
    int curr_num = 0;
    long long history = 0;

    while (!q.priority.empty())
    {
        q.PopNext();

        int local_guesses = (int)q.guesses.size();
        int global_guesses = 0;
        MPI_Allreduce(&local_guesses, &global_guesses, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        q.total_guesses = global_guesses;

        if (q.total_guesses - curr_num >= 100000)
        {
            if (rank == 0)
                cout << "[Rank 0] Guesses: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            if (history + q.total_guesses >= LIMIT) break;
        }

        if (curr_num > 1000000)
        {
            double h0 = MPI_Wtime();
            bit32 state[4];
            for (string pw : q.guesses)
                MD5Hash(pw, state);
            time_hash += MPI_Wtime() - h0;

            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }
    time_guess = MPI_Wtime() - t_start - time_hash;

    // ======== 输出 ========
    if (rank == 0) {
        cout << "\n========== MPI Results ==========" << endl;
        cout << "MPI processes:     " << size << endl;
        cout << "Total guesses:     " << history + q.total_guesses << endl;
        cout << "Guess time:        " << time_guess << " seconds" << endl;
        cout << "Hash time:         " << time_hash  << " seconds" << endl;
        cout << "Train time:        " << time_train << " seconds" << endl;
        cout << "Total time:        " << time_train + time_guess + time_hash << " seconds" << endl;
        cout << "=================================" << endl;
    }

    MPI_Finalize();
    return 0;
}
