#include "PCFG.h"
#include "md5.h"
#include <mpi.h>
using namespace std;

// mpic++ main_mpi_adv.cpp train.cpp guessing_mpi_adv.cpp md5.cpp -o ../main -O2 -std=c++11

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double time_train = 0, time_guess = 0, time_hash = 0;
    PriorityQueue q;

    if (rank == 0) cout << "[Rank 0] Training..." << endl;
    double t0 = MPI_Wtime();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    time_train = MPI_Wtime() - t0;
    if (rank == 0) cout << "[Rank 0] Training: " << time_train << "s" << endl;
    MPI_Barrier(MPI_COMM_WORLD);

    q.init();
    if (rank == 0) cout << "[Rank 0] Queue: " << q.priority.size() << " PTs" << endl;

    double t_guess = MPI_Wtime();
    const long long LIMIT = 20000000;
    int curr_num = 0;
    long long history = 0;

    // 全进程检查队列（队列同步），零额外 Bcast
    while (!q.priority.empty())
    {
        q.PopNextBatch_mpi(size);

        int lc = (int)q.guesses.size(), gc = 0;
        MPI_Allreduce(&lc, &gc, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        q.total_guesses = gc;

        if (q.total_guesses - curr_num >= 100000) {
            if (rank == 0) cout << "[Rank 0] Guesses: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;
            if (history + q.total_guesses >= LIMIT) break;
        }
        if (curr_num > 1000000) {
            double h0 = MPI_Wtime();
            bit32 st[4];
            for (string pw : q.guesses) MD5Hash(pw, st);
            time_hash += MPI_Wtime() - h0;
            history += curr_num; curr_num = 0; q.guesses.clear();
        }
    }
    time_guess = MPI_Wtime() - t_guess - time_hash;

    if (rank == 0) {
        cout << "\n========== MPI Adv Results ==========" << endl;
        cout << "MPI processes:     " << size << endl;
        cout << "Total guesses:     " << history + q.total_guesses << endl;
        cout << "Guess time:        " << time_guess << " seconds" << endl;
        cout << "Hash time:         " << time_hash  << " seconds" << endl;
        cout << "Train time:        " << time_train << " seconds" << endl;
        cout << "Total time:        " << time_train + time_guess + time_hash << " seconds" << endl;
        cout << "======================================" << endl;
    }
    MPI_Finalize();
    return 0;
}
