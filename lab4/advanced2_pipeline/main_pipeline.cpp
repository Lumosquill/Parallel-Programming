#include "PCFG.h"
#include "md5.h"
#include <mpi.h>
#include <functional>
using namespace std;

// mpic++ main_pipeline.cpp train.cpp pipeline.cpp md5.cpp -o ../main -O2 -std=c++11 -pthread

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double time_train = 0, time_compute = 0;
    PriorityQueue q;
    PasswordQueue pq;

    // ======== 训练：全进程 ========
    if (rank == 0) cout << "[Rank " << rank << "] Training..." << endl;
    double t0 = MPI_Wtime();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    time_train = MPI_Wtime() - t0;
    if (rank == 0) cout << "[Rank " << rank << "] Training: " << time_train << "s" << endl;
    MPI_Barrier(MPI_COMM_WORLD);

    // ======== 初始化 ========
    q.init();
    if (rank == 0) cout << "[Rank " << rank << "] Queue: " << q.priority.size() << " PTs" << endl;

    // ======== 静态 PT 分配（孙沐赟方式：各进程独立，零 MPI 通信） ========
    int total_pts = (int)q.priority.size();
    int base = total_pts / size, rem = total_pts % size;
    int start, end;
    if (rank < rem) { start = rank * (base + 1);      end = start + base + 1; }
    else           { start = rem * (base + 1) + (rank - rem) * base; end = start + base; }

    // 只保留自己分到的 PT
    q.priority = vector<PT>(q.priority.begin() + start, q.priority.begin() + end);
    cout << "[Rank " << rank << "] My PTs: " << q.priority.size() << " (indices " << start << "-" << end << ")" << endl;

    // ======== Pipeline：生产者线程 + 消费者线程 ========
    double t_comp = MPI_Wtime();
    const int CONSUMERS = 4;
    const long long LIMIT = 20000000 / size;  // 每进程上限，全局共 2000 万

    // 生产者线程
    thread producer([&q, &pq, LIMIT]() { q.PopNext_pipeline(pq, LIMIT); });

    // 消费者线程
    vector<thread> consumers;
    vector<long long> local_hashed(CONSUMERS, 0);
    for (int i = 0; i < CONSUMERS; i++)
    {
        consumers.emplace_back([&pq, &local_hashed, i]() {
            vector<string> batch;
            while (pq.pop(batch))
            {
                bit32 st[4];
                for (const string &pw : batch)
                    MD5Hash(pw, st);
                local_hashed[i] += (long long)batch.size();
            }
        });
    }

    producer.join();
    for (auto &t : consumers) t.join();

    time_compute = MPI_Wtime() - t_comp;

    // ======== 汇总 ========
    long long local_total = q.total_guesses;
    long long global_total = 0;
    MPI_Reduce(&local_total, &global_total, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    long long local_hash = 0;
    for (long long h : local_hashed) local_hash += h;
    long long global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // ======== 输出 ========
    if (rank == 0) {
        cout << "\n========== Pipeline Results ==========" << endl;
        cout << "MPI processes:     " << size << endl;
        cout << "Consumers/proc:    " << CONSUMERS << endl;
        cout << "Total guesses:     " << global_total << endl;
        cout << "Compute time:      " << time_compute << " seconds" << endl;
        cout << "Train time:        " << time_train << " seconds" << endl;
        cout << "Total time:        " << time_train + time_compute << " seconds" << endl;
        cout << "=======================================" << endl;
    }

    MPI_Finalize();
    return 0;
}
