#include <iostream>
#include <Windows.h>
#include <vector>
#include <iomanip>

using namespace std;

int main() {
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    int k = 100; // 采样次数

    cout << left << setw(10) << "n" << setw(15) << "Total(ms)" << endl;
    cout << "--------------------------------------" << endl;

    // 分配最大内存块
    long long max_n = 1LL << 20;
    int* a = new int[max_n];
    for (long long i = 0; i < max_n; i++) a[i] = (int)i;

    for (int exp = 8; exp <= 20; exp++) {
        long long n = 1LL << exp;
        volatile int sum = 0; 

        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        
        // 重复执行 k 次以获得更稳定的测量值
        for (int cnt = 0; cnt < k; cnt++) {
            sum = 0;
            for (long long i = 0; i < n; i++) {
                sum += a[i];
            }
        }
        
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);

        double total_time = (double)(tail - head) * 1000.0 / freq;
        
        // 打印耗时统计
        cout << "2^" << left << setw(7) << exp 
             << fixed << setprecision(3) << setw(10) << total_time << " ms" << endl;

        // --- 你想加的代码放在这里 ---
        cout << "2^" << exp << " done." << endl;
        // ---------------------------
    }

    delete[] a;
    return 0;
}