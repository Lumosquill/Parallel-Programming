#include<iostream>
#include<Windows.h>
#include<cmath> 
#include<iomanip> 

using namespace std;

int main() {
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    int k = 1000; 

    cout << left << setw(15) << "n (规模)" 
         << setw(20) << "总时间 (ms)" 
         << setw(20) << "平均时间 (ms)" << endl;
    cout << "------------------------------------------------------------" << endl;

    
    for (int exp = 8; exp <= 20; exp++) {
        int n = static_cast<int>(pow(2, exp));
        

        int* a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
        }

        int sum, sum1, sum2;
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        
        for (int cnt = 0; cnt < k; cnt++) {
           
            sum1 = sum2 = 0;
            for (int i = 0; i < n; i += 2) {
                sum1 += a[i];
                sum2 += a[i + 1];
            }
            sum = sum1 + sum2;
        }

        QueryPerformanceCounter((LARGE_INTEGER*)&tail);

        double total_time = (tail - head) * 1000.0 / freq;
        double avg_time = total_time / k;

        cout << left << "2^" << setw(13) << exp 
             << setw(20) << total_time 
             << setw(20) << avg_time << endl;

        delete[] a;
    }

    return 0;
}