#include <iostream>
#include <Windows.h>
#include <fstream>
#include <string>

using namespace std;

void run(int n, int k, double **resultsTable, int index)
{
    if (n <= 0) {
        resultsTable[index] = new double[k]();
        cout << "Testing n = 0 ... Avg: 0 ms" << endl;
        return;
    }

    int **b = new int *[n];
    for (int i = 0; i < n; i++)
    {
        b[i] = new int[n];
        for (int j = 0; j < n; j++) 
            b[i][j] = i + j;
    }
    
    int *a = new int[n];
    int *sum = new int[n];
    for (int i = 0; i < n; i++) 
        a[i] = i;

    LARGE_INTEGER freq, head, tail;
    QueryPerformanceFrequency(&freq);
    
    double *record = new double[k];
    
    cout << "Testing n = " << n << " ... ";

    QueryPerformanceCounter(&head);

    for (int iter = 0; iter < k; iter++)
    {
        LARGE_INTEGER tick_h, tick_t;
        QueryPerformanceCounter(&tick_h);

        for (int i = 0; i < n; i++) 
            sum[i] = 0;

        for (int j = 0; j < n; j++)
        {
            int temp_a = a[j];
            for (int i = 0; i < n; i++)
            {
                sum[i] += b[j][i] * temp_a; 
            }
        }

        QueryPerformanceCounter(&tick_t);
        record[iter] = (double)(tick_t.QuadPart - tick_h.QuadPart) * 1000.0 / freq.QuadPart;
    }

    QueryPerformanceCounter(&tail);

    double totalTime = (double)(tail.QuadPart - head.QuadPart) * 1000.0 / freq.QuadPart;
    cout << "Avg: " << totalTime / k << " ms" << endl;

    resultsTable[index] = new double[k];
    for (int i = 0; i < k; i++) 
        resultsTable[index][i] = record[i];

    delete[] a;
    delete[] sum;
    delete[] record;
    for (int i = 0; i < n; i++) 
        delete[] b[i];
    delete[] b;
}

void save_data(const string &filename, double **arr, int rows, int cols)
{
    ofstream out_csv(filename);
    if (!out_csv.is_open())
    {
        cerr << "文件打开失败 " << filename << endl;
        return;
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            out_csv << arr[i][j];
            if (j < cols - 1) 
                out_csv << ",";
        }
        out_csv << endl;
    }

    out_csv.close();
    cout << "文件写入成功 " << filename << endl;
}

int main()
{
    cout << "*********** Performance Test: Optimal Way ***********" << endl;
    cout << "Cache Info -> L1: 320KB, L2: 5.0MB, L3: 12.0MB" << endl;
    cout << "------------------------------------------------------" << endl;

    const int testCount = 51;
    const int iterations = 1000; 
    
    int n[testCount];
    for (int i = 0; i < testCount; i++)
        n[i] = i * 100;
    
    double **dataMatrix = new double *[testCount];

    for (int i = 0; i < testCount; i++)
        run(n[i], iterations, dataMatrix, i);

    save_data("optimal_results.csv", dataMatrix, testCount, iterations);

    cout << "\nPress Enter to clean memory and exit...";
    cin.get();

    for (int i = 0; i < testCount; ++i)
        delete[] dataMatrix[i];
    delete[] dataMatrix;

    return 0;
}