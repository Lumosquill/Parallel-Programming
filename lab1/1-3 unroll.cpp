#include <iostream>
#include <Windows.h>
#include <fstream>
#include <string>

using namespace std;

// 循环展开优化算法实现
void run_unroll(int n, int k, double **resultsTable, int index)
{
    // 1. 数据初始化 
    // 注意：为了配合 unroll 逻辑，b[j][i] 对应原平凡算法中的 b[行][列]
    int **b = new int *[n];
    for (int i = 0; i < n; i++)
    {
        b[i] = new int[n];
        for (int j = 0; j < n; j++) b[i][j] = i + j;
    }
    
    int *a = new int[n];
    int *sum = new int[n];
    for (int i = 0; i < n; i++) a[i] = i;

    // 2. 准备计时器
    LARGE_INTEGER freq, head, tail;
    QueryPerformanceFrequency(&freq);
    
    double *record = new double[k];
    
    cout << "Testing n = " << n << " (Unroll) ... ";

    // 3. 开始总计时
    QueryPerformanceCounter(&head);

    for (int iter = 0; iter < k; iter++)
    {
        LARGE_INTEGER tick_h, tick_t;
        QueryPerformanceCounter(&tick_h);

        // --- 核心逻辑替换：循环展开 (Unroll by 10) ---
        // 初始化 sum 数组
        for (int i = 0; i < n; i++) sum[i] = 0;

        // 只有当剩余元素足够 10 个时才进行展开计算，防止越界
        for (int j = 0; j + 9 < n; j += 10)
        {
            int tmp0 = 0, tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0, 
                tmp5 = 0, tmp6 = 0, tmp7 = 0, tmp8 = 0, tmp9 = 0;

            for (int i = 0; i < n; i++)
            {
                tmp0 += a[j + 0] * b[j + 0][i];
                tmp1 += a[j + 1] * b[j + 1][i];
                tmp2 += a[j + 2] * b[j + 2][i];
                tmp3 += a[j + 3] * b[j + 3][i];
                tmp4 += a[j + 4] * b[j + 4][i];
                tmp5 += a[j + 5] * b[j + 5][i];
                tmp6 += a[j + 6] * b[j + 6][i];
                tmp7 += a[j + 7] * b[j + 7][i];
                tmp8 += a[j + 8] * b[j + 8][i];
                tmp9 += a[j + 9] * b[j + 9][i];
            }
            sum[j + 0] = tmp0;
            sum[j + 1] = tmp1;
            sum[j + 2] = tmp2;
            sum[j + 3] = tmp3;
            sum[j + 4] = tmp4;
            sum[j + 5] = tmp5;
            sum[j + 6] = tmp6;
            sum[j + 7] = tmp7;
            sum[j + 8] = tmp8;
            sum[j + 9] = tmp9;
        }

        // 处理不足 10 个的剩余部分（余数处理）
        for (int j = (n / 10) * 10; j < n; j++) {
            int tmp = 0;
            for (int i = 0; i < n; i++) {
                tmp += a[j] * b[j][i];
            }
            sum[j] = tmp;
        }

        QueryPerformanceCounter(&tick_t);
        record[iter] = (double)(tick_t.QuadPart - tick_h.QuadPart) * 1000.0 / freq.QuadPart;
    }

    QueryPerformanceCounter(&tail);

    // 4. 统计并存储结果
    double totalTime = (double)(tail.QuadPart - head.QuadPart) * 1000.0 / freq.QuadPart;
    cout << "Avg: " << totalTime / k << " ms" << endl;

    resultsTable[index] = new double[k];
    for (int i = 0; i < k; i++) resultsTable[index][i] = record[i];

    // 5. 释放当前规模的内存
    delete[] a;
    delete[] sum;
    delete[] record;
    for (int i = 0; i < n; i++) delete[] b[i];
    delete[] b;
}

// 存储数据
void save_data(const string &filename, double **arr, int rows, int cols)
{
    ofstream out_csv(filename);
    if (!out_csv.is_open())
    {
        cerr << "文件打开失败: " << filename << endl;
        return;
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            out_csv << arr[i][j];
            if (j < cols - 1) out_csv << ",";
        }
        out_csv << endl;
    }

    out_csv.close();
    cout << "文件写入成功: " << filename << endl;
}

int main()
{
    cout << "*********** Performance Test: Loop Unrolling Way ***********" << endl;
    cout << "------------------------------------------------------------" << endl;

    const int testCount = 51;
    const int iterations = 1000; 
    
    int n[testCount];
    for (int i = 0; i < testCount; i++)
    {
        n[i] = i * 100;
    }
    
    double **dataMatrix = new double *[testCount];

    for (int i = 0; i < testCount; i++)
    {
        run_unroll(n[i], iterations, dataMatrix, i);
    }

    // 导出数据
    save_data("unroll_results.csv", dataMatrix, testCount, iterations);

    cout << "\n测试完成。按下回车键退出...";
    cin.get();

    for (int i = 0; i < testCount; ++i)
    {
        delete[] dataMatrix[i];
    }
    delete[] dataMatrix;

    return 0;
}