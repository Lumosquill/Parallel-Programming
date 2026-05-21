#include "md5.h"
#include "md5_neon.h"
#include <iostream>
#include <iomanip>
using namespace std;

void printHash(const bit32 s[4]) {
    for (int i = 0; i < 4; i++)
        cout << setw(8) << setfill('0') << hex << s[i];
    cout << dec << "\n";
}

int main() {
    string inputs[4] = {"", "abc", "123456", "hello_world!"};
    
    // 串行结果
    cout << "串行 MD5：" << endl;
    bit32 ser[4][4];
    for (int i = 0; i < 4; i++) {
        MD5Hash(inputs[i], ser[i]);
        cout << "  [" << i << "] \"" << inputs[i] << "\": ";
        printHash(ser[i]);
    }

    // SIMD结果
    cout << "\nNEON SIMD MD5：" << endl;
    bit32 sim[4][4];
    MD5Hash_SIMD4(inputs, sim);
    for (int i = 0; i < 4; i++) {
        cout << "  [" << i << "] \"" << inputs[i] << "\": ";
        printHash(sim[i]);
    }

    // 对比
    cout << "\n验证结果：" << endl;
    bool all_pass = true;
    for (int i = 0; i < 4; i++) {
        bool ok = true;
        for (int j = 0; j < 4; j++)
            if (ser[i][j] != sim[i][j]) ok = false;
        cout << "  [" << i << "] " << (ok ? "PASS" : "FAIL") << endl;
        if (!ok) all_pass = false;
    }
    cout << (all_pass ? "\n全部通过 ✓" : "\n存在错误 ✗") << endl;
    return all_pass ? 0 : 1;
}