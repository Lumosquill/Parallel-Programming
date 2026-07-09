#!/bin/bash
# 一键生成并提交所有对比实验
# 用法: bash run_all.sh

cd ~/guess/final_lab

echo "=== 1. 编译不同优化等级 ==="
mpic++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon.cpp -o ../main_O0 -O0 -fopenmp -std=c++11 -march=native
mpic++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon.cpp -o ../main_O1 -O1 -fopenmp -std=c++11 -march=native
echo "编译完成"

echo "=== 2. 生成提交脚本 ==="

# 辅助函数: 写qsub脚本
write_qsub() {
    local name=$1 np=$2 omp=$3 mode=$4 bin=$5
    cat > ~/guess/qsub_${name}.sh << EOF
#!/bin/sh
#PBS -N ${name}
#PBS -e test.e
#PBS -o test_${name}.o
#PBS -l nodes=1:ppn=8

NODES=\$(cat \$PBS_NODEFILE | sort | uniq)
for node in \$NODES; do
    ssh \${node} "mkdir -p /home/\${USER}" 1>&2
    scp master_ubss1:/home/\${USER}/guess/${bin} \${node}:/home/\${USER}/main 1>&2
    scp -r master_ubss1:/home/\${USER}/guess/files \${node}:/home/\${USER}/ 1>&2
done

export OMP_NUM_THREADS=${omp}
/usr/local/bin/mpiexec -np ${np} -machinefile \$PBS_NODEFILE /home/\${USER}/main ${mode}
EOF
    chmod +x ~/guess/qsub_${name}.sh
}

# ---- 扩展性实验: Mode 2, 不同 进程×线程 ----
write_qsub "s1_8x1"  8 1 2 "main"     # 8进程×1线程
write_qsub "s2_2x4"  2 4 2 "main"     # 2进程×4线程
write_qsub "s3_1x8"  1 8 2 "main"     # 1进程×8线程

# ---- 线程数实验: Mode 2, 固定4进程 ----
write_qsub "t1_4x1"  4 1 2 "main"     # 4进程×1线程
write_qsub "t3_4x4"  4 4 2 "main"     # 4进程×4线程

# ---- 编译优化: Mode 2, 4×2, 不同二进制 ----
write_qsub "o0_O0"   4 2 2 "main_O0"  # -O0
write_qsub "o1_O1"   4 2 2 "main_O1"  # -O1

echo "脚本生成完成"

echo "=== 3. 提交所有任务 ==="
for s in s1_8x1 s2_2x4 s3_1x8 t1_4x1 t3_4x4 o0_O0 o1_O1; do
    qsub ~/guess/qsub_${s}.sh
    echo "  已提交 ${s}"
done

echo ""
echo "=== 全部提交完毕 ==="
echo "查看状态: qstat"
echo "查看结果: cat ~/guess/test_{实验名}.o"
echo ""
echo "输出文件列表:"
echo "  test_s1_8x1.o   8进程×1线程"
echo "  test_s2_2x4.o   2进程×4线程"
echo "  test_s3_1x8.o   1进程×8线程"
echo "  test_t1_4x1.o   4进程×1线程"
echo "  test_t3_4x4.o   4进程×4线程"
echo "  test_o0_O0.o    -O0 优化"
echo "  test_o1_O1.o    -O1 优化"
