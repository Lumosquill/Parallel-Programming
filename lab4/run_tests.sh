#!/bin/bash
# ============================================================
# Lab4 自动化测试脚本
# 用法: bash run_tests.sh
# 测试矩阵: 基础版 × (-O0/-O1/-O2) × (1/2/4/8进程) × (1000w/2000w/3000w)
# 结果保存在 ~/guess/results/ 目录下
# ============================================================

set -e

GUESS_DIR=~/guess
SRC_DIR=${GUESS_DIR}/lab4/base
MAIN=${GUESS_DIR}/main
RES_DIR=${GUESS_DIR}/results

mkdir -p ${RES_DIR}

# 生成 qsub_mpi.sh（参数：节点数, 每节点核数, 总进程数）
gen_qsub() {
    local NODES=$1 PPN=$2 NP=$3
    cat > ${GUESS_DIR}/qsub_mpi.sh << EOF
#!/bin/sh
#PBS -N qsub_mpi
#PBS -e test.e
#PBS -o test.o
#PBS -l nodes=${NODES}:ppn=${PPN}

NODES=\$(cat \$PBS_NODEFILE | sort | uniq)

for node in \$NODES; do
    scp master_ubss1:/home/\${USER}/guess/main \${node}:/home/\${USER} 1>&2
    scp -r master_ubss1:/home/\${USER}/guess/files \${node}:/home/\${USER}/ 1>&2
done

/usr/local/bin/mpiexec -np ${NP} -machinefile \$PBS_NODEFILE /home/\${USER}/main > /home/\${USER}/output.txt 2>&1

scp /home/\${USER}/output.txt master_ubss1:/home/\${USER}/guess/output.txt 2>&1
scp -r /home/\${USER}/files/ master_ubss1:/home/\${USER}/guess/ 2>&1
EOF
}

# 等待任务完成
wait_job() {
    local job_id=$1
    echo -n "  Waiting for job ${job_id}..."
    while qstat ${job_id} 2>/dev/null | grep -q " R \| Q "; do
        sleep 3
    done
    echo " Done"
}

# ============================================================
# 阶段1: 正确性验证 (只跑一次)
# ============================================================
echo "========== Phase 1: Correctness Check =========="
cd ${SRC_DIR}
mpic++ correctness_mpi.cpp train.cpp guessing_mpi.cpp md5.cpp -o ${MAIN} -O2 -std=c++11

gen_qsub 1 4 4
> ${GUESS_DIR}/test.o && > ${GUESS_DIR}/test.e && > ${GUESS_DIR}/output.txt
JOB_ID=$(qsub ${GUESS_DIR}/qsub_mpi.sh 2>&1 | grep -o '[0-9]*')
wait_job ${JOB_ID}
CRACKED=$(grep "Cracked:" ${GUESS_DIR}/output.txt | grep -o '[0-9]*')
echo "  Correctness (4 proc): Cracked = ${CRACKED}"
echo "${CRACKED}" > ${RES_DIR}/correctness.txt

# ============================================================
# 阶段2: 基础版 × 不同优化 × 不同进程数 × 2000w
# ============================================================
echo ""
echo "========== Phase 2: Base Version -O0/-O1/-O2 =========="

for OPT in O0 O1 O2; do
    echo "--- Optimization: -${OPT} ---"
    cd ${SRC_DIR}
    mpic++ main_mpi.cpp train.cpp guessing_mpi.cpp md5.cpp -o ${MAIN} -${OPT} -std=c++11

    for NP in 1 2 4 8; do
        # 计算 nodes 和 ppn
        if [ ${NP} -le 8 ]; then
            NODES=1; PPN=${NP}
        else
            NODES=2; PPN=8
        fi

        echo "  NP=${NP} (nodes=${NODES}, ppn=${PPN})"
        gen_qsub ${NODES} ${PPN} ${NP}
        > ${GUESS_DIR}/test.o && > ${GUESS_DIR}/test.e && > ${GUESS_DIR}/output.txt
        JOB_ID=$(qsub ${GUESS_DIR}/qsub_mpi.sh 2>&1 | grep -o '[0-9]*')
        wait_job ${JOB_ID}
        cp ${GUESS_DIR}/output.txt ${RES_DIR}/base_${OPT}_np${NP}.txt
        grep "Guess time:" ${GUESS_DIR}/output.txt | head -1
    done
done

# ============================================================
# 阶段3: 不同问题规模 (1000w, 3000w)
# ============================================================
echo ""
echo "========== Phase 3: Different Scales =========="

# 备份原 main_mpi.cpp
cp ${SRC_DIR}/main_mpi.cpp ${SRC_DIR}/main_mpi.cpp.bak

for LIMIT in 10000000 30000000; do
    SCALE=$(( LIMIT / 10000 ))
    echo "--- Scale: ${SCALE}w ---"

    # 修改 LIMIT
    sed -i "s/const long long LIMIT = [0-9]*/const long long LIMIT = ${LIMIT}/" ${SRC_DIR}/main_mpi.cpp
    cd ${SRC_DIR}
    mpic++ main_mpi.cpp train.cpp guessing_mpi.cpp md5.cpp -o ${MAIN} -O2 -std=c++11

    for NP in 1 4 8; do
        if [ ${NP} -le 8 ]; then NODES=1; PPN=${NP}; else NODES=2; PPN=8; fi
        echo "  NP=${NP} (nodes=${NODES}, ppn=${PPN})"
        gen_qsub ${NODES} ${PPN} ${NP}
        > ${GUESS_DIR}/test.o && > ${GUESS_DIR}/test.e && > ${GUESS_DIR}/output.txt
        JOB_ID=$(qsub ${GUESS_DIR}/qsub_mpi.sh 2>&1 | grep -o '[0-9]*')
        wait_job ${JOB_ID}
        cp ${GUESS_DIR}/output.txt ${RES_DIR}/base_O2_scale${SCALE}w_np${NP}.txt
        grep "Guess time:" ${GUESS_DIR}/output.txt | head -1
    done
done

# 恢复
mv ${SRC_DIR}/main_mpi.cpp.bak ${SRC_DIR}/main_mpi.cpp

# ============================================================
# 结果汇总
# ============================================================
echo ""
echo "========== All Results =========="
echo "Results saved in: ${RES_DIR}/"
ls -la ${RES_DIR}/
echo ""
echo "Summary (Guess times):"
for f in ${RES_DIR}/base_*.txt; do
    NAME=$(basename $f .txt)
    GT=$(grep "Guess time:" $f | grep -o '[0-9.]*' | head -1)
    echo "  ${NAME}: ${GT}s"
done
