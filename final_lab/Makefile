# ============================================================
#  final_lab Makefile
#  MPI + OpenMP + SIMD 三层混合并行口令猜测
# ============================================================

# 编译器
CXX = mpic++
CXXFLAGS = -O2 -std=c++11 -fopenmp

# ARM (NEON) 编译：默认
ARCH_FLAGS_ARM = -march=armv8-a
# x86 (SSE) 编译：make ARCH=x86
ARCH_FLAGS_X86 = -msse4.1

# 默认 ARM
ARCH ?= arm
ifeq ($(ARCH), arm)
    ARCH_FLAGS = $(ARCH_FLAGS_ARM)
    SIMD_SRC   = md5_neon.cpp
else
    ARCH_FLAGS = $(ARCH_FLAGS_X86)
    SIMD_SRC   = md5_sse.cpp
endif

# 源文件
SRCS = main.cpp train.cpp guessing.cpp md5.cpp $(SIMD_SRC)
OBJS = $(SRCS:.cpp=.o)

# 目标
TARGET = final_hybrid

# 默认目标
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(ARCH_FLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(ARCH_FLAGS) -c -o $@ $<

# 清理
clean:
	rm -f $(OBJS) $(TARGET)

# 运行（默认4进程）
run:
	mpirun -np 4 ./$(TARGET) 2

# 对比测试：三种模式
test:
	@echo "=== Mode 0: Pure MPI ==="
	mpirun -np 4 ./$(TARGET) 0
	@echo ""
	@echo "=== Mode 1: MPI + OpenMP ==="
	mpirun -np 4 ./$(TARGET) 1
	@echo ""
	@echo "=== Mode 2: MPI + OpenMP + SIMD ==="
	mpirun -np 4 ./$(TARGET) 2

.PHONY: all clean run test
