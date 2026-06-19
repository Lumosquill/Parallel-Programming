# ============================================================
# lab5 GPU 口令猜测 — Makefile
# 四个版本: basic / batch / pipeline / adaptive
# ============================================================
NVCC       = nvcc
ARCH       = sm_70
STD        = -std=c++14
NVFLAGS    = -arch=$(ARCH) $(STD) -O2
LDFLAGS    = -lcudart

COMMON_DIR = common
COMMON_SRC = $(COMMON_DIR)/md5.cpp $(COMMON_DIR)/train.cpp

# ──────────────────────────────────────
# v1: 基础单PT GPU
# ──────────────────────────────────────
V1_DIR  = v1_gpu_basic
v1: $(V1_DIR)/main
$(V1_DIR)/main: $(V1_DIR)/main.cu $(V1_DIR)/guessing_gpu.cu $(COMMON_SRC)
	$(NVCC) $(NVFLAGS) $^ -o $@ $(LDFLAGS)

# ──────────────────────────────────────
# v2: 多PT批处理
# ──────────────────────────────────────
V2_DIR  = v2_gpu_batch
v2: $(V2_DIR)/main
$(V2_DIR)/main: $(V2_DIR)/main.cu $(V2_DIR)/guessing_gpu.cu $(COMMON_SRC)
	$(NVCC) $(NVFLAGS) $^ -o $@ $(LDFLAGS)

# ──────────────────────────────────────
# v3: CPU/GPU重叠 (异步流水线)
# ──────────────────────────────────────
V3_DIR  = v3_gpu_pipeline
v3: $(V3_DIR)/main
$(V3_DIR)/main: $(V3_DIR)/main.cu $(V3_DIR)/guessing_gpu.cu $(COMMON_SRC)
	$(NVCC) $(NVFLAGS) $^ -o $@ $(LDFLAGS)

# ──────────────────────────────────────
# v4: 自适应负载均衡 (全进阶)
# ──────────────────────────────────────
V4_DIR  = v4_gpu_adaptive
v4: $(V4_DIR)/main
$(V4_DIR)/main: $(V4_DIR)/main.cu $(V4_DIR)/guessing_gpu.cu $(COMMON_SRC)
	$(NVCC) $(NVFLAGS) $^ -o $@ $(LDFLAGS)

# ──────────────────────────────────────
# 快捷目标
# ──────────────────────────────────────
all: v1 v2 v3 v4

clean:
	rm -f $(V1_DIR)/main $(V2_DIR)/main $(V3_DIR)/main $(V4_DIR)/main

run-v1: v1
	$(V1_DIR)/main

run-v2: v2
	$(V2_DIR)/main

run-v3: v3
	$(V3_DIR)/main

run-v4: v4
	$(V4_DIR)/main

.PHONY: all clean v1 v2 v3 v4 run-v1 run-v2 run-v3 run-v4
