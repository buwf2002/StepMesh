#!/bin/bash
# run_stepmesh_connector.sh - Test between ffn.py and attn.py
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

function cleanup() {
    echo "kill all testing process of ps lite for user $USER"
    pkill -9 -f test_remote_moe
    pkill -9 -f test_fserver
    pkill -f -9 "VLLM*"
    pkill -f -9 python3
    pkill -9 -f python
    pkill -9 -f vllm
    pkill -9 -f fserver
    pkill -9 -f ffn.py
    pkill -9 -f attn.py
    sleep 1
}

cleanup

trap cleanup EXIT

export STEPMESH_BAKCEND=DCU
# export PYTHONPATH=/workspace/my_vllm_source:$PYTHONPATH

export DMLC_INTERFACE=${RNIC:-ibp1s0}
export SCHEDULER_IP=$(ip -o -4 addr | grep ${DMLC_INTERFACE} | awk '{print $4}' | cut -d'/' -f1)
export DMLC_NUM_WORKER=${NUM_WORKER:-1}
export DMLC_NUM_SERVER=${NUM_SERVER:-1}
export DMLC_GROUP_SIZE=2
export DMLC_NODE_RANK=${NODE_RANK:-0}
export DMLC_PS_ROOT_PORT=8123
export DMLC_PS_ROOT_URI=$SCHEDULER_IP
export DMLC_ENABLE_RDMA=zmq
export NCCL_DEBUG=warning
export STEPMESH_SPLIT_QP_LAG=0
export STEPMESH_BIND_CPU_CORE=1

export PS_VERBOSE=2

# Ensure logs directory exists
mkdir -p logs

echo "Starting ffn.py (server) instance..."

# Start ffn.py (server role)
export STEPMESH_CPU_START_OFFSET=10
DMLC_ROLE=server  numactl -m 0 torchrun --master_port=29500 --nproc_per_node=2 ffn.py > logs/ffn.log 2>&1 &
FFN_PID=$!
echo "Started ffn.py (PID: $FFN_PID)"

# Wait for ffn.py to initialize
echo "Waiting 15 seconds for ffn.py to initialize..."
sleep 10

echo "Starting attn.py (worker) instance..."

# Start attn.py (worker role)
export STEPMESH_CPU_START_OFFSET=15

DMLC_ROLE=worker numactl -m 0 torchrun --master_port=29501 --nproc_per_node=2 attn.py > logs/attn.log 2>&1

ATTN_PID=$!
echo "Started attn.py (PID: $ATTN_PID)"

echo "All processes started. Waiting..."
echo "Logs: logs/ffn.log, logs/attn.log"
echo "Press Ctrl+C to stop"

# Wait for all background processes
wait
