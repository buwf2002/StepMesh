#!/bin/bash
# run_demo_zmq.sh (Physical Isolation Fixed)
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
function cleanup() {
    echo "kill all testing process of ps lite for user $USER"
    # pkill -9 -f test_bench
    pkill -9 -f test_remote_moe
    pkill -9 -f test_fserver
    sleep 1
}
trap cleanup EXIT

pkill -f -9 "VLLM*"
pkill -f -9 python3
pkill -9 -f python
pkill -9 -f vllm
pkill -9 -f fserver

export DMLC_INTERFACE="ibp1s0"
export SCHEDULER_IP=$(ip -o -4 addr | grep ${DMLC_INTERFACE} | awk '{print $4}' | cut -d'/' -f1)

export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=$SCHEDULER_IP  # scheduler's RDMA interface IP 
export DMLC_PS_ROOT_PORT=8123     # scheduler's port (can random choose)
export DMLC_ENABLE_RDMA=zmq
export DMLC_USE_RDMA=1

export DMLC_NODE_HOST=${SCHEDULER_IP}
# export DMLC_INTERFACE=auto
export STEPMESH_SPLIT_QP_LAG=0
export STEPMESH_BIND_CPU_CORE=0
export STEPMESH_GPU=0
export PS_VERBOSE=2

export DMLC_RANK=0

export STEPMESH_CPU_START_OFFSET=10

echo ">>> [1/2] Starting FFN Server on PHYSICAL GPU 0"

CUDA_VISIBLE_DEVICES=0,1 HIP_VISIBLE_DEVICES=0,1 \
DMLC_ROLE=server nohup numactl -m 0 python3 ffn.py > logs/ffn_stepmesh_connector.log 2>&1 &

echo "FFN PID: $!"
echo "Sleep 15s"
sleep 15s

export DMLC_ROLE=worker
export DMLC_RANK=0

echo ">>> [1/2] Starting Attn Server on PHYSICAL GPU 0"

CUDA_VISIBLE_DEVICES=2,3 HIP_VISIBLE_DEVICES=2,3 \
DMLC_ROLE=worker python3 attn.py 2>&1 | tee logs/attn_stepmesh_connector.log