THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
function cleanup() {
    echo "kill all testing process of ps lite for user $USER"
    # pkill -9 -f test_bench
    pkill -9 -f python3
    sleep 1
}
trap cleanup EXIT
# cleanup

export BIN=${BIN:-test_fserver}
# common setup
export DMLC_INTERFACE=${RNIC:-ibp1s0}
export SCHEDULER_IP=$(ip -o -4 addr | grep ${DMLC_INTERFACE} | awk '{print $4}' | cut -d'/' -f1)
export DMLC_NUM_WORKER=${NUM_WORKER:-1}
export DMLC_NUM_SERVER=${NUM_SERVER:-1}
export DMLC_GROUP_SIZE=2 #这是每个worker或者serve有多少个process
export DMLC_NODE_RANK=${NODE_RANK:-0}
export DMLC_PS_ROOT_PORT=8123
export DMLC_PS_ROOT_URI=$SCHEDULER_IP  # scheduler's RDMA interface IP
export DMLC_ENABLE_RDMA=zmq
export NCCL_DEBUG=warning
export STEPMESH_SPLIT_QP_LAG=0
export STEPMESH_BIND_CPU_CORE=1

export PS_VERBOSE=0

ROLE=${ROLE:-server}
if [ $ROLE == "server" ]; then
  echo "Run server and scheduler, scheduler ip $SCHEDULER_IP "
  export DMLC_NODE_HOST=${SCHEDULER_IP}
  DMLC_ROLE=scheduler DMLC_LOCAL=0 python3 $THIS_DIR/${BIN}.py &

  sleep 1 # wait scheduler

  #export DMLC_INTERFACE=auto
  for P in {0..2}; do
    DMLC_ROLE=server STEPMESH_GPU=${P}  python3 $THIS_DIR/${BIN}.py $@ &
  done
elif [ $ROLE == "worker" ]; then
  echo "Run worker with scheduler ip: $1"
  export DMLC_PS_ROOT_URI=$1
  #export DMLC_INTERFACE=auto
  export DMLC_NODE_HOST=${SCHEDULER_IP}
  for P in {0..2}; do
    DMLC_ROLE=worker STEPMESH_GPU=${P}  python3 $THIS_DIR/${BIN}.py "${@:2}" &
  done
elif [ $ROLE == "server-slave" ]; then
  echo "Run server with scheduler ip: $1"
  export DMLC_PS_ROOT_URI=$1
  #export DMLC_INTERFACE=auto
  export DMLC_NODE_HOST=${SCHEDULER_IP}
  for P in {0..2}; do
    DMLC_ROLE=server STEPMESH_GPU=${P}  python3 $THIS_DIR/${BIN}.py "${@:2}" &
  done
elif [ $ROLE == "joint" ]; then
  echo "Run scheduler, server, and worker jointly"
  export DMLC_NODE_HOST=${SCHEDULER_IP}
  export DMLC_PS_ROOT_URI=$SCHEDULER_IP
  DMLC_ROLE=scheduler numactl -m 0 python3 $THIS_DIR/${BIN}.py &
  PIDS=($!)
  echo "[Test]Start scheduler $PIDS"

  sleep 1
  export STEPMESH_CPU_START_OFFSET=10
  for P in {0..1}; do
    DMLC_ROLE=server STEPMESH_GPU=${P} numactl -m 0 python3 $THIS_DIR/${BIN}.py &
    PIDS=($!)
    echo "[Test]Start server $PIDS"
  done

  export STEPMESH_CPU_START_OFFSET=15
  for P in {0..1}; do
    DMLC_ROLE=worker STEPMESH_GPU=${P} numactl -m 0 python3 $THIS_DIR/${BIN}.py &
    PIDS=($!)
    echo "[Test]Start worker $PIDS"
  done
    
  # Wait for each PID and report status
  for pid in "${PIDS[@]}"; do
    wait $pid
    echo "PID $pid exited with status $?"
  done
else
  wait
fi
