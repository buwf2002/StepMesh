# StepMesh: A High-Performance, Low-Latency Communication Library for Attention-FFN Disaggregation

This repository is forked from [StepMesh](https://github.com/stepfun-ai/StepMesh) and has been modified to support SHCA (Tianlong NIC) and AMD.

## Getting Started

### Prerequisites

- Ubuntu OS
- Servers with RDMA NICs and GPU cards.
- torch >= 2.0.0
- CUDA NVCC or other compilers

## Build

Download code and install dependencies
```bash
git clone https://github.com/buwf2002/StepMesh.git
cd StepMesh
bash tools/install_deps.sh # only once
```

Build StepMesh with hip

```bash

# Build AF library
USE_CUDA=0 USE_HIP=1 make af
# Build and install Fserver （AF's Python SDK）
USE_CUDA=0 USE_HIP=1 pip3 install -v -e . --no-build-isolation
```

## Tutorial

After build, you will have testing applications under `tests/` dir. 
Below we elaborate how you can run with them. 

To debug, set `PS_VERBOSE=1` to see important logs during connection setup, and `PS_VERBOSE=2` to see each message log.

### 1. Single-Node Example

- Single GPU Example: Suppose you want to run with 1 worker and 1 server on same server.

```bash
export STEPMESH_BAKCEND=DCU
# ROLE: jointly run scheduler, worker and server; RNIC: your first rdma nic; 
ROLE=joint RNIC=ib0 bash tests/fserver/run_single_gpu.sh
```
- Multiple GPU Example: Suppose you want to run with 8 workers and 8 servers on different GPUs of the same server.
```bash
export STEPMESH_BAKCEND=DCU
# ROLE: jointly run scheduler, worker and server; RNIC: your first rdma nic; 
ROLE=joint RNIC=ib0 bash tests/fserver/run_multi_gpu.sh
```

### 2. Two-Node Example
- Run scheduler and servers
```bash
export STEPMESH_BAKCEND=DCU
# server: RNIC: your first rdma nic; 
ROLE=server RNIC=ib0 bash tests/fserver/run_multi_gpu.sh
# the first line will print scheduler ip used for worker
```

- Run workers
```bash
export STEPMESH_BAKCEND=DCU
# worker: RNIC: your first rdma nic; 
ROLE=worker RNIC=ib0 bash tests/fserver/run_multi_gpu.sh ${scheduler ip}
```

For more test cases and examples, please refer to [tests](./tests).

For more documents, please refer to [docs](./docs).

For more details, please refer to the [**Step-3 system technical report**](https://arxiv.org/abs/2507.19427) and our [**Introduction**](Introduction.md)([Chinese Version](Introduction_cn.md)).
