# StepMesh 测试环境变量说明

StepMesh 的测试涉及多个环境变量的设置，相互影响。这里介绍常见环境变量的作用。

---

**`STEPMESH_BACKEND`**：StepMesh 支持的 backend，包括 `CPU`、`GPU`、`DCU`（DCU 由 GPU backend 迁移到 HIP API）。

---

**`DMLC_ENABLE_RDMA`**：支持 `zmq`、`ibverbs`；`ibverbs` 表示使用 RDMA。

**`DMLC_INTERFACE`**：网卡名称，可由 `ip link show` 查看。

---

**`DMLC_NUM_WORKER`**：worker 节点的个数（多少个物理机用于放置 worker）。

**`DMLC_NUM_SERVER`**：server 节点的个数（多少个物理机用于放置 server）。

> 单机多卡时，`DMLC_NUM_WORKER = DMLC_NUM_SERVER = 1`。

**`DMLC_GROUP_SIZE`**：每个 worker 或者 server 节点有多少个 worker/server 进程。

> 例如 `DMLC_GROUP_SIZE = 4`，表示一个物理节点有 4 个 worker/server（8 个 GPU）。

---

**`DMLC_NODE_RANK`**：表示在第几个物理机节点。注意：不允许超过 `DMLC_NUM_WORKER` 或 `DMLC_NUM_SERVER`，否则无法正确建立连接。

**`STEPMESH_GPU`**：告诉 StepMesh 当前的 worker/server 是当前节点的第几个进程（类似 local rank），可根据这个在 Python 里面控制 tensor 在 GPU 的位置。

**`DMLC_NODE_HOST`**：用于指定当前节点在分布式系统中的 IP 地址。一般和 scheduler ID 一样（单机多卡时）。

---

**`DMLC_ROLE`**：

StepMesh 的通信由 `scheduler`、`worker`、`server` 三个角色组成。在 AFD 里面：
- `worker` 表示 Attention Rank
- `server` 表示 FFN Rank
- `worker` 和 `server` 通过相同的 `scheduler` 进行连接

一般一个程序只有一个 `scheduler`。一个 `worker`/`server` 一般代表一个 GPU，`scheduler` 不需要 GPU。

`ps.init()` 启动 `scheduler`、`worker`、`server`：

| 配置 | 说明 |
|------|------|
| `DMLC_ROLE=scheduler` | 启动 scheduler，打开 scheduler IP 和 port，供 worker 和 server 连接 |
| `DMLC_ROLE=worker` | 启动一个 worker，连接到 scheduler |
| `DMLC_ROLE=server` | 启动一个 server，连接到 scheduler |

> **注意**：必须所有的 worker 和 server 正常启动并连接到 scheduler，才能正常进行数据传输。

---

**`DMLC_PS_ROOT_URI`**：记录 scheduler 的 IP，供 worker 和 server 找到 scheduler 位置。

**`DMLC_PS_ROOT_PORT`**：记录 scheduler 的 port。

---
