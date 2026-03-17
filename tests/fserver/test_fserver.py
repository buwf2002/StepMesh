import torch, os
import time
import fserver_lib as f

is_worker = os.environ.get('DMLC_ROLE') == 'worker'
is_server = os.environ.get('DMLC_ROLE') == 'server'

f.init()

if is_worker:
    gpu = int(os.environ.get('STEPMESH_GPU'))

    push_tensors = [
        torch.rand([1, 8192], dtype=torch.float32, device=f'cuda:{gpu}'),
        torch.rand([1, 8192], dtype=torch.float32, device=f'cuda:{gpu}'),
        torch.rand([1, 8192], dtype=torch.float32, device=f'cuda:{gpu}'),
    ]
    pull_tensors = [
        torch.rand([1, 8192], dtype=torch.float32, device=f'cuda:{gpu}')
    ]
    handler = f.push_pull(
        push_tensors, 
        [i for i in range(len(push_tensors))], 
        pull_tensors, 
        [i for i in range(len(pull_tensors))]
    )
    f.wait(handler)
    sum_tensor = torch.stack(push_tensors).sum(dim=0)
    print(f"A Push: {push_tensors}")
    print(f"A Recv: {pull_tensors[0]}")
    assert torch.allclose(sum_tensor, pull_tensors[0])
    print(f"{gpu} worker test done")

elif is_server:
    gpu = int(os.environ.get('STEPMESH_GPU'))
    worker_gpu = int(os.environ.get('DMLC_GROUP_SIZE', 0))
    gpu += worker_gpu # 单 node 上 multi gpu 测试需要让 server 和 worker 使用不同的gpu
    torch.set_default_device('cuda:{}'.format(gpu))

    res = []
    while len(res) == 0:
        time.sleep(1)
        res = f.get_batch()
    print(f"F Recv: {res}")
    for r in res:
        comm_id, batch, _ = r
        res = torch.stack(batch).sum(dim=0)
        print(f"F Push: {res}")
        torch.cuda.synchronize() # 确保计算完再发送
        f.respond([res], comm_id)

f.stop()
