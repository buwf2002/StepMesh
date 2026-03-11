import torch, os
import time
import fserver_lib as f

is_worker = os.environ.get('DMLC_ROLE') == 'worker'
is_server = os.environ.get('DMLC_ROLE') == 'server'
gpu = os.environ.get('STEPMESH_GPU', '0')

# Each GPU pair (server gpu, worker gpu) runs independently
# Set DMLC_NUM_WORKER=1 so each server handles one worker
os.environ['DMLC_NUM_WORKER'] = '1'
os.environ['DMLC_GROUP_SIZE'] = '1'

print(f"DEBUG: role={os.environ.get('DMLC_ROLE')}, gpu={gpu}, DMLC_NUM_WORKER={os.environ.get('DMLC_NUM_WORKER')}, DMLC_GROUP_SIZE={os.environ.get('DMLC_GROUP_SIZE')}")

f.init()

if is_worker:
    push_tensors = [
        torch.rand([1, 8192], dtype=torch.float32, device=f'cuda:{gpu}'),
        torch.rand([1, 8192], dtype=torch.float32, device=f'cuda:{gpu}'),
        torch.rand([1, 8192], dtype=torch.float32, device=f'cuda:{gpu}'),
    ]
    pull_tensors = [
        torch.rand([1, 8192], dtype=torch.float32, device=f'cuda:{gpu}')
    ]
    # Use gpu-specific keys to avoid collision
    keys_offset = int(gpu) * 100
    handler = f.push_pull(
        push_tensors,
        [i + keys_offset for i in range(len(push_tensors))],
        pull_tensors,
        [i + keys_offset for i in range(len(pull_tensors))]
    )
    f.wait(handler)
    print(f"worker {gpu} test done")

elif is_server:
    torch.set_default_device('cuda:{}'.format(gpu))
    res = []
    while len(res) == 0:
        time.sleep(0.1)
        res = f.get_batch()
    print(f"server {gpu} received: {res}")
    for r in res:
        comm_id, batch, _ = r
        f.respond([sum(batch)], comm_id)
    print(f"server {gpu} responded")

f.stop()
