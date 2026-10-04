"""One GPU inference server for many CPU search processes.

Tree search is Python code and needs many CPU cores, while the network needs the GPU. Giving every search process its
own CUDA context doesn't work well: the processes take turns on the GPU (a context switch each time), each context
costs over 1 GB of RAM, and the many small batches keep the GPU busy. Instead:

- **workers** (CPU only) run the searches. When a worker needs a batch of leaves evaluated, it writes the planes into
  a shared-memory buffer, puts `(worker, slot, network id, batch size, shape)` on a request queue, and waits;
- the **server** (the main process, which owns the networks) collects the waiting requests, evaluates them in one batch
  per network, writes the probabilities and values back into the workers' buffers, and wakes them up.

A batch of a few hundred positions costs the GPU about as much as a batch of a few, so the server waits briefly for
the other busy workers before evaluating. And each worker has `SLOTS` buffers: it splits its games into groups that
take turns, so it searches one group while the other group's leaves are on the GPU (`drive_interleaved`).
"""

import multiprocessing as mp
import queue
import sys
import time
from contextlib import contextmanager
from multiprocessing import shared_memory

import numpy as np

MAX_CELLS = 15 * 15
MAX_PLANES = 5
SLOTS = 2  # buffers per worker, i.e. groups of games one worker interleaves


class Channels:
    """Shared buffers, the request queue and one semaphore per buffer (created by the server, passed to workers)."""

    def __init__(self, ctx, n_workers: int, max_batch: int):
        self.n_workers, self.max_batch = n_workers, max_batch
        self.requests = ctx.Queue()
        self.ready = [[ctx.Semaphore(0) for _ in range(SLOTS)] for _ in range(n_workers)]
        self.ids = ctx.Queue()
        for w in range(n_workers):
            self.ids.put(w)
        size = max_batch * (MAX_PLANES * MAX_CELLS + MAX_CELLS + 4 * MAX_CELLS + 4)
        self.memory = [[shared_memory.SharedMemory(create=True, size=size) for _ in range(SLOTS)]
                       for _ in range(n_workers)]
        self.names = [[m.name for m in row] for row in self.memory]

    def __getstate__(self):  # what a worker receives: the names of the buffers, not the buffers
        state = self.__dict__.copy()
        state['memory'] = None
        return state

    def views(self, memory: shared_memory.SharedMemory):
        """numpy views (obs uint8, mask bool, probs float32, values float32) into one buffer."""
        b = self.max_batch
        offset, out = 0, []
        for dtype, shape in ((np.uint8, (b, MAX_PLANES * MAX_CELLS)), (np.bool_, (b, MAX_CELLS)),
                             (np.float32, (b, MAX_CELLS)), (np.float32, (b,))):
            arr = np.ndarray(shape, dtype, memory.buf, offset)
            offset += arr.nbytes
            out.append(arr)
        return out

    def close(self):
        for row in self.memory:
            for m in row:
                m.close()
                m.unlink()


class RemoteEvaluator:
    """Worker side: evaluates batches on the server. Calling it is `submit` then `wait`, like `nets.NetEvaluator`."""

    def __init__(self, net_id: str, slot: int = 0):
        channels, self.worker = _WORKER['channels'], _WORKER['id']
        self.channels, self.net_id, self.slot = channels, net_id, slot
        self.obs, self.mask, self.probs, self.values = channels.views(_WORKER['memory'][slot])
        self.ready = channels.ready[self.worker][slot]
        self.n = self.cells = 0

    def submit(self, obs: np.ndarray, masks: np.ndarray):
        n, self.cells = len(obs), masks.shape[1]
        if n > self.channels.max_batch:
            raise ValueError(f'batch of {n} exceeds max_batch {self.channels.max_batch}')
        flat = obs.reshape(n, -1)
        self.obs[:n, :flat.shape[1]] = flat
        self.mask[:n, :self.cells] = masks
        self.n = n
        self.channels.requests.put((self.worker, self.slot, self.net_id, n, obs.shape[1:]))

    def wait(self) -> tuple[np.ndarray, np.ndarray]:
        while not self.ready.acquire(timeout=5):
            if not mp.parent_process().is_alive():  # the server is gone: don't wait forever
                raise SystemExit('inference server died')
        return self.probs[:self.n, :self.cells].copy(), self.values[:self.n].copy()

    def __call__(self, obs: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.submit(obs, masks)
        return self.wait()


# ------------------------------------------------------------------------------------- worker process

_WORKER = {}


def init_worker(channels: Channels):
    """Pool initializer: claim a worker id and attach to its shared buffers."""
    w = channels.ids.get()
    _WORKER.update(channels=channels, id=w, memory=[shared_memory.SharedMemory(name=n) for n in channels.names[w]])


def evaluator(net_id: str) -> RemoteEvaluator:
    """In a worker: an evaluator for the server's network `net_id`."""
    return RemoteEvaluator(net_id)


def drive_interleaved(steps: list, net_id: str) -> list:
    """Run up to `SLOTS` request generators (like `PUCT.search_steps`) in turns, each with its own buffer:
    while one generator's batch is on the GPU, the next one computes its own. Returns their results."""
    if len(steps) > SLOTS:
        raise ValueError(f'at most {SLOTS} generators')
    evaluators = [RemoteEvaluator(net_id, slot) for slot in range(len(steps))]
    results, running = [None] * len(steps), []
    for k, gen in enumerate(steps):
        try:
            evaluators[k].submit(*next(gen))
            running.append(k)
        except StopIteration as stop:
            results[k] = stop.value
    while running:
        for k in list(running):
            try:
                evaluators[k].submit(*steps[k].send(evaluators[k].wait()))
            except StopIteration as stop:
                results[k] = stop.value
                running.remove(k)
    return results


# ------------------------------------------------------------------------------------- server


@contextmanager
def without_main_module():
    """Start processes without re-importing the parent's `__main__` in them.

    A spawned process normally imports the main module first. When that is `omok_rl.alphazero` (or a script), it imports
    torch, which costs ~460 MB of RAM per worker; the workers only run functions from torch-free modules."""
    main = sys.modules['__main__']
    saved = {k: main.__dict__.pop(k) for k in ('__spec__', '__file__') if k in main.__dict__}
    main.__spec__ = None
    try:
        yield
    finally:
        del main.__spec__
        main.__dict__.update(saved)


class Server:
    """Main-process side: a pool of CPU workers and the loop that answers their requests.

    `evaluators` maps network ids to `nets.NetEvaluator`s; it can be changed between calls to `run`."""

    def __init__(self, n_workers: int, max_batch: int = 1024, max_wait: float = 0.002):
        ctx = mp.get_context('spawn')
        self.channels = Channels(ctx, n_workers, max_batch)
        self.views = [[self.channels.views(m) for m in row] for row in self.channels.memory]
        with without_main_module():
            self.pool = ctx.Pool(n_workers, initializer=init_worker, initargs=(self.channels,))
        self.max_wait = max_wait  # seconds to wait for more requests before evaluating a batch
        self.evaluators = {}
        self.batches = self.positions = 0

    def run(self, fn, args_list: list[tuple]) -> list:
        """Run `fn(*args)` for every entry in the workers (fn must be importable), serving inference meanwhile."""
        jobs = [self.pool.apply_async(fn, args) for args in args_list]
        while True:
            busy = min(self.channels.n_workers, sum(not j.ready() for j in jobs))
            if not busy:
                break
            try:
                requests = [self.channels.requests.get(timeout=0.01)]
            except queue.Empty:
                continue
            deadline = time.perf_counter() + self.max_wait
            while len(requests) < busy:  # one request per busy worker is a full round
                try:
                    requests.append(self.channels.requests.get(timeout=max(0.0, deadline - time.perf_counter())))
                except queue.Empty:
                    break
            self.answer(requests)
        return [j.get() for j in jobs]

    def answer(self, requests: list[tuple]):
        groups = {}
        for r in requests:
            groups.setdefault(r[2], []).append(r)
        for net_id, group in groups.items():
            shape = group[0][4]
            planes, cells = int(np.prod(shape)), shape[1] * shape[2]
            views = [(self.views[w][slot], n) for w, slot, _, n, _ in group]
            obs = np.concatenate([v[0][:n, :planes] for v, n in views]).reshape(-1, *shape)
            mask = np.concatenate([v[1][:n, :cells] for v, n in views])
            probs, values = self.evaluators[net_id](obs.astype(np.float32), mask)
            self.batches += 1
            self.positions += len(obs)
            start = 0
            for (w, slot, _, n, _), (v, _) in zip(group, views):
                v[2][:n, :cells] = probs[start:start + n]
                v[3][:n] = values[start:start + n]
                start += n
                self.channels.ready[w][slot].release()

    def close(self):
        self.pool.close()
        self.pool.join()
        self.channels.close()
