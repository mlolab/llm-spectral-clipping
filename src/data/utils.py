from multiprocessing import shared_memory
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist

from .arxiv import get_arxiv_2000, get_arxiv_full
from .benchmarks import SUPPORTED_TASK_MAP
from .c4 import get_c4_data
from .fineweb import get_fineweb_data
from .fineweb_edu import get_fineweb_edu_data
from .openwebtext2 import get_openwebtext2_data
from .redpajama import get_redpajama_data, get_redpajamav2_data
from .shakespeare import get_shakespeare_data
from .slimpajama import get_slimpajama_data
from .wikitext import get_wikitext_data


def get_dataset(args) -> Dict[str, np.ndarray]:
    """Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
    contained in its own python file. The expected format at the moment is a dictionary of np.memmap
    containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data.
    """
    if args.dataset == "wikitext":
        return get_wikitext_data(args.datasets_dir)
    if args.dataset == "shakespeare-char":
        return get_shakespeare_data(args.datasets_dir)
    if args.dataset == "arxiv2000":
        return get_arxiv_2000(args.datasets_dir)
    if args.dataset == "arxiv":
        return get_arxiv_full(args.datasets_dir)
    if args.dataset == "arxiv+wiki":
        arxiv_data = get_arxiv_full(args.datasets_dir)
        wiki_data = get_wikitext_data(args.datasets_dir)
        train_data = np.concatenate((arxiv_data["train"], wiki_data["train"]))
        val_data = np.concatenate((arxiv_data["val"], wiki_data["val"]))
        return {"train": train_data, "val": val_data}
    if args.dataset == "openwebtext2":
        return get_openwebtext2_data(args.datasets_dir)
    if args.dataset == "redpajama":
        return get_redpajama_data(args.datasets_dir)
    if args.dataset == "redpajamav2":
        return get_redpajamav2_data(args.datasets_dir)
    if args.dataset == "slimpajama":
        return get_slimpajama_data(args.datasets_dir)
    if args.dataset == "fineweb":
        return get_fineweb_data(args.datasets_dir)
    if args.dataset == "finewebedu":
        return get_fineweb_edu_data(args.datasets_dir)
    if args.dataset == "c4":
        return get_c4_data(args.datasets_dir)
    if args.dataset in SUPPORTED_TASK_MAP:
        return get_benchmark_task(args.dataset)
    else:
        raise NotImplementedError(f"Unknow dataset key '{args.dataset}'")


def get_benchmark_task(name, **kwargs):
    """Fetch the right benchmark task given by the name parameter. The logic for each task is
    contained in its own python file.
    """
    try:
        fn = SUPPORTED_TASK_MAP[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: {sorted(SUPPORTED_TASK_MAP.keys())}"
        )
    return fn(**kwargs)


class DataReader:
    def __init__(
        self,
        data_src,
        batch_size,
        sequence_length,
        seed=1337,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=False,
    ):
        if isinstance(data_src, (str, Path)):
            self.data_path = Path(data_src)
            self.keep_in_ram = keep_in_ram
            if keep_in_ram:
                self.data = np.array(
                    np.memmap(self.data_path, dtype=np.uint16, mode="r")
                )
            else:
                self.data = None
        elif isinstance(data_src, (np.ndarray, np.memmap)):
            self.data_path = None
            self.data = data_src
            self.keep_in_ram = True

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.seed = seed
        self.with_replacement = with_replacement

        self.num_tokens = len(self._get_data())

        if auto_shard and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            print(
                f"Distributed DataReader Initialized for Worker {self.rank}/{self.world_size}"
            )
        else:
            self.world_size = 1
            self.rank = 0

        # Sampling without replacement
        self.last_epoch = None
        self.order = None
        self.epoch_offset = None
        self.step = 0
        self.num_batches_of_seqlen = 0
        if not with_replacement:
            self._shuffle_epoch(0)

    def __len__(self):
        # Length in valid start indices for a sequence
        # Extra -1 to have a valid next token for the final token of the last idx
        return self.num_tokens - self.sequence_length - 1

    def _get_data(self):
        if self.data is not None:
            return self.data
        else:
            # Construct the memmap each time to avoid a memory leak per NanoGPT
            # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
            return np.memmap(self.data_path, dtype=np.uint16, mode="r")

    def __getitem__(self, idx):
        # Return the underlying datapoint, no random sampling, no worker sharding
        assert 0 <= idx < len(self)
        data = self._get_data()
        x = torch.from_numpy(data[idx : idx + self.sequence_length].astype(np.int64))
        y = torch.from_numpy(
            data[idx + 1 : idx + self.sequence_length + 1].astype(torch.int64)
        )
        return x, y

    def set_step(self, step):
        self.step = step

    def sample_batch(self):
        data = self._get_data()

        if self.with_replacement:
            idxs = self._sample_with_replacement(self.step)
        else:
            idxs = self._sample_without_replacement(self.step)
        self.step += 1

        xy = np.stack([data[i : i + self.sequence_length + 1] for i in idxs]).astype(
            np.int64
        )
        x = torch.from_numpy(xy[:, :-1]).contiguous()
        y = torch.from_numpy(xy[:, 1:]).contiguous()
        return x, y

    def _sample_with_replacement(self, idx):
        # Return an array of token indices of length self.batch_size
        # Sampled with replacement, can get repeats at any time
        seed = self.seed + idx * self.world_size + self.rank
        rng = np.random.default_rng(seed)
        return rng.integers(low=0, high=len(self), size=self.batch_size)

    def _shuffle_epoch(self, epoch):
        seed = self.seed + epoch
        rng = np.random.default_rng(seed)
        # Drop one sequence to allow different offsets per epoch:
        self.order = rng.permutation((len(self)) // self.sequence_length - 1)
        # Shift all sequences in this epoch by this amount:
        self.epoch_offset = rng.integers(self.sequence_length)
        self.last_epoch = epoch
        self.num_batches_of_seqlen = (
            len(self.order) // self.batch_size
        )  # Drops remainder batch

    def _sample_without_replacement(self, step):
        # Return an array of token indices of length self.batch_size
        # Sampled without replacement, cycle all sequences before potential repeats
        # Sequences are randomly offset in every epoch as well
        batch_idx = self.world_size * step + self.rank
        epoch_length = self.num_batches_of_seqlen

        epoch = batch_idx // epoch_length
        if epoch != self.last_epoch:
            self._shuffle_epoch(epoch)
        epoch_idx = batch_idx % epoch_length

        start = epoch_idx * self.batch_size
        end = start + self.batch_size
        return self.order[start:end] * self.sequence_length + self.epoch_offset

    def num_batches(self):
        if self.with_replacement:
            return self.num_tokens // self.batch_size
        return self.num_batches_of_seqlen


class SharedMemoryDataReader:
    """
    DataReader that uses POSIX shared memory for multi-GPU training.

    Only rank 0 loads the data into shared memory, other ranks attach to it.
    This avoids loading the dataset multiple times (once per GPU worker).

    Usage:
        # All ranks call this - rank 0 creates, others attach
        reader = SharedMemoryDataReader(
            data_path="/path/to/train.bin",
            batch_size=64,
            sequence_length=512,
            shm_name="train_data",  # unique name for shared memory
        )

        # Use like normal DataReader
        x, y = reader.sample_batch()

        # Cleanup (call on all ranks, but only rank 0 unlinks)
        reader.cleanup()
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        sequence_length: int,
        shm_name: str = "train_shm",
        seed: int = 1337,
        with_replacement: bool = False,
    ):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.seed = seed
        self.with_replacement = with_replacement
        self.shm_name = shm_name

        # Get rank info
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        # Rank 0 creates shared memory, others wait and attach
        self._setup_shared_memory()

        self.num_tokens = len(self.data)

        print(
            f"SharedMemoryDataReader initialized for Worker {self.rank}/{self.world_size}, "
            f"num_tokens={self.num_tokens:,}"
        )

        # Sampling without replacement state
        self.last_epoch = None
        self.order = None
        self.epoch_offset = None
        self.step = 0
        self.num_batches_of_seqlen = 0
        if not with_replacement:
            self._shuffle_epoch(0)

    def _setup_shared_memory(self):
        """Create or attach to shared memory."""
        # Get file size to determine shared memory size
        file_size = self.data_path.stat().st_size
        num_elements = file_size // 2  # uint16 = 2 bytes

        if self.rank == 0:
            # Rank 0: Load data and create shared memory
            print(f"Rank 0: Loading {self.data_path} into shared memory...")

            # Try to unlink any existing shared memory with same name
            try:
                existing_shm = shared_memory.SharedMemory(name=self.shm_name)
                existing_shm.close()
                existing_shm.unlink()
                print(f"Rank 0: Cleaned up existing shared memory '{self.shm_name}'")
            except FileNotFoundError:
                pass

            # Load data from disk
            data_mmap = np.memmap(self.data_path, dtype=np.uint16, mode="r")

            # Create shared memory
            self.shm = shared_memory.SharedMemory(
                name=self.shm_name,
                create=True,
                size=file_size,
            )

            # Copy data to shared memory
            shm_array = np.ndarray(
                (num_elements,), dtype=np.uint16, buffer=self.shm.buf
            )
            shm_array[:] = data_mmap[:]
            del data_mmap

            print(
                f"Rank 0: Data loaded into shared memory '{self.shm_name}' ({file_size / 1e9:.2f} GB)"
            )

        # Barrier to ensure rank 0 has created shared memory.
        # Use a gloo CPU group to avoid NCCL timeout when rank 0 is busy loading
        # a large file — NCCL communicator setup has a short timeout that fires
        # before rank 0 finishes the copy.
        if dist.is_initialized():
            if dist.get_backend() == "nccl":
                cpu_group = dist.new_group(backend="gloo")
                dist.barrier(group=cpu_group)
            else:
                dist.barrier()

        if self.rank != 0:
            # Other ranks: Attach to existing shared memory
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            print(f"Rank {self.rank}: Attached to shared memory '{self.shm_name}'")

        # Create numpy array view of shared memory (no copy)
        self.data = np.ndarray((num_elements,), dtype=np.uint16, buffer=self.shm.buf)

    def __len__(self):
        return self.num_tokens - self.sequence_length - 1

    def set_step(self, step):
        self.step = step

    def sample_batch(self):
        if self.with_replacement:
            idxs = self._sample_with_replacement(self.step)
        else:
            idxs = self._sample_without_replacement(self.step)
        self.step += 1

        xy = np.stack(
            [self.data[i : i + self.sequence_length + 1] for i in idxs]
        ).astype(np.int64)
        x = torch.from_numpy(xy[:, :-1]).contiguous()
        y = torch.from_numpy(xy[:, 1:]).contiguous()
        return x, y

    def _sample_with_replacement(self, idx):
        seed = self.seed + idx * self.world_size + self.rank
        rng = np.random.default_rng(seed)
        return rng.integers(low=0, high=len(self), size=self.batch_size)

    def _shuffle_epoch(self, epoch):
        seed = self.seed + epoch
        rng = np.random.default_rng(seed)
        self.order = rng.permutation((len(self)) // self.sequence_length - 1)
        self.epoch_offset = rng.integers(self.sequence_length)
        self.last_epoch = epoch
        self.num_batches_of_seqlen = len(self.order) // self.batch_size

    def _sample_without_replacement(self, step):
        batch_idx = self.world_size * step + self.rank
        epoch_length = self.num_batches_of_seqlen

        epoch = batch_idx // epoch_length
        if epoch != self.last_epoch:
            self._shuffle_epoch(epoch)
        epoch_idx = batch_idx % epoch_length

        start = epoch_idx * self.batch_size
        end = start + self.batch_size
        return self.order[start:end] * self.sequence_length + self.epoch_offset

    def num_batches(self):
        if self.with_replacement:
            return self.num_tokens // self.batch_size
        return self.num_batches_of_seqlen

    def cleanup(self):
        """Clean up shared memory. Call on all ranks at end of training."""
        self.shm.close()
        if self.rank == 0:
            try:
                self.shm.unlink()
                print(f"Rank 0: Unlinked shared memory '{self.shm_name}'")
            except FileNotFoundError:
                pass
