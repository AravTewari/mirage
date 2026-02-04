#!/usr/bin/env python3
import os
import sys
import tempfile
from unittest import mock
import types

import torch

repo_python = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, repo_python)

mirage_pkg = types.ModuleType("mirage")
mirage_pkg.__path__ = [os.path.join(repo_python, "mirage")]
sys.modules["mirage"] = mirage_pkg

mpk_pkg = types.ModuleType("mirage.mpk")
mpk_pkg.__path__ = [os.path.join(repo_python, "mirage", "mpk")]
sys.modules["mirage.mpk"] = mpk_pkg

core_stub = types.ModuleType("mirage.core")


class DTensor:
    pass


class dtype:
    pass


core_stub.CyKNGraph = lambda *args, **kwargs: object()
core_stub.DTensor = DTensor
core_stub.dtype = dtype
core_stub.int64 = object()
core_stub.bfloat16 = object()
sys.modules["mirage.core"] = core_stub

kernel_stub = types.ModuleType("mirage.kernel")
kernel_stub.KNGraph = lambda *args, **kwargs: object()
kernel_stub.TBGraph = lambda *args, **kwargs: object()
kernel_stub.get_key_paths = lambda: (".", ".", ".")
sys.modules["mirage.kernel"] = kernel_stub

spec_stub = types.ModuleType("mirage.mpk.speculative")

class SpecDecodeConfig:
    method = "promptlookup"


class PromptLookupConfig:
    pass

spec_stub.SpecDecodeConfig = SpecDecodeConfig
spec_stub.PromptLookupConfig = PromptLookupConfig
sys.modules["mirage.mpk.speculative"] = spec_stub

from mirage.mpk.persistent_kernel import PersistentKernel


class DummyCudaProps:
    major = 8
    minor = 0


def make_meta_tensors():
    return {
        "step": torch.zeros((1,), dtype=torch.int32),
        "tokens": torch.zeros((1, 4), dtype=torch.int64),
        "input_tokens": torch.zeros((1, 1), dtype=torch.int64),
        "output_tokens": torch.zeros((1, 1), dtype=torch.int64),
        "num_new_tokens": torch.ones((1,), dtype=torch.int32),
        "prompt_lengths": torch.zeros((1,), dtype=torch.int32),
        "qo_indptr_buffer": torch.zeros((2,), dtype=torch.int32),
        "paged_kv_indptr_buffer": torch.zeros((2,), dtype=torch.int32),
        "paged_kv_indices_buffer": torch.zeros((1,), dtype=torch.int32),
        "paged_kv_last_page_len_buffer": torch.zeros((1,), dtype=torch.int32),
    }


def install_noop_launcher(persistent_kernel: PersistentKernel):
    def _noop_loader(_so_path: str):
        persistent_kernel.init_func = lambda *args, **kwargs: None
        persistent_kernel.launch_func = lambda *args, **kwargs: None
        persistent_kernel.init_request_func = lambda *args, **kwargs: None
        persistent_kernel.finalize_func = lambda *args, **kwargs: None

    persistent_kernel._load_launcher_module = _noop_loader


def main():
    compile_calls = 0
    with tempfile.TemporaryDirectory() as output_dir:
        with mock.patch("torch.cuda.get_device_properties", return_value=DummyCudaProps()):
            meta_tensors = make_meta_tensors()
            first = PersistentKernel(
                mode="offline",
                world_size=1,
                mpi_rank=0,
                num_workers=1,
                num_local_schedulers=1,
                num_remote_schedulers=0,
                max_seq_length=4,
                max_num_batched_requests=1,
                max_num_batched_tokens=1,
                max_num_pages=1,
                page_size=1,
                meta_tensors=meta_tensors,
                profiler_tensor=None,
                trace_name=None,
                spec_decode_config=None,
                use_cutlass_kernel=True,
            )
            install_noop_launcher(first)

            compile_calls += 1
            os.makedirs(output_dir, exist_ok=True)
            launcher_path = os.path.join(output_dir, first._launcher_filename())
            with open(launcher_path, "wb") as f:
                f.write(b"\x7fELF")

            second = PersistentKernel(
                mode="offline",
                world_size=1,
                mpi_rank=0,
                num_workers=1,
                num_local_schedulers=1,
                num_remote_schedulers=0,
                max_seq_length=4,
                max_num_batched_requests=1,
                max_num_batched_tokens=1,
                max_num_pages=1,
                page_size=1,
                meta_tensors=make_meta_tensors(),
                profiler_tensor=None,
                trace_name=None,
                spec_decode_config=None,
                use_cutlass_kernel=True,
            )
            install_noop_launcher(second)
            second.load_mpk_kernel(output_dir=output_dir, eos_token_id=1)

    assert compile_calls == 1, f"Expected 1 compilation, got {compile_calls}"
    print("PASS: load_mpk_kernel reuse avoids extra compilation.")


if __name__ == "__main__":
    main()
