"""
UCX CUDA tensor transfer
=========================

A server listens for UCX connections and receives a PyTorch CUDA tensor.
The UCX listener port is shared via a temp file (replace with any
out-of-band mechanism: env var, etcd, command-line arg, etc.).

The tensor stays on the GPU throughout -- ``cupy`` is used as the
intermediate buffer format so no device-to-host copy is needed.

Usage
-----
Terminal 1 (server):
    python tcp_bootstrap_tensor.py server

Terminal 2 (client):
    python tcp_bootstrap_tensor.py client
"""

import argparse
import os
import tempfile

import cupy as cp
import msgpack
import numpy as np
import trio

import ucxx

HOST = "127.0.0.1"
PORT_FILE = os.path.join(tempfile.gettempdir(), "ucxx_example_port")


# ── server ───────────────────────────────────────────────────────────


async def server(host):
    async with ucxx.init():
        transfer_done = trio.Event()

        async def on_connection(ep):
            header = await ep.recv_obj()
            meta = msgpack.unpackb(bytes(header), raw=False)

            buf = cp.empty(meta["nbytes"], dtype=cp.uint8)
            await ep.recv(buf)

            import torch

            typed = buf.view(cp.dtype(meta["dtype"])).reshape(meta["shape"])
            tensor = torch.from_dlpack(typed)
            print(f"[server] Received tensor: shape={tuple(tensor.shape)}, "
                  f"dtype={tensor.dtype}, device={tensor.device}")
            print(f"[server] Data:\n{tensor}")

            transfer_done.set()

        async with ucxx.create_listener(on_connection, port=0) as listener:
            print(f"[server] UCX listener on {host}:{listener.port}")

            with open(PORT_FILE, "w") as f:
                f.write(str(listener.port))

            with trio.fail_after(30):
                await transfer_done.wait()

            print("[server] Transfer complete")


# ── client ───────────────────────────────────────────────────────────


async def client(host):
    import torch

    async with ucxx.init():
        with open(PORT_FILE) as f:
            port = int(f.read())
        print(f"[client] Connecting to {host}:{port}")

        async with await ucxx.create_endpoint(host, port) as ep:
            tensor = torch.arange(12, dtype=torch.float32, device="cuda").reshape(3, 4)
            print(f"[client] Sending tensor: shape={tuple(tensor.shape)}, "
                  f"dtype={tensor.dtype}, device={tensor.device}")
            print(f"[client] Data:\n{tensor}")

            gpu_view = cp.from_dlpack(tensor)
            meta = msgpack.packb({
                "shape": list(tensor.shape),
                "dtype": str(gpu_view.dtype),
                "nbytes": gpu_view.nbytes,
            })
            await ep.send_obj(np.frombuffer(meta, dtype=np.uint8))
            await ep.send(gpu_view.view(cp.uint8).ravel())

            print("[client] Tensor sent (GPU → GPU)")


# ── entry point ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="UCX CUDA tensor transfer"
    )
    parser.add_argument("role", choices=["server", "client"])
    parser.add_argument("--host", default=HOST)
    args = parser.parse_args()
    trio.run(server if args.role == "server" else client, args.host)


if __name__ == "__main__":
    main()
