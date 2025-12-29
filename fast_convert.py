"""
AMD EPYC OPTIMIZED CONVERTER
Target: AMD EPYC 9354 (32 Cores / 64 Threads)

Optimizations:
1. Distributed Writing: Workers write directly to memmap (bypasses main thread bottleneck)
2. CPU Saturation: Auto-scales to use ~95% of available cores
3. Persistent Decoders: Initializes TurboJPEG once per worker

Controls: [+] add worker  [-] remove worker  [q] quit
"""
import sys
import time
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from src.data.file_utils import list_parquet_files

# Config
OUTPUT_DIR = Path("/dev/shm/face_cache")
BATCH_SIZE = 1024
TOTAL_CORES = os.cpu_count()
DEFAULT_WORKERS = max(1, TOTAL_CORES - 2)


def init_worker():
    """Initialize decoder once per process."""
    global _decoder
    try:
        from turbojpeg import TurboJPEG
        _decoder = TurboJPEG()
    except ImportError:
        _decoder = None


def worker_task(args):
    """Decode batch and write DIRECTLY to memmap (no return to main thread)."""
    filename, shape, dtype, start_idx, byte_batch = args

    # Open memmap in read-write mode
    arr = np.lib.format.open_memmap(str(filename), mode='r+', shape=shape, dtype=dtype)

    success = 0
    for i, data in enumerate(byte_batch):
        try:
            if _decoder:
                img = _decoder.decode(data)
            else:
                from PIL import Image
                import io
                img = np.array(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)

            arr[start_idx + i] = img
            success += 1
        except Exception:
            pass

    del arr
    return success


def get_key_nonblocking():
    import select
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def collect_jpeg_bytes(parquet_glob: str) -> list[bytes]:
    files = list_parquet_files(parquet_glob)
    if not files:
        raise FileNotFoundError(f"No parquet files: {parquet_glob}")

    print(f"Found {len(files)} parquet files")
    all_bytes = []

    for fp in tqdm(files, desc="Loading Parquet"):
        try:
            pf = pq.ParquetFile(fp)
            for batch in pf.iter_batches(batch_size=10000, columns=["image_bytes"]):
                all_bytes.extend([b for b in batch.column(0).to_pylist() if b])
        except Exception:
            continue

    return all_bytes


def main():
    import termios
    import tty

    # Check TurboJPEG
    try:
        import turbojpeg
        decoder_name = "TurboJPEG (SIMD)"
    except ImportError:
        decoder_name = "PIL (slow - install PyTurboJPEG)"

    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())

        print("=" * 60)
        print("AMD EPYC MAX-PERFORMANCE CONVERTER")
        print(f"CPU: {TOTAL_CORES} threads | Decoder: {decoder_name}")
        print("Controls: [+] add worker  [-] remove worker  [q] quit")
        print("=" * 60)

        # Load data
        all_bytes = collect_jpeg_bytes("data/digiface1m_*.parquet")
        total = len(all_bytes)
        if total == 0:
            return

        # Get image shape
        try:
            from turbojpeg import TurboJPEG
            first_img = TurboJPEG().decode(all_bytes[0])
        except ImportError:
            from PIL import Image
            import io
            first_img = np.array(Image.open(io.BytesIO(all_bytes[0])).convert("RGB"))

        img_shape = first_img.shape
        full_shape = (total, *img_shape)
        dtype = np.uint8

        # Create memmap file
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "images.npy"

        print(f"Creating {total * np.prod(img_shape) / 1e9:.1f} GB memmap: {out_path}")
        fp = np.lib.format.open_memmap(str(out_path), mode='w+', dtype=dtype, shape=full_shape)
        fp[0] = first_img
        fp.flush()
        del fp  # Close, workers will open their own handles

        # Create tasks
        tasks = []
        for i in range(1, total, BATCH_SIZE):
            chunk = all_bytes[i:i + BATCH_SIZE]
            tasks.append((out_path, full_shape, dtype, i, chunk))

        print(f"Processing {total:,} images in {len(tasks):,} batches...")

        num_workers = DEFAULT_WORKERS
        ctx = mp.get_context('spawn')
        executor = ProcessPoolExecutor(
            max_workers=TOTAL_CORES,
            mp_context=ctx,
            initializer=init_worker
        )

        futures = {}
        next_task_idx = 0
        completed_images = 1
        active_tasks = 0
        start_time = time.time()

        pbar = tqdm(total=total, initial=1, unit="img", smoothing=0.1)
        pbar.set_description(f"Decoding (w={num_workers})")

        def submit_tasks():
            nonlocal next_task_idx, active_tasks
            target = num_workers * 2
            while active_tasks < target and next_task_idx < len(tasks):
                t = tasks[next_task_idx]
                f = executor.submit(worker_task, t)
                futures[f] = len(t[-1])
                next_task_idx += 1
                active_tasks += 1

        submit_tasks()

        while completed_images < total:
            key = get_key_nonblocking()
            if key == '+' and num_workers < TOTAL_CORES:
                num_workers += 1
                pbar.set_description(f"Decoding (w={num_workers})")
                submit_tasks()
            elif key == '-' and num_workers > 1:
                num_workers -= 1
                pbar.set_description(f"Decoding (w={num_workers})")
            elif key == 'q':
                break

            done_list = [f for f in futures if f.done()]
            if not done_list:
                time.sleep(0.001)
                continue

            for f in done_list:
                batch_count = futures.pop(f)
                active_tasks -= 1

                try:
                    f.result()
                    completed_images += batch_count
                    pbar.update(batch_count)
                except Exception as e:
                    print(f"Batch failed: {e}")

            submit_tasks()

            elapsed = time.time() - start_time
            if elapsed > 0:
                pbar.set_postfix({"ips": f"{completed_images / elapsed:.0f}"})

        pbar.close()
        executor.shutdown(wait=False)

        elapsed = time.time() - start_time
        print(f"\nDone! {completed_images:,} images in {elapsed:.1f}s ({completed_images/elapsed:.0f} ips)")
        print(f"File ready at: {out_path}")

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
