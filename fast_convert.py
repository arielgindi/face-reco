"""
Cross-platform Binary Cache Converter
Optimizations:
1. Distributed Writing: Workers write directly to memmap
2. CPU Saturation: Auto-scales to available cores
3. Persistent Decoders: Initializes TurboJPEG once per worker
"""
import sys
import time
import os
import glob
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


def list_parquet_files(glob_pattern: str) -> list[Path]:
    """Return sorted parquet file paths for a glob pattern."""
    return sorted(Path(p) for p in glob.glob(glob_pattern, recursive=True))

# Config - auto-detect platform
if sys.platform == "win32":
    OUTPUT_DIR = Path.home() / "face_cache"  # Windows: user home directory
else:
    OUTPUT_DIR = Path("/dev/shm/face_cache")  # Linux: shared memory

BATCH_SIZE = 1024
TOTAL_CORES = os.cpu_count() or 8
DEFAULT_WORKERS = max(1, TOTAL_CORES - 2)


def init_worker():
    """Initialize decoder once per process."""
    global _decoder
    try:
        from turbojpeg import TurboJPEG
        _decoder = TurboJPEG()
    except Exception:
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
    """Non-blocking keyboard input (cross-platform)."""
    if sys.platform == "win32":
        import msvcrt
        if msvcrt.kbhit():
            return msvcrt.getch().decode('utf-8', errors='ignore')
        return None
    else:
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def count_images(parquet_glob: str) -> int:
    """Count total images without loading data."""
    files = list_parquet_files(parquet_glob)
    total = 0
    for fp in files:
        try:
            pf = pq.ParquetFile(fp)
            total += pf.metadata.num_rows
        except Exception:
            continue
    return total


def iter_jpeg_bytes(parquet_glob: str):
    """Iterate over JPEG bytes without loading all into memory."""
    files = list_parquet_files(parquet_glob)
    if not files:
        raise FileNotFoundError(f"No parquet files: {parquet_glob}")

    print(f"Found {len(files)} parquet files")

    for fp in files:
        try:
            pf = pq.ParquetFile(fp)
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["image_bytes"]):
                yield from (b for b in batch.column(0).to_pylist() if b)
        except Exception:
            continue


def main():
    # Check TurboJPEG
    try:
        from turbojpeg import TurboJPEG
        TurboJPEG()  # Test if library is available
        decoder_name = "TurboJPEG (SIMD)"
    except Exception:
        decoder_name = "PIL"

    # Setup terminal for non-blocking input (Linux only)
    old_settings = None
    if sys.platform != "win32":
        import termios
        import tty
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    try:
        print("=" * 60)
        print("BINARY CACHE CONVERTER")
        print(f"CPU: {TOTAL_CORES} threads | Decoder: {decoder_name}")
        print(f"Output: {OUTPUT_DIR}")
        print("=" * 60)

        parquet_glob = "data/digiface1m_*.parquet"

        # Count total images first
        print("Counting images...")
        total = count_images(parquet_glob)
        if total == 0:
            print("No images found!")
            return

        print(f"Found {total:,} images")

        # Get first image to determine shape
        first_bytes = next(iter_jpeg_bytes(parquet_glob))
        try:
            from turbojpeg import TurboJPEG
            first_img = TurboJPEG().decode(first_bytes)
        except Exception:
            from PIL import Image
            import io
            first_img = np.array(Image.open(io.BytesIO(first_bytes)).convert("RGB"))

        img_shape = first_img.shape
        full_shape = (total, *img_shape)
        dtype = np.uint8

        # Create memmap file
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "images.npy"

        print(f"Creating {total * np.prod(img_shape) / 1e9:.1f} GB file: {out_path}")
        fp = np.lib.format.open_memmap(str(out_path), mode='w+', dtype=dtype, shape=full_shape)
        fp[0] = first_img
        fp.flush()
        del fp  # Close, workers will open their own handles

        print(f"Processing {total:,} images...")

        num_workers = DEFAULT_WORKERS
        ctx = mp.get_context('spawn')
        executor = ProcessPoolExecutor(
            max_workers=TOTAL_CORES,
            mp_context=ctx,
            initializer=init_worker
        )

        futures = {}
        completed_images = 1
        active_tasks = 0
        start_time = time.time()

        pbar = tqdm(total=total, initial=1, unit="img", smoothing=0.1)
        pbar.set_description(f"Decoding (w={num_workers})")

        # Stream batches from parquet files
        batch = []
        batch_start_idx = 1  # First image already written
        jpeg_iter = iter_jpeg_bytes(parquet_glob)
        next(jpeg_iter)  # Skip first (already processed)

        for idx, jpeg_bytes in enumerate(jpeg_iter, start=1):
            batch.append(jpeg_bytes)

            if len(batch) >= BATCH_SIZE:
                # Submit batch
                task = (out_path, full_shape, dtype, batch_start_idx, batch)
                f = executor.submit(worker_task, task)
                futures[f] = len(batch)
                active_tasks += 1
                batch_start_idx += len(batch)
                batch = []

                # Limit concurrent tasks to avoid memory issues
                while active_tasks >= num_workers * 2:
                    done_list = [f for f in futures if f.done()]
                    for f in done_list:
                        batch_count = futures.pop(f)
                        active_tasks -= 1
                        try:
                            f.result()
                            completed_images += batch_count
                            pbar.update(batch_count)
                        except Exception as e:
                            print(f"Batch failed: {e}")
                    if active_tasks >= num_workers * 2:
                        time.sleep(0.01)

                elapsed = time.time() - start_time
                if elapsed > 0:
                    pbar.set_postfix({"ips": f"{completed_images / elapsed:.0f}"})

        # Submit final batch
        if batch:
            task = (out_path, full_shape, dtype, batch_start_idx, batch)
            f = executor.submit(worker_task, task)
            futures[f] = len(batch)

        # Wait for remaining tasks
        for f in futures:
            try:
                f.result()
                completed_images += futures[f]
                pbar.update(futures[f])
            except Exception as e:
                print(f"Batch failed: {e}")

        pbar.close()
        executor.shutdown(wait=True)

        elapsed = time.time() - start_time
        print(f"\nDone! {completed_images:,} images in {elapsed:.1f}s ({completed_images/elapsed:.0f} ips)")
        print(f"File ready at: {out_path}")

    finally:
        if old_settings is not None:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
