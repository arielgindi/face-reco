# Face Recognition Datasets

## Format

Parquet with zstd compression.

| Column | Type | Description |
|--------|------|-------------|
| `identity_id` | string | Unique person ID |
| `image_filename` | string | Original filename |
| `image_bytes` | binary | Raw image (PNG/JPG) |

## Datasets

### DigiFace-1M (Synthetic)
- Files: `digiface1m_000.parquet` - `digiface1m_023.parquet`
- Identities: 109,999
- Images: 1,219,995
- Size: ~23 GB

### Digi2Real
- Files: `digi2real_000.parquet` - `digi2real_002.parquet`
- Identities: 20,000
- Images: 399,355
- Size: ~2.8 GB

## Usage

```python
import pyarrow.parquet as pq
from PIL import Image
import io

table = pq.read_table("data/digiface1m_000.parquet")
row = table.slice(0, 1).to_pydict()
img = Image.open(io.BytesIO(row["image_bytes"][0]))
```

## Sources
- DigiFace-1M: https://github.com/microsoft/DigiFace1M
- Digi2Real: Zenodo
