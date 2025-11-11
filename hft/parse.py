#!/usr/bin/env python3
# convert_parquet.py
# Convert .parquet files to csv/jsonl/feather/pickle using pandas+pyarrow.

import argparse, sys
from pathlib import Path

import pandas as pd  # pip install pandas pyarrow

def check_file_integrity(src: Path):
    """Check if file appears to be a valid parquet file"""
    try:
        with open(src, 'rb') as f:
            header = f.read(4)
            f.seek(-4, 2)  # Seek to 4 bytes from end
            footer = f.read(4)
            has_valid_header = header == b'PAR1'
            has_valid_footer = footer == b'PAR1'
            return has_valid_header, has_valid_footer
    except Exception as e:
        print(f"Warning: Could not check file integrity: {e}", file=sys.stderr)
        return False, False

def try_recover_data(src: Path):
    """Attempt to recover data from corrupted parquet file"""
    print("Attempting data recovery from corrupted file...", file=sys.stderr)
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
        
        # Try to read row groups individually
        pf = pq.ParquetFile(src)
        # This will fail if footer is missing, but let's try reading raw data
        return None
    except:
        pass
    
    # If that fails, the file is too corrupted to recover
    return None

def read_parquet_with_fallback(src: Path):
    """Try multiple methods to read parquet file"""
    # Method 1: Try pyarrow engine (default)
    try:
        return pd.read_parquet(src, engine="pyarrow"), "pyarrow"
    except Exception as e1:
        print(f"PyArrow failed: {e1}", file=sys.stderr)
        
        # Method 2: Try fastparquet engine
        try:
            return pd.read_parquet(src, engine="fastparquet"), "fastparquet"
        except ImportError:
            print("FastParquet not installed. Install with: pip3 install fastparquet", file=sys.stderr)
        except Exception as e2:
            print(f"FastParquet failed: {e2}", file=sys.stderr)
        
        # Method 3: Try reading as file handle
        try:
            with open(src, 'rb') as f:
                import pyarrow.parquet as pq
                table = pq.read_table(f)
                return table.to_pandas(), "pyarrow-file-handle"
        except Exception as e3:
            print(f"File handle method failed: {e3}", file=sys.stderr)
        
        # Method 4: Try recovery mode for corrupted files
        recovered = try_recover_data(src)
        if recovered is not None:
            return recovered, "recovered"
        
        # If all methods fail, raise the original error
        raise Exception(f"All reading methods failed. Last error: {e1}")

def convert_file(src: Path, out_dir: Path, fmt: str):
    """Convert parquet file to specified format"""
    # Check file integrity first
    has_header, has_footer = check_file_integrity(src)
    if not has_header:
        print(f"Warning: {src} does not have valid PAR1 header", file=sys.stderr)
    if not has_footer:
        print(f"Warning: {src} does not have valid PAR1 footer (file may be incomplete)", file=sys.stderr)
    
    # Try to read the file
    try:
        df, method = read_parquet_with_fallback(src)
        print(f"Successfully read {src} using {method} engine")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading {src}: {e}", file=sys.stderr)
        raise

    stem = src.stem  # removes .parquet
    if fmt == "csv":
        dst = out_dir / f"{stem}.csv"
        df.to_csv(dst, index=False)
    elif fmt == "jsonl":
        dst = out_dir / f"{stem}.jsonl"
        df.to_json(dst, orient="records", lines=True)
    elif fmt == "feather":
        dst = out_dir / f"{stem}.feather"
        df.reset_index(drop=True).to_feather(dst)  # feather dislikes non-range index
    elif fmt == "pickle":
        dst = out_dir / f"{stem}.pkl"
        df.to_pickle(dst, protocol=4)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    print(f"{src} -> {dst}")

def iter_parquet_inputs(pathlike: Path):
    if pathlike.is_dir():
        yield from sorted(pathlike.rglob("*.parquet"))
    elif pathlike.is_file() and pathlike.suffix == ".parquet":
        yield pathlike
    else:
        # support simple globs like "*.parquet"
        for p in Path().glob(str(pathlike)):
            if p.suffix == ".parquet":
                yield p

def diagnose_file(src: Path):
    """Print diagnostic information about a parquet file"""
    print(f"\n{'='*60}")
    print(f"Diagnosing: {src}")
    print(f"{'='*60}")
    
    # File size
    size = src.stat().st_size
    print(f"File size: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")
    
    # Check magic bytes
    has_header, has_footer = check_file_integrity(src)
    print(f"Has PAR1 header: {has_header}")
    print(f"Has PAR1 footer: {has_footer}")
    
    if not has_footer:
        print("\n⚠️  WARNING: File appears to be incomplete or corrupted!")
        print("   Parquet files must end with PAR1 magic bytes.")
        print("   The file may have been truncated during download/transfer.")
        print("\n   Suggestions:")
        print("   1. Re-download the file if possible")
        print("   2. Check if the file transfer was interrupted")
        print("   3. Verify the source file is complete")
    
    # Try to read metadata if possible
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(src)
        print(f"\nSchema: {pf.schema}")
        print(f"Metadata: {pf.metadata}")
    except Exception as e:
        print(f"\nCould not read metadata: {e}")
    
    # Try to read with pandas
    print("\nAttempting to read file...")
    try:
        df, method = read_parquet_with_fallback(src)
        print(f"✅ Successfully read using {method}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Data types:\n{df.dtypes}")
        print(f"\n   First few rows:")
        print(df.head())
        return True
    except Exception as e:
        print(f"❌ Failed to read: {e}")
        return False

def main():
    p = argparse.ArgumentParser(description="Convert .parquet to other formats")
    p.add_argument("input", help="Parquet file, directory, or glob (e.g., data/*.parquet)")
    p.add_argument(
        "--out-format",
        choices=["csv", "jsonl", "feather", "pickle"],
        default="csv",
        help="Output format (default: csv)",
    )
    p.add_argument("--out-dir", default="out", help="Output directory (default: ./out)")
    p.add_argument("--diagnose", action="store_true", help="Diagnose file(s) without converting")
    args = p.parse_args()

    inputs = list(iter_parquet_inputs(Path(args.input)))
    if not inputs:
        print("No .parquet files found.", file=sys.stderr)
        sys.exit(1)

    if args.diagnose:
        # Diagnostic mode
        for src in inputs:
            diagnose_file(src)
        return

    # Conversion mode
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in inputs:
        convert_file(src, out_dir, args.out_format)

if __name__ == "__main__":
    main()
