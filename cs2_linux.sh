#!/usr/bin/env python3
import argparse
import os
import sys
import time
import glob
import shutil
import signal
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


def require(tool: str):
    if shutil.which(tool) is None:
        print(f"[ERROR] Required tool not found in PATH: {tool}", file=sys.stderr)
        sys.exit(1)


def apply_intel_undervolt(skip_undervolt: bool, intel_undervolt_conf: str):
    if skip_undervolt:
        print("[INFO] Skipping undervolt application as requested.")
        return

    require("intel-undervolt")

    # Optional: sanity check that config exists
    conf_path = Path(intel_undervolt_conf)
    if not conf_path.exists():
        print(f"[ERROR] intel-undervolt config not found at {conf_path}", file=sys.stderr)
        sys.exit(1)

    # Requires root; user can run this script with sudo or have NOPASSWD for intel-undervolt
    cmd = ["sudo", "intel-undervolt", "apply"]
    print(f"[INFO] Applying undervolt via: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] intel-undervolt apply failed: {e}", file=sys.stderr)
        sys.exit(1)


def launch_cs2_with_mangohud(steam_app_id: str, mango_out_dir: Path, extra_launch_opts: str):
    """
    Launch CS2 wrapped by MangoHud so it writes CSV logs to mango_out_dir.
    Returns the Popen object for the launched Steam process.
    """
    require("mangohud")
    require("steam")

    mango_out_dir.mkdir(parents=True, exist_ok=True)

    # MangoHud runtime configuration:
    # - read_cfg=0 ensures we use only what we pass here
    # - no_display=1 hides the HUD overlay (optional)
    # - output_folder=<dir> location for CSV files
    # - output_format=csv ensures CSV logs
    # - benchmark=1 enables logging suitable for benchmarks
    # (If a particular option doesn’t exist in your MangoHud version, remove it; logging still works with defaults.)
    mangohud_cfg = ",".join([
        "read_cfg=0",
        "no_display=1",
        f"output_folder={str(mango_out_dir)}",
        "output_format=csv",
        "benchmark=1"
    ])

    env = os.environ.copy()
    env["MANGOHUD"] = "1"
    env["MANGOHUD_CONFIG"] = mangohud_cfg

    # Common CS2 launch opts; customize as needed
    launch_opts = [
        "-novid",
        "-fullscreen",
        "-high",
        "-vulkan",  # remove if you don't use Vulkan
        "-console"
    ]
    if extra_launch_opts:
        launch_opts.extend(extra_launch_opts.split())

    cmd = ["mangohud", "--dlsym", "steam", "-applaunch", steam_app_id] + launch_opts

    print(f"[INFO] Launching CS2 with MangoHud:\n       {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env)
    return proc


def find_newest_csv(folder: Path, older_than: float) -> Path | None:
    """
    Find the newest MangoHud CSV created after 'older_than' timestamp.
    """
    newest = None
    newest_mtime = older_than
    for path in glob.glob(str(folder / "**" / "*.csv"), recursive=True):
        p = Path(path)
        try:
            mtime = p.stat().st_mtime
            if mtime > newest_mtime:
                newest = p
                newest_mtime = mtime
        except FileNotFoundError:
            continue
    return newest


def compute_metrics_from_csv(csv_path: Path):
    """
    Compute min FPS, average FPS, and 1% low from MangoHud CSV.
    MangoHud CSV typically includes 'fps' and/or 'frametime_ms' columns.
    We’ll prefer 'fps' directly; if missing, derive from frametime.
    """
    df = pd.read_csv(csv_path)

    if "fps" in df.columns:
        fps = df["fps"].astype(float)
    elif "frametime_ms" in df.columns:
        # FPS = 1000 / frametime_ms
        fps = 1000.0 / df["frametime_ms"].astype(float)
    else:
        raise ValueError(f"CSV missing both 'fps' and 'frametime_ms' columns: {csv_path}")

    # Drop any non-finite values that can appear at start/stop
    fps = fps.replace([pd.NA, pd.NaT, float("inf"), -float("inf")], pd.NA).dropna()
    if len(fps) == 0:
        raise ValueError("No valid FPS samples in the CSV.")

    min_fps = float(fps.min())
    avg_fps = float(fps.mean())

    # 1% low (1st percentile of FPS samples)
    p1_low = float(fps.quantile(0.01, interpolation="linear"))

    # Return metrics and the full raw series in case the caller wants to reuse
    return {
        "min_fps": round(min_fps, 2),
        "avg_fps": round(avg_fps, 2),
        "p1_low_fps": round(p1_low, 2),
        "samples": int(fps.shape[0]),
    }


def kill_children(proc: subprocess.Popen):
    """
    Best-effort kill of the launched process tree.
    Steam spawns children (CS2). We try to terminate nicely first.
    """
    try:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Apply undervolt, run CS2 benchmark, save FPS stats.")
    parser.add_argument("--skip-undervolt", action="store_true",
                        help="Skip applying intel-undervolt.")
    parser.add_argument("--intel-undervolt-conf", default="/etc/intel-undervolt.conf",
                        help="Path to intel-undervolt config (default: /etc/intel-undervolt.conf).")
    parser.add_argument("--appid", default="730", help="Steam app id for CS2 (default: 730).")
    parser.add_argument("--duration", type=int, default=120,
                        help="Benchmark duration in seconds before terminating the game (default: 120).")
    parser.add_argument("--mangohud-out", default=str(Path.home() / "MangoHud"),
                        help="Output directory where MangoHud writes CSV logs.")
    parser.add_argument("--extra-launch-opts", default="",
                        help="Extra launch options passed to CS2 (space-separated).")
    parser.add_argument("--results-out", default="cs2_benchmark_results.csv",
                        help="Path to save the computed results table (CSV).")
    parser.add_argument("--tag", default="default",
                        help="A short label to tag this run (e.g., 'uv-120mV').")
    parser.add_argument("--save-parquet", action="store_true",
                        help="Also save a Parquet copy next to the CSV.")

    args = parser.parse_args()

    mango_out_dir = Path(args.mangohud_out)
    results_csv = Path(args.results_out)
    parquet_path = results_csv.with_suffix(".parquet")

    # 1) Apply undervolt
    apply_intel_undervolt(args.skip_undervolt, args.intel_undervolt_conf)

    # 2) Launch CS2 wrapped by MangoHud
    before_launch = time.time()
    proc = launch_cs2_with_mangohud(args.appid, mango_out_dir, args.extra_launch_opts)

    # 3) Wait for duration, then terminate
    print(f"[INFO] Running benchmark for {args.duration} seconds...")
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user; proceeding to collect data.")
    finally:
        print("[INFO] Stopping CS2/Steam...")
        kill_children(proc)

    time.sleep(3)  # let MangoHud flush files

    # 4) Find the newest CSV created after we launched
    newest_csv = find_newest_csv(mango_out_dir, older_than=before_launch - 1)
    if not newest_csv or not newest_csv.exists():
        print(f"[ERROR] Could not find a MangoHud CSV in {mango_out_dir} created during the run.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Using MangoHud CSV: {newest_csv}")

    # 5) Compute metrics and save a small DataFrame
    try:
        metrics = compute_metrics_from_csv(newest_csv)
    except Exception as e:
        print(f"[ERROR] Failed to compute metrics: {e}", file=sys.stderr)
        sys.exit(1)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "timestamp": now,
        "tag": args.tag,
        "appid": args.appid,
        "duration_s": args.duration,
        "min_fps": metrics["min_fps"],
        "avg_fps": metrics["avg_fps"],
        "p1_low_fps": metrics["p1_low_fps"],
        "samples": metrics["samples"],
        "csv_path": str(newest_csv)
    }

    df_row = pd.DataFrame([row])

    if results_csv.exists():
        try:
            old = pd.read_csv(results_csv)
            df_out = pd.concat([old, df_row], ignore_index=True)
        except Exception:
            # If something's wrong with the existing file, just overwrite with the new row
            df_out = df_row
    else:
        df_out = df_row

    df_out.to_csv(results_csv, index=False)
    if args.save_parquet:
        try:
            df_out.to_parquet(parquet_path, index=False)
        except Exception as e:
            print(f"[WARN] Failed to save Parquet: {e}")

    print("\n[RESULTS]")
    print(df_row.to_string(index=False))
    print(f"\n[INFO] Saved results to: {results_csv}")
    if args.save_parquet:
        print(f"[INFO] Saved Parquet to: {parquet_path}")


if __name__ == "__main__":
    main()

