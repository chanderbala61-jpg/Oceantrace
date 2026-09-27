import os
import sys
import csv
import json
from datetime import datetime

def main():
    print("=== INSPECTING AIS DATASETS & FALSE DETECTION DATA ===")

    # 1. AIS Dataset 1
    ais1_path = "outputs/investigations/00955_ais_traffic.csv"
    print(f"\n1. AIS Dataset 1: {ais1_path}")
    if os.path.exists(ais1_path):
        with open(ais1_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows1 = list(reader)
        print(f"   Size: {os.path.getsize(ais1_path)} bytes")
        print(f"   Columns: {list(rows1[0].keys())}")
        print(f"   Row count: {len(rows1)}")
        mmsis1 = set(r["mmsi"] for r in rows1)
        print(f"   Unique MMSIs ({len(mmsis1)}): {mmsis1}")
        lats1 = [float(r["latitude"]) for r in rows1]
        lons1 = [float(r["longitude"]) for r in rows1]
        print(f"   Latitude range: [{min(lats1):.4f}, {max(lats1):.4f}]")
        print(f"   Longitude range: [{min(lons1):.4f}, {max(lons1):.4f}]")
        times1 = [r["timestamp"] for r in rows1]
        print(f"   Temporal coverage: {min(times1)} to {max(times1)}")
        # Check invalid or missing
        missing_coords1 = sum(1 for r in rows1 if not r["latitude"] or not r["longitude"])
        print(f"   Missing coords: {missing_coords1}")
    else:
        print("   File not found!")

    # 2. AIS Dataset 2
    ais2_path = "see thsis/Ocean Trace 2/data/synthetic_ais_demo.csv"
    print(f"\n2. AIS Dataset 2: {ais2_path}")
    if os.path.exists(ais2_path):
        with open(ais2_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows2 = list(reader)
        print(f"   Size: {os.path.getsize(ais2_path)} bytes")
        print(f"   Columns: {list(rows2[0].keys())}")
        print(f"   Row count: {len(rows2)}")
        mmsis2 = set(r["mmsi"] for r in rows2)
        print(f"   Unique MMSIs ({len(mmsis2)}): {mmsis2}")
        lats2 = [float(r["latitude"]) for r in rows2]
        lons2 = [float(r["longitude"]) for r in rows2]
        print(f"   Latitude range: [{min(lats2):.4f}, {max(lats2):.4f}]")
        print(f"   Longitude range: [{min(lons2):.4f}, {max(lons2):.4f}]")
        times2 = [r["timestamp"] for r in rows2]
        print(f"   Temporal coverage: {min(times2)} to {max(times2)}")
        missing_coords2 = sum(1 for r in rows2 if not r["latitude"] or not r["longitude"])
        print(f"   Missing coords: {missing_coords2}")
    else:
        print("   File not found!")

    # 3. False-Detection Analysis Data
    eval_path = "outputs/experiment2_test_evaluation/test_scene_results.csv"
    print(f"\n3. False Detection Data: {eval_path}")
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            eval_rows = list(reader)
        print(f"   Evaluated test scenes: {len(eval_rows)}")
        def safe_int(v, default=0):
            try:
                return int(v) if v and v != "null" else default
            except ValueError:
                return default

        fp_rows = [r for r in eval_rows if safe_int(r.get("fp", 0)) > 0]
        pure_fp_rows = [r for r in eval_rows if safe_int(r.get("fp", 0)) > 0 and safe_int(r.get("tp", 0)) == 0]
        null_rows = [r for r in eval_rows if r.get("fp") == "null" or r.get("status") != "success"]
        print(f"   Scenes with status != success or null metrics: {len(null_rows)}")
        print(f"   Scenes with false positive pixels (fp > 0): {len(fp_rows)}")
        print(f"   Pure false positive scenes (tp == 0, fp > 0): {len(pure_fp_rows)}")
        
        # Total FP pixels across test set
        total_fp = sum(safe_int(r.get("fp", 0)) for r in eval_rows)
        total_tp = sum(safe_int(r.get("tp", 0)) for r in eval_rows)
        print(f"   Total FP pixels: {total_fp:,}, Total TP pixels: {total_tp:,}")

    # 4. Cross reference with metadata_index.csv for timestamps and geospatial bounds
    meta_path = "metadata_index.csv"
    meta_dict = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            raw_headers = next(reader)
            headers = [h.strip().strip('"') for h in raw_headers]
            for row in reader:
                item = dict(zip(headers, [v.strip().strip('"') for v in row]))
                fn = item.get("filename")
                if fn:
                    meta_dict[fn] = item
        print(f"\n4. Metadata Index: {len(meta_dict)} scenes indexed.")

    # Check top FP scenes and see their dates
    print("\n   Top 10 FP scenes with dates from metadata_index:")
    fp_sorted = sorted(fp_rows, key=lambda x: safe_int(x.get("fp", 0)), reverse=True)
    for r in fp_sorted[:10]:
        fn = r["filename"]
        meta = meta_dict.get(fn, {})
        start_time = meta.get("acquisition_start_time", "N/A")
        print(f"   - {fn} (scene {r['scene_id']}): FP={r['fp']} px, TP={r['tp']} px, Start={start_time}")

    # Check scene 00955
    meta_955 = meta_dict.get("00955.tif", {})
    print(f"\n   Scene 00955 metadata: {meta_955.get('acquisition_start_time', 'N/A')}, parent: {meta_955.get('parent_acquisition_id', 'N/A')}")

if __name__ == "__main__":
    main()
