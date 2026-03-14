import csv
import re
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATE_DIR = SCRIPT_DIR.parent
DATA_ROOT = EVALUATE_DIR / "data"
TARGET_DIR = DATA_ROOT / "GSE211956" / "P3"

# input_csv = DATA_ROOT / "GSE211956" / "P2" / "lr_communication.csv"
# output_csv = DATA_ROOT / "GSE211956" / "P2" / "lr_communication_fixed.csv"
# backup_csv = DATA_ROOT / "GSE211956" / "P2" / "lr_communication_backup.csv"

input_csv = TARGET_DIR / "lr_communication.csv"
output_csv = TARGET_DIR / "lr_communication_fixed.csv"
backup_csv = TARGET_DIR / "lr_communication_backup.csv"

print(f"Reading: {input_csv}")

shutil.copy(input_csv, backup_csv)
print(f"Backup created: {backup_csv}")

with open(input_csv, "r", encoding="utf-8") as f_in:
    with open(output_csv, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out, quoting=csv.QUOTE_NONNUMERIC)

        for i, line in enumerate(f_in):
            line = line.strip()

            if i == 0:
                headers = line.split(",")
                writer.writerow(headers)
                print(f"Headers ({len(headers)} columns): {headers}")
                continue

            parts = line.split(",")
            if len(parts) > 7:
                row = parts[:2]
                remaining = parts[2:]
                scores_and_pair = remaining[-3:]
                cell_parts = remaining[:-3]
                cells_text = ",".join(cell_parts)
                cells_split = re.split(r"\),(?=[A-Z])", cells_text, maxsplit=1)

                if len(cells_split) == 2:
                    source_cell = cells_split[0] + ")"
                    target_cell = cells_split[1]
                else:
                    middle_idx = len(cell_parts) // 2
                    source_cell = ",".join(cell_parts[:middle_idx])
                    target_cell = ",".join(cell_parts[middle_idx:])

                row.extend([source_cell, target_cell])
                row.extend(scores_and_pair)
            else:
                row = parts

            if len(row) == 7:
                writer.writerow(row)
            else:
                print(f"Warning: Line {i + 1} has {len(row)} fields (expected 7)")
                print(f"  Content: {line[:100]}...")

print(f"\nFixed CSV saved to: {output_csv}")
print("\nReplacing original file...")
shutil.move(output_csv, input_csv)
print("Done! Original file has been fixed and backup saved.")
