import csv
import json
import os
import sys
import time


def save_csv(rows, output_path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"       Saved: {output_path}")


def save_json(rows, output_path):
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, default=str)
    print(f"       Saved: {output_path}")


def print_summary(rows):
    if not rows:
        print("\nNo candidates to display.")
        return

    header = f"{'Rank':<6} {'Candidate':<35} {'Score':<8} {'Exp':<5} {'Skills'}"
    sep = "-" * len(header)
    print(f"\n{header}")
    print(sep)
    for row in rows[:10]:
        skills = row["detected_skills"][:40] if row["detected_skills"] else "—"
        exp = str(row["experience_years"]) if row["experience_years"] is not None else "?"
        print(
            f"{row['rank']:<6} {row['candidate'][:34]:<35} "
            f"{row['composite_index']:<8.3f} {exp:<5} {skills}"
        )
    if len(rows) > 10:
        print(f"... and {len(rows) - 10} more")
    print()


def generate_report(rows, output_dir):
    csv_path = os.path.join(output_dir, "ranked_candidates.csv")
    json_path = os.path.join(output_dir, "ranked_candidates.json")

    save_csv(rows, csv_path)
    save_json(rows, json_path)
    print_summary(rows)


class ProgressTracker:
    def __init__(self, total, prefix=""):
        self.total = total
        self.count = 0
        self.prefix = prefix
        self.start = time.time()

    def update(self, n=1):
        self.count += n
        pct = self.count / self.total * 100
        elapsed = time.time() - self.start
        bar_len = 30
        filled = int(bar_len * self.count / self.total)
        bar = "#" * filled + "-" * (bar_len - filled)
        msg = f"\r{self.prefix} [{bar}] {self.count}/{self.total} ({pct:.0f}%)"
        sys.stdout.write(msg.encode("utf-8", errors="replace").decode("cp1252", errors="replace"))
        sys.stdout.flush()
        if self.count >= self.total:
            print(f" — {elapsed:.1f}s")
