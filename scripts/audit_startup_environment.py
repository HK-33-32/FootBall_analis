"""Save non-secret Docker workload metadata without changing unrelated services."""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def compare_snapshots(before, after, exclude=()):
    def selected(snapshot):
        return {row["name"]: row for row in snapshot["containers"]
                if row["name"] not in exclude}

    left, right = selected(before), selected(after)
    return {"same_service_set": left == right, "excluded": list(exclude),
            "before_timestamp": before["timestamp"], "after_timestamp": after["timestamp"],
            "added": sorted(right.keys() - left.keys()),
            "removed": sorted(left.keys() - right.keys()),
            "changed": sorted(name for name in left.keys() & right.keys()
                              if left[name] != right[name]),
            "note": "Same services/start times do not imply equal CPU load or OS cache state."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path, nargs=2)
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()
    if args.compare:
        result = compare_snapshots(*(json.loads(p.read_text()) for p in args.compare), args.exclude)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result, indent=2))
        return
    ids = subprocess.check_output(["docker", "ps", "-q"], text=True).split()
    template = ('{"name":{{json .Name}},"image_id":{{json .Image}},'
                '"started_at":{{json .State.StartedAt}}}')
    values = subprocess.check_output(
        ["docker", "inspect", "--format", template, *ids], text=True,
    ).splitlines() if ids else []
    result = {"timestamp": datetime.now(UTC).isoformat(),
              "containers": [json.loads(row) for row in values],
              "note": "Service-set snapshot, not control of CPU load or OS disk cache. "
                      "No environment variables, credentials or container mutations collected."}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(f"Saved {len(values)} running containers to {args.output}")


if __name__ == "__main__":
    main()
