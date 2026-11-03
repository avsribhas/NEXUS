import argparse
import json
import sys
from pathlib import Path
from nexus.format.exam import lock_files, verify_lock

def main():
    ROOT = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Hash files into a lock record, or check a lock record.")
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make")
    make.add_argument("record")
    make.add_argument("files", nargs="+")
    make.add_argument("--stamp", action="store_true")
    check = sub.add_parser("verify")
    check.add_argument("record")
    args = parser.parse_args()
    if args.command == "make":
        record = lock_files(args.files, args.record, ROOT, stamp=args.stamp)
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    else:
        changed = verify_lock(args.record, ROOT)
        for name in changed:
            print(name)
        print("lock ok" if not changed else f"{len(changed)} file(s) changed")
        return 1 if changed else 0

if __name__ == "__main__":
    sys.exit(main())
