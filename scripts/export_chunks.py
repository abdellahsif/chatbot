import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_loader import load_bundle

OUT_PATH = ROOT / "scripts" / "all_chunks.jsonl"


def main() -> None:
    bundle = load_bundle(ROOT)
    count = 0
    with OUT_PATH.open("w", encoding="utf-8") as handle:
        for chunk in bundle.transcripts:
            handle.write(json.dumps(chunk, ensure_ascii=False))
            handle.write("\n")
            count += 1
    print(f"Wrote {count} chunks to {OUT_PATH}")


if __name__ == "__main__":
    main()
