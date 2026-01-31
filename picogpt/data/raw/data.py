from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Book sources
# -----------------------------
# urls = {
#     "gatsby": "https://www.gutenberg.org/cache/epub/64317/pg64317.txt",
# }
urls = {
    "gatsby": "https://www.gutenberg.org/cache/epub/64317/pg64317.txt",
    "pride_prejudice": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "sherlock_holmes": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    "alice": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "time_machine": "https://www.gutenberg.org/cache/epub/35/pg35.txt",
}


def _build_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {"User-Agent": "picogpt-data-downloader/1.0 (+https://github.com)"}
    )
    return session


# -----------------------------
# Gutenberg header/footer remover
# -----------------------------
def clean_gutenberg_text(text):
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
    ]

    start_idx = 0
    end_idx = len(text)

    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            start_idx = text.find("\n", idx) + 1
            break

    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    cleaned = text[start_idx:end_idx].strip()
    return cleaned


def download_books():
    print("Downloading books...")
    session = _build_session()

    for name, url in urls.items():
        output_path = RAW_DIR / f"{name}.txt"
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"- Skipping {name} (already downloaded)")
            continue

        response = session.get(url, timeout=30)
        response.raise_for_status()
        output_path.write_text(response.text, encoding="utf-8")
        print(f"- Downloaded {name}")

    print("Download complete\n")


def combine_books():
    print("Cleaning and combining texts...")
    combined_text_parts = []

    for path in sorted(RAW_DIR.glob("*.txt")):
        if path.name == "combined_novels.txt":
            continue
        raw = path.read_text(encoding="utf-8")
        cleaned = clean_gutenberg_text(raw)
        combined_text_parts.append(cleaned)

    combined_text = "\n\n".join(combined_text_parts)
    output_path = RAW_DIR / "combined_novels.txt"
    output_path.write_text(combined_text, encoding="utf-8")

    print("Dataset ready!")
    print(f"Saved to: {output_path}")
    print(f"Total characters: {len(combined_text):,}")
    print(f"Approx size: {len(combined_text.encode('utf-8'))/1024/1024:.2f} MB")
