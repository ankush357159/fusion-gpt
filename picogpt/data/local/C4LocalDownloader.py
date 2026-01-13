from datasets import load_dataset
import os
import json


class C4LocalDownloader:
    def __init__(self, save_dir="./c4_training_data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def download(self, max_size_mb=1, split="train", output_file="c4_train_1mb.jsonl"):
        output_path = os.path.join(self.save_dir, output_file)

        max_size_bytes = max_size_mb * 1024 * 1024
        total_size = 0
        doc_count = 0
        
        # allenai/c4 - allenai = organization on Hugging Face Hub, c4 = dataset name
        ds = load_dataset("allenai/c4", "en", split=split, streaming=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for example in ds:
                text = example["text"]
                url = example.get("url", "")
                timestamp = example.get("timestamp", "")

                json_obj = {"text": text, "url": url, "timestamp": timestamp}
                line = json.dumps(json_obj) + "\n"
                line_size = len(line.encode("utf-8"))

                if total_size + line_size > max_size_bytes:
                    break

                f.write(line)
                total_size += line_size
                doc_count += 1

        print(f"Saved {doc_count} docs ({total_size / 1024 / 1024:.2f} MB)")
        return output_path


# Example usage
if __name__ == "__main__":
    # Download C4 data to local directory
    downloader = C4LocalDownloader(save_dir="./c4_training_data")
    data_file = downloader.download(
        max_size_mb=1, split="train", output_file="c4_train_1mb.jsonl"
    )
    print("\n" + "=" * 50)
    print("Download complete! Now creating DataLoader...")
    print("=" * 50 + "\n")
