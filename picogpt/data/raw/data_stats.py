"""
Data Statistics for PicoGPT Dataset
Analyzes the combined novel dataset and individual books to provide useful insights.
"""

import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"


class TextStats:
    """Compute comprehensive statistics for text data."""

    def __init__(self, text: str):
        self.text = text
        self.tokens = self._tokenize()
        self.words = self._extract_words()
        self.sentences = self._extract_sentences()
        self.paragraphs = self._extract_paragraphs()

    def _tokenize(self) -> List[str]:
        """Basic tokenization by splitting on whitespace."""
        return self.text.split()

    def _extract_words(self) -> List[str]:
        """Extract words (alphabetic only, lowercased)."""
        words = re.findall(r"\b[a-zA-Z]+\b", self.text.lower())
        return words

    def _extract_sentences(self) -> List[str]:
        """Extract sentences using basic sentence boundary detection."""
        sentences = re.split(r"[.!?]+", self.text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _extract_paragraphs(self) -> List[str]:
        """Extract paragraphs by splitting on double newlines."""
        paragraphs = self.text.split("\n\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs

    def basic_stats(self) -> Dict[str, Any]:
        """Get basic text statistics."""
        return {
            "total_characters": len(self.text),
            "total_tokens": len(self.tokens),
            "total_words": len(self.words),
            "unique_words": len(set(self.words)),
            "total_sentences": len(self.sentences),
            "total_paragraphs": len(self.paragraphs),
            "avg_word_length": (
                np.mean([len(word) for word in self.words]) if self.words else 0
            ),
            "avg_sentence_length": (
                np.mean([len(sent.split()) for sent in self.sentences])
                if self.sentences
                else 0
            ),
            "avg_paragraph_length": (
                np.mean([len(para.split()) for para in self.paragraphs])
                if self.paragraphs
                else 0
            ),
            "type_token_ratio": (
                len(set(self.words)) / len(self.words) if self.words else 0
            ),
        }

    def character_frequency(self) -> Dict[str, int]:
        """Get character frequency distribution."""
        char_counts = Counter(self.text.lower())
        # Filter out whitespace and newlines for cleaner display
        char_counts = {
            k: v for k, v in char_counts.items() if k not in [" ", "\n", "\t", "\r"]
        }
        return dict(char_counts.most_common(50))

    def word_frequency(self, top_n: int = 50) -> Dict[str, int]:
        """Get most common words."""
        word_counts = Counter(self.words)
        return dict(word_counts.most_common(top_n))

    def word_length_distribution(self) -> Dict[int, int]:
        """Get distribution of word lengths."""
        lengths = [len(word) for word in self.words]
        length_counts = Counter(lengths)
        return dict(sorted(length_counts.items()))

    def lexical_diversity_stats(self) -> Dict[str, float]:
        """Calculate various lexical diversity measures."""
        unique_words = len(set(self.words))
        total_words = len(self.words)

        if total_words == 0:
            return {"ttr": 0, "mattr": 0, "vocabulary_size": 0}

        # Type-Token Ratio
        ttr = unique_words / total_words

        # Moving Average Type-Token Ratio (with window size 100)
        window_size = min(100, total_words)
        ttrs = []
        for i in range(0, total_words - window_size + 1, window_size // 2):
            window = self.words[i : i + window_size]
            window_ttr = len(set(window)) / len(window)
            ttrs.append(window_ttr)
        mattr = np.mean(ttrs) if ttrs else 0

        return {
            "ttr": ttr,
            "mattr": mattr,
            "vocabulary_size": unique_words,
            "hapax_legomena": sum(
                1 for count in Counter(self.words).values() if count == 1
            ),
        }


def analyze_individual_books() -> Dict[str, Dict]:
    """Analyze each book individually."""
    results = {}

    for path in sorted(RAW_DIR.glob("*.txt")):
        if path.name == "combined_novels.txt":
            continue

        book_name = path.stem
        print(f"Analyzing {book_name}...")

        try:
            text = path.read_text(encoding="utf-8")
            stats = TextStats(text)
            results[book_name] = {
                "basic_stats": stats.basic_stats(),
                "top_words": stats.word_frequency(20),
                "word_length_dist": stats.word_length_distribution(),
                "lexical_diversity": stats.lexical_diversity_stats(),
            }
        except Exception as e:
            print(f"Error analyzing {book_name}: {e}")
            results[book_name] = {"error": str(e)}

    return results


def analyze_combined_dataset() -> Dict:
    """Analyze the combined dataset."""
    combined_path = RAW_DIR / "combined_novels.txt"

    if not combined_path.exists():
        print("Combined dataset not found. Please run data.py first.")
        return {}

    print("Analyzing combined dataset...")
    text = combined_path.read_text(encoding="utf-8")
    stats = TextStats(text)

    return {
        "basic_stats": stats.basic_stats(),
        "character_frequency": stats.character_frequency(),
        "top_words": stats.word_frequency(50),
        "word_length_dist": stats.word_length_distribution(),
        "lexical_diversity": stats.lexical_diversity_stats(),
    }


def print_stats_summary(stats: Dict, title: str):
    """Print a formatted summary of statistics."""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

    if "error" in stats:
        print(f"Error: {stats['error']}")
        return

    basic = stats.get("basic_stats", {})
    print(f"Total Characters: {basic.get('total_characters', 0):,}")
    print(f"Total Tokens: {basic.get('total_tokens', 0):,}")
    print(f"Total Words: {basic.get('total_words', 0):,}")
    print(f"Unique Words: {basic.get('unique_words', 0):,}")
    print(f"Total Sentences: {basic.get('total_sentences', 0):,}")
    print(f"Total Paragraphs: {basic.get('total_paragraphs', 0):,}")
    print(f"Average Word Length: {basic.get('avg_word_length', 0):.2f}")
    print(f"Average Sentence Length: {basic.get('avg_sentence_length', 0):.2f} words")
    print(f"Type-Token Ratio: {basic.get('type_token_ratio', 0):.4f}")

    if "lexical_diversity" in stats:
        lex = stats["lexical_diversity"]
        print(f"Vocabulary Richness (MATTR): {lex.get('mattr', 0):.4f}")
        print(
            f"Hapax Legomena (words appearing once): {lex.get('hapax_legomena', 0):,}"
        )

    if "top_words" in stats:
        print(f"\nTop 10 Most Common Words:")
        for i, (word, count) in enumerate(list(stats["top_words"].items())[:10], 1):
            print(f"  {i:2d}. {word:15s} - {count:,}")


def create_visualizations(combined_stats: Dict, individual_stats: Dict):
    """Create visualizations for the data statistics."""
    try:
        import matplotlib.pyplot as plt

        # Word length distribution
        plt.figure(figsize=(15, 10))

        # Plot 1: Combined dataset word length distribution
        plt.subplot(2, 3, 1)
        if "word_length_dist" in combined_stats:
            lengths, counts = zip(*combined_stats["word_length_dist"].items())
            plt.bar(lengths, counts, alpha=0.7)
            plt.title("Word Length Distribution (Combined)")
            plt.xlabel("Word Length")
            plt.ylabel("Frequency")

        # Plot 2: Top words frequency (combined)
        plt.subplot(2, 3, 2)
        if "top_words" in combined_stats:
            words = list(combined_stats["top_words"].keys())[:15]
            counts = list(combined_stats["top_words"].values())[:15]
            plt.barh(words[::-1], counts[::-1])
            plt.title("Top 15 Most Common Words")
            plt.xlabel("Frequency")

        # Plot 3: Character frequency (combined)
        plt.subplot(2, 3, 3)
        if "character_frequency" in combined_stats:
            chars = list(combined_stats["character_frequency"].keys())[:20]
            freqs = list(combined_stats["character_frequency"].values())[:20]
            plt.bar(chars, freqs)
            plt.title("Top 20 Character Frequencies")
            plt.xlabel("Character")
            plt.ylabel("Frequency")
            plt.xticks(rotation=45)

        # Plot 4: Comparison of book lengths
        plt.subplot(2, 3, 4)
        book_names = []
        word_counts = []
        for book, stats in individual_stats.items():
            if "basic_stats" in stats:
                book_names.append(book.replace("_", " ").title())
                word_counts.append(stats["basic_stats"]["total_words"])

        if book_names:
            plt.barh(book_names, word_counts)
            plt.title("Book Length Comparison")
            plt.xlabel("Word Count")

        # Plot 5: Lexical diversity comparison
        plt.subplot(2, 3, 5)
        if book_names:
            ttrs = []
            for book, stats in individual_stats.items():
                if "basic_stats" in stats:
                    ttrs.append(stats["basic_stats"]["type_token_ratio"])

            plt.bar(range(len(book_names)), ttrs)
            plt.title("Lexical Diversity (TTR) by Book")
            plt.ylabel("Type-Token Ratio")
            plt.xticks(
                range(len(book_names)), [name[:10] for name in book_names], rotation=45
            )

        # Plot 6: Average sentence length comparison
        plt.subplot(2, 3, 6)
        if book_names:
            sent_lens = []
            for book, stats in individual_stats.items():
                if "basic_stats" in stats:
                    sent_lens.append(stats["basic_stats"]["avg_sentence_length"])

            plt.bar(range(len(book_names)), sent_lens)
            plt.title("Average Sentence Length by Book")
            plt.ylabel("Words per Sentence")
            plt.xticks(
                range(len(book_names)), [name[:10] for name in book_names], rotation=45
            )

        plt.tight_layout()
        plt.savefig(RAW_DIR / "data_statistics.png", dpi=300, bbox_inches="tight")
        print(f"\nVisualization saved to: {RAW_DIR / 'data_statistics.png'}")

    except ImportError:
        print("\nMatplotlib not available. Skipping visualizations.")
        print("Install with: pip install matplotlib")


def export_stats_to_file(combined_stats: Dict, individual_stats: Dict):
    """Export statistics to a text file."""
    output_path = RAW_DIR / "data_statistics_report.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("PicoGPT Dataset Statistics Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {Path(__file__).stat().st_mtime}\n\n")

        # Combined stats
        f.write("COMBINED DATASET STATISTICS\n")
        f.write("-" * 30 + "\n")
        if combined_stats and "basic_stats" in combined_stats:
            basic = combined_stats["basic_stats"]
            f.write(f"Total Characters: {basic['total_characters']:,}\n")
            f.write(f"Total Tokens: {basic['total_tokens']:,}\n")
            f.write(f"Total Words: {basic['total_words']:,}\n")
            f.write(f"Unique Words: {basic['unique_words']:,}\n")
            f.write(
                f"Vocabulary Size: {combined_stats['lexical_diversity']['vocabulary_size']:,}\n"
            )
            f.write(f"Type-Token Ratio: {basic['type_token_ratio']:.4f}\n")
            f.write(f"MATTR: {combined_stats['lexical_diversity']['mattr']:.4f}\n\n")

        # Individual book stats
        f.write("INDIVIDUAL BOOK STATISTICS\n")
        f.write("-" * 30 + "\n")
        for book_name, stats in individual_stats.items():
            if "basic_stats" in stats:
                f.write(f"\n{book_name.replace('_', ' ').title()}:\n")
                basic = stats["basic_stats"]
                f.write(f"  Words: {basic['total_words']:,}\n")
                f.write(f"  Unique Words: {basic['unique_words']:,}\n")
                f.write(f"  Sentences: {basic['total_sentences']:,}\n")
                f.write(f"  TTR: {basic['type_token_ratio']:.4f}\n")

    print(f"Statistics report exported to: {output_path}")


def main():
    """Main function to run all analyses."""
    print("PicoGPT Data Statistics Analysis")
    print("=" * 50)

    # Analyze individual books
    individual_stats = analyze_individual_books()

    # Print individual book summaries
    for book_name, stats in individual_stats.items():
        print_stats_summary(stats, f"{book_name.replace('_', ' ').title()} Statistics")

    # Analyze combined dataset
    combined_stats = analyze_combined_dataset()
    if combined_stats:
        print_stats_summary(combined_stats, "Combined Dataset Statistics")

    # Create visualizations
    if combined_stats and individual_stats:
        create_visualizations(combined_stats, individual_stats)
        export_stats_to_file(combined_stats, individual_stats)

    # Summary comparison table
    if individual_stats:
        print(f"\n{'='*80}")
        print(" BOOK COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"{'Book':<20} {'Words':<10} {'Unique':<8} {'TTR':<6} {'Avg Sent':<8}")
        print("-" * 80)
        for book_name, stats in individual_stats.items():
            if "basic_stats" in stats:
                basic = stats["basic_stats"]
                print(
                    f"{book_name.replace('_', ' ').title():<20} "
                    f"{basic['total_words']:<10,} "
                    f"{basic['unique_words']:<8,} "
                    f"{basic['type_token_ratio']:<6.3f} "
                    f"{basic['avg_sentence_length']:<8.1f}"
                )


if __name__ == "__main__":
    main()
