"""Fetch WLASL100 and Google ISLR into data/."""

from asl_benchmark import load


def main() -> None:
    for source in ("wlasl100", "google_islr"):
        dataset = load(source, "data")
        print(
            f"{dataset.source}: {len(dataset.samples)} samples, "
            f"{len(dataset.glosses)} glosses -> {dataset.root}"
        )


if __name__ == "__main__":
    main()
