"""
Stretch Feature: Test Harness / Evaluation Script

Standalone CLI that runs the full VibeFinder evaluation suite on any
catalog and prints a structured PASS/FAIL report with confidence ratings.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --catalog data/songs_extended.csv
    python scripts/run_evaluation.py --min-relevance 0.5 --min-diversity 0.6
    python scripts/run_evaluation.py --quiet

Exit codes:
    0  all profiles passed all thresholds
    1  one or more profiles failed

Example output:
    VibeFinder Evaluation Harness
    ==============================
    Catalog : data/songs.csv (10 songs)
    Profiles: 5
    Thresholds: relevance>=0.50  diversity>=0.60  consistency>=0.90  confidence>=0.50

    PROFILE: High-Energy Pop Fan
      relevance   0.6143  [PASS >= 0.50]
      diversity   0.8000  [PASS >= 0.60]
      consistency 1.0000  [PASS >= 0.90]
      confidence  0.7000  [PASS >= 0.50]
      STATUS: PASS
    ...
    SUMMARY
    =======
    Profiles passed: 5 / 5
    ...
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import EvaluationSuite, EVAL_PROFILES
from src.recommender import load_songs

# Default pass thresholds
DEFAULT_MIN_RELEVANCE   = 0.50
DEFAULT_MIN_DIVERSITY   = 0.60
DEFAULT_MIN_CONSISTENCY = 0.90
DEFAULT_MIN_CONFIDENCE  = 0.50


def _pass_fail(value: float, threshold: float) -> str:
    status = "PASS" if value >= threshold else "FAIL"
    return f"[{status} >= {threshold:.2f}]"


def run_harness(
    catalog_path: str,
    min_relevance: float   = DEFAULT_MIN_RELEVANCE,
    min_diversity: float   = DEFAULT_MIN_DIVERSITY,
    min_consistency: float = DEFAULT_MIN_CONSISTENCY,
    min_confidence: float  = DEFAULT_MIN_CONFIDENCE,
    quiet: bool            = False,
) -> bool:
    """
    Runs the evaluation suite and prints a structured report.
    Returns True if all profiles passed, False otherwise.
    """
    songs = load_songs(catalog_path)
    suite = EvaluationSuite(catalog_path)

    if not quiet:
        print()
        print("VibeFinder Evaluation Harness")
        print("=" * 60)
        print(f"Catalog  : {catalog_path} ({len(songs)} songs)")
        print(f"Profiles : {len(EVAL_PROFILES)}")
        print(
            f"Thresholds: "
            f"relevance>={min_relevance}  "
            f"diversity>={min_diversity}  "
            f"consistency>={min_consistency}  "
            f"confidence>={min_confidence}"
        )
        print()

    report = suite.run_all()
    all_passed = True
    profile_statuses = []

    for entry in report["results"]:
        name     = entry["profile_name"]
        rel      = entry["mean_relevance_score"]
        div      = entry["diversity_index"]
        con      = entry["consistency_score"]
        conf     = entry["mean_confidence_score"]

        checks = {
            "relevance":   (rel,  min_relevance),
            "diversity":   (div,  min_diversity),
            "consistency": (con,  min_consistency),
            "confidence":  (conf, min_confidence),
        }
        profile_pass = all(val >= thr for val, thr in checks.values())
        all_passed = all_passed and profile_pass
        profile_statuses.append((name, profile_pass))

        if not quiet:
            print(f"PROFILE: {name}")
            for metric, (val, thr) in checks.items():
                pf = _pass_fail(val, thr)
                print(f"  {metric:<12} {val:.4f}  {pf}")
            status_label = "PASS" if profile_pass else "FAIL"
            print(f"  STATUS: {status_label}")
            print()

    # Summary
    summary = report["summary"]
    passed_count = sum(1 for _, ok in profile_statuses if ok)
    total_count  = len(profile_statuses)

    overall_checks = {
        "avg_relevance":   (summary["avg_relevance"],   min_relevance),
        "avg_diversity":   (summary["avg_diversity"],   min_diversity),
        "avg_consistency": (summary["avg_consistency"], min_consistency),
        "avg_confidence":  (summary["avg_confidence"],  min_confidence),
    }
    overall_pass = all(v >= t for v, t in overall_checks.values())

    print("SUMMARY")
    print("=" * 60)
    print(f"Profiles passed  : {passed_count} / {total_count}")
    print()
    print("Aggregate metrics:")
    for metric, (val, thr) in overall_checks.items():
        pf = _pass_fail(val, thr)
        print(f"  {metric:<18} {val:.4f}  {pf}")
    print()

    for name, ok in profile_statuses:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}")

    print()
    final_label = "OVERALL: PASS" if overall_pass else "OVERALL: FAIL"
    print(final_label)
    print(f"Exit code: {0 if all_passed else 1}")

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VibeFinder CLI Evaluation Harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--catalog",
        default="data/songs.csv",
        help="Path to the song catalog CSV",
    )
    parser.add_argument(
        "--min-relevance",
        type=float,
        default=DEFAULT_MIN_RELEVANCE,
        help="Minimum passing mean relevance score",
    )
    parser.add_argument(
        "--min-diversity",
        type=float,
        default=DEFAULT_MIN_DIVERSITY,
        help="Minimum passing diversity index",
    )
    parser.add_argument(
        "--min-consistency",
        type=float,
        default=DEFAULT_MIN_CONSISTENCY,
        help="Minimum passing consistency score",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help="Minimum passing mean confidence score",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-profile output, show summary only",
    )
    args = parser.parse_args()

    passed = run_harness(
        catalog_path    = args.catalog,
        min_relevance   = args.min_relevance,
        min_diversity   = args.min_diversity,
        min_consistency = args.min_consistency,
        min_confidence  = args.min_confidence,
        quiet           = args.quiet,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
