#!/usr/bin/env python3
"""Tag training examples with AI-SETT categories and build balanced samples.

Categories (matching AI-SETT framework):
  1. Knowledge        7. Metacognition
  2. Reasoning        8. Emotional Intelligence
  3. Calibration      9. Pedagogy
  4. Generation      10. Learning
  5. Understanding   11. Boundaries
  6. Tool Use        12. Interaction
"""

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# AI-SETT categories
CATEGORIES = [
    "Knowledge", "Reasoning", "Calibration", "Generation",
    "Understanding", "Tool Use", "Metacognition",
    "Emotional Intelligence", "Pedagogy", "Learning",
    "Boundaries", "Interaction"
]

# File-to-category mapping (primary categories per source file)
FILE_CATEGORIES = {
    "boundaries_direct.jsonl": ["Boundaries"],
    "boundaries_gen.jsonl": ["Boundaries"],
    "calibration_zpd.jsonl": ["Calibration"],
    "emotional_intelligence_gen.jsonl": ["Emotional Intelligence"],
    "generation_gen.jsonl": ["Generation"],
    "interaction_gen.jsonl": ["Interaction"],
    "learning_gen.jsonl": ["Learning"],
    "metacognition_direct.jsonl": ["Metacognition"],
    "metacognition_extra.jsonl": ["Metacognition"],
    "metacognition_gen.jsonl": ["Metacognition"],
    "reasoning_gen.jsonl": ["Reasoning"],
    "teaching_gen.jsonl": ["Pedagogy"],
    "teaching_extra.jsonl": ["Pedagogy"],
    "pedagogy_gen.jsonl": ["Pedagogy"],
    "pedagogy_extra.jsonl": ["Pedagogy"],
    "tool_use_direct.jsonl": ["Tool Use"],
    "tool_use_gen.jsonl": ["Tool Use"],
    "understanding_gen.jsonl": ["Understanding"],
}

# These files have mixed categories — we'll use content-based tagging
MIXED_FILES = {
    "reasoning_learning_interaction.jsonl",
    "reasoning_tooluse_extra.jsonl",
    "learning_ei_extra.jsonl",
    "weak_areas_boost.jsonl",
    "gemini_converted.jsonl",
    "harvested_combined.jsonl",
}

# Keyword patterns for content-based category detection
CATEGORY_SIGNALS = {
    "Knowledge": [
        r"(?i)\b(fact|history|capital of|population|GDP|geography|science fact|who (was|is|invented)|when (was|did)|where is)\b",
    ],
    "Reasoning": [
        r"(?i)\b(logic|syllogism|premise|therefore|deduc|infer|probability|proof|if.*then|fallacy|argument|valid|causal|cause.{0,10}effect|counterfactual|fermi|estimat)\b",
        r"(?i)\b(step.by.step|work through|let me (solve|calculate|reason)|puzzle|brain teaser)\b",
    ],
    "Calibration": [
        r"(?i)\b(format|tone|style|response length|adjust my|write (in|as|like)|formal|casual|bullet point|paragraph|concise|verbose)\b",
    ],
    "Generation": [
        r"(?i)\b(write (a|an|me|this)|compose|draft|create (a|an)|generate|README|documentation|email|letter|story|poem|code.*(function|class|script))\b",
    ],
    "Understanding": [
        r"(?i)\b(summarize|summary|main point|key takeaway|what does.*mean|interpret|comprehension|analyze this|extract)\b",
    ],
    "Tool Use": [
        r"(?i)\b(API|REST|endpoint|curl|request|database|SQL|query|pandas|CSV|file (handling|operation)|bash|command line|script|deploy|docker|server)\b",
        r"(?i)\b(import (requests|pandas|os|json)|def \w+\(|SELECT.*FROM|CREATE TABLE)\b",
    ],
    "Metacognition": [
        r"(?i)\b(I('m| am) not (sure|certain|confident)|I (believe|think) but|my (confidence|uncertainty)|I should (flag|note|acknowledge)|bias|limitation|self-aware|calibrat)\b",
        r"(?i)\b(let me (reconsider|reflect|correct)|I was wrong|my earlier|upon reflection|transparent about)\b",
    ],
    "Emotional Intelligence": [
        r"(?i)\b(I (hear|understand|appreciate) (that|your|you)|empathy|emotion|feel(ing|s)?|grief|frustrat|anxiet|depress|stress|overwhelm|supportive)\b",
        r"(?i)\b(that must be|it('s| is) okay to|you('re| are) not alone|take care of yourself)\b",
    ],
    "Pedagogy": [
        r"(?i)\b(analogy|metaphor|like a|think of it as|scaffold|let me (explain|teach|walk you)|beginner|intermediate|advanced|misconception|common mistake)\b",
        r"(?i)\b(does (this|that) make sense|try (this|it) yourself|exercise|practice|quiz|review your)\b",
    ],
    "Learning": [
        r"(?i)\b(adapt|adjust|I('ll| will) (change|modify|switch)|you (said|asked|mentioned)|based on your (feedback|preference)|let me try (again|differently))\b",
        r"(?i)\b(follow.*instruction|as you requested|in the format you|here('s| is) the (revised|updated))\b",
    ],
    "Boundaries": [
        r"(?i)\b(I can('t| not) (help|assist|provide)|refus|decline|inappropriate|illegal|unethical|harmful|dangerous|security (threat|vulnerab)|phishing|injection|malware)\b",
        r"(?i)\b(SAFE|PHISHING|INJECTION|VISHING|BLOCK|classification)\b",
    ],
    "Interaction": [
        r"(?i)\b(could you clarify|what (specifically|exactly)|narrow.*down|which (of these|part)|let me (ask|understand)|multiple (request|question))\b",
        r"(?i)\b(you('re| are) right|good point|I (should have|missed)|thanks for (the correction|pointing))\b",
    ],
}


def detect_categories(messages):
    """Detect categories from message content using keyword patterns."""
    detected = set()

    # Combine all message text for scanning
    text = " ".join(m.get("content", "") for m in messages)

    for category, patterns in CATEGORY_SIGNALS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                detected.add(category)
                break

    return list(detected) if detected else ["Uncategorized"]


def tag_examples(data_dir):
    """Tag all examples with categories. Returns list of tagged examples."""
    tagged = []

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".jsonl") or fname in ("all_training_data.jsonl", "tagged_training_data.jsonl"):
            continue

        fpath = os.path.join(data_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    messages = obj.get("messages", [])
                    if len(messages) < 2:
                        continue

                    # Determine categories
                    if fname in FILE_CATEGORIES:
                        # Known file — use file mapping as primary
                        categories = list(FILE_CATEGORIES[fname])
                        # Also detect additional categories from content
                        content_cats = detect_categories(messages)
                        for c in content_cats:
                            if c not in categories and c != "Uncategorized":
                                categories.append(c)
                    else:
                        # Mixed/unknown file — detect from content
                        categories = detect_categories(messages)

                    tagged.append({
                        "messages": messages,
                        "metadata": {
                            "categories": categories,
                            "primary_category": categories[0],
                            "source_file": fname,
                        }
                    })
                except (json.JSONDecodeError, KeyError):
                    pass

    return tagged


def print_distribution(tagged):
    """Print category distribution."""
    cat_counts = Counter()
    for ex in tagged:
        for cat in ex["metadata"]["categories"]:
            cat_counts[cat] += 1

    primary_counts = Counter(ex["metadata"]["primary_category"] for ex in tagged)

    print(f"\n{'='*60}")
    print(f"CATEGORY DISTRIBUTION ({len(tagged)} total examples)")
    print(f"{'='*60}")
    print(f"\n{'Category':<25} {'Primary':>8} {'Any':>8}")
    print(f"{'-'*25} {'-'*8} {'-'*8}")
    for cat in CATEGORIES:
        print(f"{cat:<25} {primary_counts.get(cat, 0):>8} {cat_counts.get(cat, 0):>8}")
    uncat = primary_counts.get("Uncategorized", 0)
    if uncat:
        print(f"{'Uncategorized':<25} {uncat:>8}")
    print(f"{'-'*25} {'-'*8} {'-'*8}")
    print(f"{'TOTAL (unique examples)':<25} {len(tagged):>8}")


def balanced_sample(tagged, target_per_category=None, seed=42):
    """Create a balanced training set.

    Strategy:
    - For categories with enough examples: sample target_per_category
    - For categories with fewer: use all available + oversample
    - Returns deduplicated list
    """
    random.seed(seed)

    # Group by primary category
    by_category = defaultdict(list)
    for ex in tagged:
        by_category[ex["metadata"]["primary_category"]].append(ex)

    if target_per_category is None:
        # Default: use the median category size
        sizes = [len(v) for k, v in by_category.items() if k != "Uncategorized"]
        target_per_category = sorted(sizes)[len(sizes) // 2] if sizes else 20

    print(f"\nTarget per category: {target_per_category}")

    sampled = []
    seen = set()

    for cat in CATEGORIES:
        examples = by_category.get(cat, [])
        if not examples:
            print(f"  {cat}: 0 available, 0 sampled (NO DATA)")
            continue

        if len(examples) >= target_per_category:
            chosen = random.sample(examples, target_per_category)
        else:
            # Use all + oversample (repeat examples)
            chosen = list(examples)
            while len(chosen) < target_per_category:
                chosen.append(random.choice(examples))

        added = 0
        for ex in chosen:
            key = json.dumps(ex["messages"], sort_keys=True)
            if key not in seen:
                seen.add(key)
                sampled.append(ex)
                added += 1
            else:
                # Duplicate from oversampling — still add for balance
                sampled.append(ex)
                added += 1

        print(f"  {cat}: {len(examples)} available, {added} sampled")

    # Add uncategorized
    for ex in by_category.get("Uncategorized", []):
        key = json.dumps(ex["messages"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            sampled.append(ex)

    random.shuffle(sampled)
    return sampled


def write_tagged(tagged, output_path):
    """Write tagged examples to JSONL."""
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in tagged:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def write_training_file(examples, output_path):
    """Write clean training file (messages only, no metadata)."""
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")


def write_per_category(tagged, output_dir):
    """Write one file per category."""
    os.makedirs(output_dir, exist_ok=True)
    by_cat = defaultdict(list)
    for ex in tagged:
        for cat in ex["metadata"]["categories"]:
            by_cat[cat].append(ex)

    for cat, examples in sorted(by_cat.items()):
        safe_name = cat.lower().replace(" ", "_")
        fpath = os.path.join(output_dir, f"{safe_name}.jsonl")
        with open(fpath, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")
        print(f"  {cat}: {len(examples)} examples -> {safe_name}.jsonl")


def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))

    print("Tagging examples...")
    tagged = tag_examples(data_dir)
    print(f"Tagged {len(tagged)} examples.")

    # Print distribution
    print_distribution(tagged)

    # Write tagged file (with metadata)
    tagged_path = os.path.join(data_dir, "tagged_training_data.jsonl")
    write_tagged(tagged, tagged_path)
    print(f"\nTagged data: {tagged_path}")

    # Write per-category files
    cat_dir = os.path.join(data_dir, "by_category")
    print(f"\nWriting per-category files to {cat_dir}/")
    write_per_category(tagged, cat_dir)

    # Create balanced sample
    print(f"\nCreating balanced sample...")
    balanced = balanced_sample(tagged)
    balanced_path = os.path.join(data_dir, "balanced_training_data.jsonl")
    write_training_file(balanced, balanced_path)
    print(f"\nBalanced training set: {balanced_path} ({len(balanced)} examples)")

    # Also write the full unbalanced set (messages only)
    full_path = os.path.join(data_dir, "all_training_data.jsonl")
    write_training_file(tagged, full_path)
    print(f"Full training set: {full_path} ({len(tagged)} examples)")


if __name__ == "__main__":
    main()
