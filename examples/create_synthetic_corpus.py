#!/usr/bin/env python3
"""Create a deterministic, synthetic text corpus for local RAG smoke checks."""

from argparse import ArgumentParser
from pathlib import Path

SECTION_COUNT = 25
WORDS_PER_SECTION = 260
SENSORS = ("amber beacon", "thermal gauge", "wind vane", "pressure relay", "water probe")
ZONES = ("north ridge", "quiet basin", "eastern shelf", "lower tunnel", "glass garden")

CONTEXT = (
    "This synthetic record is part of a fictional observatory handbook and contains no real operational data.",
    "Operators compare the reading with the previous shift, record the instrument revision, and keep the original value unchanged.",
    "The daily checklist records the observation window, the location, and whether the calibration seal remained intact.",
    "A second reviewer checks the unit label before the reading is copied into the station summary.",
    "The archive keeps measurements separate by instrument so that a search result can be traced to its source section.",
    "Maintenance staff note unusual weather, nearby equipment changes, and any interruption in the observation schedule.",
    "The handbook distinguishes a measured value from an explanation; an explanation is not added unless the log supports it.",
    "For this example, all identifiers, locations, and values are invented and are intended only for testing local indexing.",
    "A text search should return the section containing the instrument identifier and its stated calibration threshold.",
    "The synthetic station uses the same review procedure for every zone, while each instrument retains its own record.",
    "When a reading is repeated, both the original observation and the later check remain visible in the audit trail.",
    "The archive stores plain text, section headings, and line positions; it does not execute or fetch anything in this file.",
)


def section(number: int) -> str:
    sensor = SENSORS[(number - 1) % len(SENSORS)]
    zone = ZONES[(number - 1) % len(ZONES)]
    identifier = f"ORBIT-{number:03d}"
    threshold = 12 + (number * 7) % 31
    lead = (
        f"Calibration record {identifier} in the {zone} sets the {sensor} threshold to "
        f"{threshold} units. This value is a fictional reference fact for the local retrieval example."
    )
    words = lead.split()
    cursor = number % len(CONTEXT)
    while len(words) < WORDS_PER_SECTION:
        sentence = CONTEXT[cursor % len(CONTEXT)].format(
            identifier=identifier,
            sensor=sensor,
            zone=zone,
            threshold=threshold,
        )
        words.extend(sentence.split())
        cursor += 1
    return f"## Section {number:02d} — {identifier}: {sensor}\n\n{' '.join(words[:WORDS_PER_SECTION])}\n"


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="path for the generated UTF-8 Markdown file")
    output = parser.parse_args().output
    output.parent.mkdir(parents=True, exist_ok=True)
    text = "# Synthetic observatory handbook\n\n" + "\n".join(
        section(number) for number in range(1, SECTION_COUNT + 1)
    )
    output.write_text(text, encoding="utf-8")
    print(f"Wrote {SECTION_COUNT} sections of {WORDS_PER_SECTION} words each to {output}")


if __name__ == "__main__":
    main()
