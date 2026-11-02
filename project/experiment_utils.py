"""Спільне збереження й друк таблиць обов'язкових експериментів."""

import csv
from pathlib import Path


def write_csv(path, headers, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(headers)
        writer.writerows(rows)


def print_table(headers, rows):
    def display(value):
        return f"{value:.9g}" if isinstance(value, float) else str(value)

    text = [[display(value) for value in row] for row in rows]
    widths = [max(len(str(h)), *(len(row[i]) for row in text))
              for i, h in enumerate(headers)]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def line(row):
        print("| " + " | ".join(str(v).ljust(widths[i])
                                 for i, v in enumerate(row)) + " |")

    print(border)
    line(headers)
    print(border)
    for row in text:
        line(row)
    print(border)


def report(path, headers, rows):
    print_table(headers, rows)
    write_csv(path, headers, rows)
    return rows
