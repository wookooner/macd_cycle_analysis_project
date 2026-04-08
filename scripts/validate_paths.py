from __future__ import annotations

from src.common.paths import PROJECT_PATHS


def main() -> int:
    PROJECT_PATHS.ensure_runtime_dirs()

    print("Path summary")
    for key, value in PROJECT_PATHS.summary().items():
        print(f"- {key}: {value}")

    issues = PROJECT_PATHS.validate()
    print("\nValidation")
    if not issues:
        print("- no blocking issues detected")
        return 0

    for issue in issues:
        print(f"- {issue}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
