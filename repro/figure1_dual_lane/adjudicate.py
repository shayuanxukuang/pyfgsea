#!/usr/bin/env python3
"""Compatibility entry point for the Figure 1 comparison."""

try:
    from .compare_results import *  # noqa: F401,F403
    from .compare_results import compare_results, main
except ImportError:  # pragma: no cover - direct script execution
    from compare_results import *  # type: ignore # noqa: F401,F403
    from compare_results import compare_results, main  # type: ignore

adjudicate = compare_results


if __name__ == "__main__":
    raise SystemExit(main())
