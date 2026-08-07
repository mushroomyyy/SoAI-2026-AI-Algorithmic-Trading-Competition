"""
The competition's hard requirements, encoded as tests.

Every assertion here maps to a stated rule in the official brief or template
README. The failure mode these guard against is unusually nasty: the README
says missing or incompatible dependencies "will cause your submission to be
skipped", i.e. we would be disqualified silently, with no error surfaced back
to us and no chance to fix it after the deadline.

Run: pytest -q tests/
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_FILE = REPO_ROOT / "strategies" / "strategy.py"

# Official environment accepts minute-, hourly- and daily-level cadences only.
# Sub-minute (tick/second) scheduling is rejected during verification.
ALLOWED_SLEEPTIME = re.compile(r"^\d+[MHD]$")


def test_strategy_is_importable_at_official_path():
    """Organizers run `from strategies.strategy import Strategy`, exactly."""
    from strategies.strategy import Strategy

    assert Strategy.__name__ == "Strategy"


def test_strategy_subclasses_lumibot_strategy():
    """The brief requires inheriting from lumibot.strategies.Strategy."""
    from lumibot.strategies import Strategy as LumibotStrategy

    from strategies.strategy import Strategy

    assert issubclass(Strategy, LumibotStrategy)


def test_strategy_implements_required_lifecycle_methods():
    """initialize() and on_trading_iteration() must be our own, not inherited stubs."""
    from strategies.strategy import Strategy

    for method in ("initialize", "on_trading_iteration"):
        assert method in vars(Strategy), f"{method} must be defined on our Strategy"


def _declared_sleeptimes() -> list[str]:
    """
    Every cadence value the strategy could assign to ``self.sleeptime``.

    Checks both spellings, because the assignment may be a literal
    (``self.sleeptime = "60M"``) or a module constant
    (``self.sleeptime = SLEEPTIME``). An earlier version of this test only
    matched literals, so refactoring the value into a constant silently
    disarmed the check -- exactly the kind of quiet regression that would let a
    rejected cadence reach the organizers.
    """
    source = STRATEGY_FILE.read_text()
    values = re.findall(r"""self\.sleeptime\s*=\s*["']([^"']+)["']""", source)

    names = re.findall(r"self\.sleeptime\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", source)
    if names:
        import strategies.strategy as strategy_module

        for name in names:
            resolved = getattr(strategy_module, name, None)
            assert isinstance(resolved, str), (
                f"self.sleeptime is assigned from {name!r}, which does not resolve "
                f"to a string constant on the module"
            )
            values.append(resolved)

    return values


def test_sleeptime_is_an_allowed_cadence():
    """Sub-minute scheduling is auto-rejected by the execution environment."""
    found = _declared_sleeptimes()

    assert found, "strategies/strategy.py must assign self.sleeptime"
    for value in found:
        assert not value.upper().endswith("S"), (
            f"sub-minute sleeptime {value!r} will be rejected at verification"
        )
        assert ALLOWED_SLEEPTIME.match(value), (
            f"sleeptime {value!r} is not a minute/hour/day cadence"
        )


def test_all_dependencies_are_pinned():
    """The brief requires 'All deps pinned in requirements.txt'."""
    lines = (REPO_ROOT / "requirements.txt").read_text().splitlines()
    unpinned = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#") and "==" not in line
    ]

    assert not unpinned, f"unpinned dependencies: {unpinned}"


def test_no_hardcoded_absolute_paths():
    """Absolute paths break reproducibility on the organizers' machine."""
    offenders = []
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in {".venv", "research", ".git"} for part in path.parts):
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if re.search(r"""["']/(Users|home)/""", line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}")

    assert not offenders, f"hard-coded absolute paths: {offenders}"


def test_no_env_file_committed():
    """Secrets must stay out of the repo."""
    assert not (REPO_ROOT / ".env").exists(), ".env must never be committed"


@pytest.mark.parametrize("required", ["strategies/strategy.py", "requirements.txt", "README.md"])
def test_required_files_exist(required):
    """Files the official environment relies on."""
    assert (REPO_ROOT / required).exists(), f"{required} is required by the organizers"
