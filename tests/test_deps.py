"""Do the requirements files cover what this repo imports, and nothing more.

Two repos in this account have shipped a requirements file that existed and did not
cover what the code imported. Checking that the file is present is not the check. This
walks every .py file with ast and compares the third party imports it finds against the
union of the declared packages.

The distribution name and the import name are different for most of what is here, so the
map below is required rather than optional. Without it PyYAML reads as undeclared while
requirements.txt declares it.
"""

import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIREMENTS = ["requirements.txt", "requirements-model.txt"]

# distribution name on the left, the name you actually import on the right
PROVIDES = {
    "pyyaml": ["yaml"],
    "scikit-learn": ["sklearn"],
}

LOCAL = {"featurestore", "tests", "scripts"}

SKIP_DIRS = {".git", "data", "__pycache__", ".venv", "venv"}


def python_files():
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in names:
            if name.endswith(".py"):
                yield os.path.join(base, name)


def declared():
    found = {}
    for name in REQUIREMENTS:
        path = os.path.join(ROOT, name)
        packages = set()
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                for sep in ("==", ">=", "<=", "~=", ">", "<"):
                    if sep in line:
                        line = line.split(sep, 1)[0]
                        break
                packages.add(line.strip().lower())
        found[name] = packages
    return found


def imported():
    out = {}
    for path in python_files():
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for name in names:
                if name and name not in sys.stdlib_module_names and name not in LOCAL:
                    out.setdefault(name, set()).add(os.path.relpath(path, ROOT))
    return out


def provided(all_declared):
    names = set()
    for packages in all_declared.values():
        for package in packages:
            names.update(PROVIDES.get(package, [package]))
    return names


def check_every_requirements_file_was_read():
    """A coverage check that reads an empty list passes on nothing.

    This is the reason the list above cannot quietly become empty. It also fails when a
    requirements file is added to the repo and not to REQUIREMENTS.
    """
    on_disk = sorted(
        n for n in os.listdir(ROOT) if n.startswith("requirements") and n.endswith(".txt")
    )
    assert on_disk == sorted(REQUIREMENTS), "on disk {} against declared {}".format(
        on_disk, sorted(REQUIREMENTS)
    )
    all_declared = declared()
    assert all_declared, "no requirements files were read"
    for name, packages in all_declared.items():
        assert packages, "{} declared nothing".format(name)


def check_nothing_is_imported_that_is_undeclared():
    have = provided(declared())
    missing = {}
    for name, where in imported().items():
        if name not in have:
            missing[name] = sorted(where)
    assert not missing, "undeclared imports: {}".format(missing)


def check_nothing_is_declared_that_nobody_imports():
    used = set(imported())
    unused = []
    for name, packages in declared().items():
        for package in packages:
            if not set(PROVIDES.get(package, [package])) & used:
                unused.append("{} in {}".format(package, name))
    assert not unused, "declared and never imported: {}".format(unused)


def check_the_suite_itself_needs_only_the_base_file():
    """A clone installing requirements.txt alone must be able to run every check.

    tests/ may not import anything that only requirements-model.txt provides, because
    the whole argument for splitting the files is that the suite runs without the heavy
    half. Driven off the file list rather than a remembered rule.
    """
    extra = set()
    for package in declared()["requirements-model.txt"]:
        extra.update(PROVIDES.get(package, [package]))
    for name, where in imported().items():
        if name not in extra:
            continue
        offenders = [w for w in where if w.startswith("tests" + os.sep) or w.startswith("featurestore" + os.sep)]
        assert not offenders, "{} is only in the model file and is imported by {}".format(
            name, offenders
        )


def check_the_scanner_found_something():
    """A walker pointed at the wrong directory reports no problems, cheerfully."""
    files = list(python_files())
    assert len(files) >= 8, "only found {} python files, the walk is wrong".format(len(files))
    assert imported(), "no third party imports found at all, the parser is wrong"
