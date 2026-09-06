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

BASE = "requirements.txt"
REQUIREMENTS = [
    BASE,
    "requirements-model.txt",
    "requirements-online.txt",
    "requirements-spark.txt",
]

# distribution name on the left, the name you actually import on the right
PROVIDES = {
    "pyyaml": ["yaml"],
    "scikit-learn": ["sklearn"],
}

SKIP_DIRS = {".git", "data", "__pycache__", ".venv", "venv"}


def python_files():
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in names:
            if name.endswith(".py"):
                yield os.path.join(base, name)


def local_names():
    """Names that resolve to something in this repo rather than to a package.

    Derived from the tree rather than listed. The listed version held the three top
    level directories, so a module imported from beside its importer read as an
    undeclared third party package. `tests/run_all.py` importing `runner` is the case
    that found it, and a list would have needed editing every time that happened.

    A repo file whose stem matches a real distribution's import name would be treated as
    local here and would hide a missing declaration.
    `check_no_local_module_shadows_a_declared_package` is what refuses that.
    """
    names = set()
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        names.update(dirs)
        names.update(f[:-3] for f in files if f.endswith(".py"))
    return names


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


def function_scoped(tree):
    """The ids of import nodes that sit inside a function body.

    A module level import runs when the file is imported and a function level one runs
    when the function is called. That difference is the whole reason a module can name a
    heavy package and still be importable by a suite that has not installed it.
    """
    inside = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    inside.add(id(sub))
    return inside


def imported(module_level_only=False):
    out = {}
    local = local_names()
    for path in python_files():
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        deferred = function_scoped(tree)
        for node in ast.walk(tree):
            if module_level_only and id(node) in deferred:
                continue
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for name in names:
                if name and name not in sys.stdlib_module_names and name not in local:
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


def check_the_local_name_scan_found_something():
    """A scan pointed at the wrong tree returns nothing and then everything reads as
    third party, which fails loudly. The dangerous direction is the other one, where a
    stray directory makes the set wide enough to swallow a real package."""
    names = local_names()
    assert "featurestore" in names, "the local scan missed the library: {}".format(
        sorted(names)
    )
    assert "spec" in names, "the local scan is collecting directories and not modules"


def check_no_local_module_shadows_a_declared_package():
    """A file named yaml.py would make `import yaml` read as local and the declaration
    check would then pass on a repo that had stopped declaring PyYAML. Deriving the local
    set from the tree is what makes this possible, so it ships with the derivation."""
    clash = sorted(local_names() & provided(declared()))
    assert not clash, "these repo modules share a name with a declared package: {}".format(
        clash
    )


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

    Nothing under tests/ or featurestore/ may import a package from an optional
    requirements file at module level, because the whole argument for splitting the
    files is that the suite runs without the heavy halves. A function level import is
    fine and is what `featurestore/spark_job.py` uses.

    Driven off every optional file rather than off the one that existed when this was
    written. The first version named requirements-model.txt and would have said nothing
    about the spark one.
    """
    optional = [name for name in REQUIREMENTS if name != BASE]
    assert optional, "there are no optional requirements files, so this checks nothing"
    all_declared = declared()
    extra = set()
    for name in optional:
        for package in all_declared[name]:
            extra.update(PROVIDES.get(package, [package]))
    for name, where in imported(module_level_only=True).items():
        if name not in extra:
            continue
        offenders = [
            w
            for w in where
            if w.startswith("tests" + os.sep) or w.startswith("featurestore" + os.sep)
        ]
        assert not offenders, "{} is optional and is imported at module level by {}".format(
            name, offenders
        )


def check_pyspark_is_never_imported_at_module_level_in_the_library():
    """The arrangement that makes the split work, asserted rather than assumed.

    `featurestore/spark_job.py` names pyspark and stays importable without it. Moving
    either import to the top of that file breaks a clone that installed only the base
    requirements, and every check in this repo would still pass here because this
    sandbox has pyspark installed.
    """
    deferred = imported().get("pyspark", set())
    assert deferred, "pyspark is not imported anywhere, this check has no subject"
    at_module_level = imported(module_level_only=True).get("pyspark", set())
    assert not at_module_level, "pyspark is imported at module level by {}".format(
        sorted(at_module_level)
    )


def check_the_scanner_found_something():
    """A walker pointed at the wrong directory reports no problems, cheerfully."""
    files = list(python_files())
    assert len(files) >= 8, "only found {} python files, the walk is wrong".format(len(files))
    assert imported(), "no third party imports found at all, the parser is wrong"
