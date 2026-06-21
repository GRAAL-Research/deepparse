# Architecture tests for deepparse.
#
# These tests freeze the architectural invariants of the package using
# ArchUnitPython (https://github.com/LukasNiessen/ArchUnitPython). They analyze
# the source with the standard library `ast` module only (no torch, no model
# download) and are meant to run on a fast, dedicated CI job.
#
# The dependency graph between the ``deepparse`` sub-packages is a clean DAG with
# four layers:
#
#   Foundation  : errors, metrics, data_validation, pre_processing  (no internal deps)
#   Pipeline    : embeddings_models -> vectorizer -> converter ; network ; dataset_container
#   Orchestrator: parser            (depends on the whole pipeline)
#   Consumers   : comparer, app, cli (depend on parser, depended on by nobody)
#
# The ``ARCHITECTURE_DIAGRAM`` below is the executable specification of that DAG.
# Any new cross-package import that is not declared there makes
# ``test_subpackage_dependencies_match_architecture`` fail, which is how cycles,
# layer leaks and accidental coupling are caught.
import re

from archunitpython import assert_passes, metrics, project_files, project_slices

# Root analyzed by every rule. ArchUnitPython resolves relative/absolute imports
# under this path into a file-level dependency graph.
PACKAGE = "deepparse/"

# Maps each source file to its sub-package (the directory directly containing the
# file). Anchored on the end of the path (``$``) so it is independent of the
# absolute checkout location, including the ``work/deepparse/deepparse`` layout
# used by GitHub Actions. The six top-level modules (download_tools.py, ...) map
# to the slice ``deepparse`` and are treated as orphan slices (see below).
SUBPACKAGE_REGEX = re.compile(r"/([^/]+)/[^/]+\.py$")

# Executable specification of the allowed sub-package dependency graph. A
# dependency present in the code but absent from this diagram is a violation;
# adding a legitimate new edge means adding the arrow here on purpose.
ARCHITECTURE_DIAGRAM = """
@startuml
component [errors]
component [metrics]
component [data_validation]
component [pre_processing]
component [embeddings_models]
component [vectorizer]
component [converter]
component [network]
component [dataset_container]
component [parser]
component [comparer]
component [app]
component [cli]

[vectorizer] --> [embeddings_models]
[converter] --> [vectorizer]
[dataset_container] --> [data_validation]
[dataset_container] --> [errors]
[parser] --> [converter]
[parser] --> [vectorizer]
[parser] --> [embeddings_models]
[parser] --> [network]
[parser] --> [dataset_container]
[parser] --> [pre_processing]
[parser] --> [metrics]
[parser] --> [errors]
[comparer] --> [parser]
[app] --> [parser]
[cli] --> [parser]
[cli] --> [dataset_container]
@enduml
"""

# Sub-packages that must not depend on the orchestrator: the model/data pipeline
# has to stay usable below ``parser``.
PIPELINE_FOLDERS = ["**/network", "**/vectorizer", "**/converter", "**/embeddings_models"]


def test_subpackage_dependencies_match_architecture():
    # Master structural guardrail: the real cross-package imports must match the
    # declared DAG exactly. Catches cycles, layer leaks and accidental coupling.
    # Orphan slices (the top-level modules, not declared as components) are
    # ignored so only true sub-package edges are checked.
    rule = (
        project_slices(PACKAGE)
        .defined_by_regex(SUBPACKAGE_REGEX)
        .should()
        .ignoring_orphan_slices()
        .adhere_to_diagram(ARCHITECTURE_DIAGRAM)
    )
    assert_passes(rule)


def test_pipeline_layer_does_not_depend_on_parser():
    # Explicit, self-documenting version of the layering invariant: the pipeline
    # never imports the orchestrator. Redundant with the diagram on purpose, for
    # a clearer failure message.
    for folder in PIPELINE_FOLDERS:
        rule = project_files(PACKAGE).in_folder(folder).should_not().depend_on_files().in_folder("**/parser")
        assert_passes(rule)


def test_foundation_packages_do_not_depend_on_parser():
    # Foundation leaves must stay at the bottom of the stack.
    for folder in ["**/errors", "**/metrics", "**/data_validation", "**/pre_processing"]:
        rule = project_files(PACKAGE).in_folder(folder).should_not().depend_on_files().in_folder("**/parser")
        assert_passes(rule)


def test_factory_files_follow_naming_convention():
    # Every file whose name mentions "factory" must use the ``*_factory.py``
    # convention (e.g. vectorizer_factory.py, model_factory.py).
    rule = project_files(PACKAGE).with_name(re.compile(r".*factory.*\.py$")).should().have_name("*_factory.py")
    assert_passes(rule)


def test_error_modules_live_in_errors_package():
    # Exception modules (``*_error.py``) must stay grouped under ``errors/``.
    rule = project_files(PACKAGE).with_name("*_error.py").should().be_in_folder("**/errors")
    assert_passes(rule)


def test_no_file_exceeds_max_lines_of_code():
    # Guards against runaway files. Threshold sits above the current largest file
    # (parser/address_parser.py, ~1020 LOC) with headroom; tighten as the
    # orchestrator gets refactored.
    rule = metrics(PACKAGE).count().lines_of_code().should_be_below(1200)
    assert_passes(rule)


def test_classes_keep_reasonable_cohesion():
    # LCOM96b in [0, 1]; lower is more cohesive. Current worst class is < 0.9, so
    # 0.95 blocks fully-incohesive god classes without flagging existing code.
    rule = metrics(PACKAGE).lcom().lcom96b().should_be_below(0.95)
    assert_passes(rule)
