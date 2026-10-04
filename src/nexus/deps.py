"""
dep_checker.py -- NEXUS Dependency Checker
Checks all required and optional Python packages at startup.
Reports which packages are missing so the GUI can show clear
instructions instead of crashing.

Phase 33 of the NEXUS implementation plan.
"""

import importlib

# ---------------------------------------------------------------------------
# Package definitions
# ---------------------------------------------------------------------------

# Required packages: NEXUS cannot function without these.
# Format: (import_name, pip_name, description)
REQUIRED_PACKAGES = [
    ("llama_cpp",             "llama-cpp-python",    "LLM inference engine"),
    ("pdfplumber",            "pdfplumber",          "PDF file reading"),
    ("openpyxl",              "openpyxl",            "Excel file reading"),
    ("psutil",                "psutil",              "System monitoring"),
    ("rich",                  "rich",                "Formatted CLI output"),
    ("sentence_transformers", "sentence-transformers", "Document embeddings"),
    ("faiss",                 "faiss-cpu",           "Vector search index"),
]

# Optional packages: missing ones disable specific features but NEXUS still runs.
# Format: (import_name, pip_name, description, feature_it_enables)
OPTIONAL_PACKAGES = [
    ("pynput",                "pynput",              "Global hotkeys",            "Ctrl+Alt+N hotkey"),
    ("transformers",          "transformers",        "Model training framework", "LoRA training (experimental)"),
    ("peft",                  "peft",                "LoRA adapter framework",   "LoRA training (experimental)"),
    ("datasets",              "datasets",            "Dataset loading",          "LoRA training (experimental)"),
    ("accelerate",            "accelerate",          "Training acceleration",    "LoRA training (experimental)"),
]


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------
def check_dependencies():
    """
    Check all required and optional packages.

    Returns:
        Dict with keys:
            'missing_required': list of (pip_name, description) tuples
            'missing_optional': list of (pip_name, description, feature) tuples
            'available_optional': list of (pip_name, feature) tuples
    """
    missing_required = []
    missing_optional = []
    available_optional = []

    # Check required
    for import_name, pip_name, description in REQUIRED_PACKAGES:
        if not _can_import(import_name):
            missing_required.append((pip_name, description))

    # Check optional
    for import_name, pip_name, description, feature in OPTIONAL_PACKAGES:
        if _can_import(import_name):
            available_optional.append((pip_name, feature))
        else:
            missing_optional.append((pip_name, description, feature))

    return {
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "available_optional": available_optional,
    }


def _can_import(module_name):
    """
    Check if a module can be imported without side effects.

    Args:
        module_name: Dotted module name, e.g. 'sentence_transformers'.

    Returns:
        True if the module is importable, False otherwise.
    """
    try:
        importlib.import_module(module_name)
        return True
    except (ImportError, ModuleNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Install command helper
# ---------------------------------------------------------------------------
def get_install_command(packages):
    """
    Build a pip install command string for the given packages.

    Args:
        packages: List of pip package names (strings).

    Returns:
        The pip install command as a string, or empty if no packages.
    """
    if not packages:
        return ""
    return "pip install " + " ".join(packages)


def get_missing_required_message():
    """
    Return a human-readable error message for missing required packages.

    Returns:
        Error message string, or empty string if all required are installed.
    """
    result = check_dependencies()
    missing = result["missing_required"]

    if not missing:
        return ""

    lines = ["The following required packages are missing:\n"]
    for pip_name, desc in missing:
        lines.append(f"  - {pip_name}  ({desc})")

    pip_names = [p[0] for p in missing]
    lines.append(f"\nInstall with:\n  {get_install_command(pip_names)}")

    return "\n".join(lines)


def get_missing_optional_summary():
    """
    Return a short summary string listing disabled features.

    Returns:
        Summary string, or empty string if all optional are installed.
    """
    result = check_dependencies()
    missing = result["missing_optional"]

    if not missing:
        return ""

    disabled_features = sorted({feat for _, _, feat in missing})
    return "Disabled: " + ", ".join(disabled_features)


# ---------------------------------------------------------------------------
# CLI entry point (for manual checking)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = check_dependencies()

    print("=== NEXUS Dependency Check ===\n")

    if result["missing_required"]:
        print("REQUIRED (missing):")
        for pip_name, desc in result["missing_required"]:
            print(f"  [X] {pip_name} -- {desc}")
        pip_names = [p[0] for p in result["missing_required"]]
        print(f"\n  Install: {get_install_command(pip_names)}\n")
    else:
        print("REQUIRED: All installed.\n")

    if result["missing_optional"]:
        print("OPTIONAL (missing):")
        for pip_name, desc, feature in result["missing_optional"]:
            print(f"  [ ] {pip_name} -- {desc} (enables: {feature})")
        pip_names = [p[0] for p in result["missing_optional"]]
        print(f"\n  Install all: {get_install_command(pip_names)}\n")
    else:
        print("OPTIONAL: All installed.\n")

    if result["available_optional"]:
        print("AVAILABLE features:")
        for pip_name, feature in result["available_optional"]:
            print(f"  [OK] {feature} ({pip_name})")
