"""
Strict settings loader with schema validation.
Fails fast on unknown keys or missing required fields.
"""
import json
import sys
import pathlib


def load_settings(path: str = "settings.json") -> dict:
    """
    Load and validate settings against schema.
    
    Raises:
        SystemExit: If jsonschema not installed or validation fails
        FileNotFoundError: If settings.json or schema not found
    """
    schema_path = pathlib.Path(__file__).parent / "settings.schema.json"
    settings_path = pathlib.Path(path)
    
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")
    
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    
    # Check for jsonschema
    try:
        import jsonschema
    except ImportError:
        print("ERROR: jsonschema package required for settings validation.", file=sys.stderr)
        print("Install with: pip install jsonschema", file=sys.stderr)
        sys.exit(1)
    
    # Load files
    try:
        with settings_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        with schema_path.open("r", encoding="utf-8") as f:
            schema = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {settings_path}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate
    try:
        jsonschema.validate(instance=cfg, schema=schema)
    except jsonschema.ValidationError as e:
        print(f"ERROR: Settings validation failed: {e.message}", file=sys.stderr)
        if e.path:
            print(f"  Path: {'/'.join(str(p) for p in e.path)}", file=sys.stderr)
        sys.exit(1)
    except jsonschema.SchemaError as e:
        print(f"ERROR: Schema error: {e.message}", file=sys.stderr)
        sys.exit(1)
    
    return cfg

