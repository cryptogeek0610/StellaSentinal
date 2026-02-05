#!/usr/bin/env python3
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"


def _print_header(title: str) -> None:
    print(f"\n== {title} ==")


def _run(cmd: list[str], env: dict[str, str] | None = None) -> int:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, cwd=ROOT, env=env)
    return result.returncode


def _pythonpath_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing}" if existing else src_path
    return env


def _check_pytest() -> bool:
    return importlib.util.find_spec("pytest") is not None


def _check_backend_import() -> bool:
    if importlib.util.find_spec("device_anomaly") is not None:
        return True
    sys.path.insert(0, str(ROOT / "src"))
    return importlib.util.find_spec("device_anomaly") is not None


def _check_frontend_deps() -> tuple[bool, str]:
    if shutil.which("npm") is None:
        return False, "npm not found; install Node.js and run 'npm install' in ./frontend"
    if not FRONTEND_DIR.exists():
        return False, "frontend directory missing"
    if not (FRONTEND_DIR / "node_modules").exists():
        return False, "frontend node_modules missing; run 'npm install' in ./frontend"
    return True, ""


def verify_database_connections() -> int:
    """Test connectivity to XSight_DW and MobiControlDB databases."""
    _print_header("Database Connectivity")

    if not _check_backend_import():
        print("device_anomaly not importable. Install with 'python -m pip install -e .'")
        return 2

    # Add src to path for imports
    sys.path.insert(0, str(ROOT / "src"))

    from sqlalchemy import text

    from device_anomaly.config.settings import get_settings
    from device_anomaly.data_access.db_connection import create_dw_engine, create_mc_engine

    settings = get_settings()
    exit_code = 0

    # Test XSight_DW connection
    print("\n--- XSight_DW Connection ---")
    print(f"Host: {settings.dw.host}")
    print(f"Database: {settings.dw.database}")
    print(f"User: {settings.dw.user or '(not set)'}")
    print(f"Password: {'***' if settings.dw.password else '(not set)'}")

    try:
        engine = create_dw_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 AS test"))
            row = result.fetchone()
            if row and row[0] == 1:
                print("XSight_DW: CONNECTED OK")
            else:
                print("XSight_DW: UNEXPECTED RESPONSE")
                exit_code = 1
    except Exception as e:
        print(f"XSight_DW: FAILED - {e}")
        exit_code = 1

    # Test MobiControlDB connection
    print("\n--- MobiControlDB Connection ---")
    print(f"Host: {settings.mc.host}")
    print(f"Database: {settings.mc.database}")
    print(f"User: {settings.mc.user or '(not set)'}")
    print(f"Password: {'***' if settings.mc.password else '(not set)'}")

    if not (settings.mc.host and settings.mc.user and settings.mc.password):
        print("MobiControlDB: SKIPPED - credentials not configured")
        print("  Set MC_DB_HOST, MC_DB_USER, MC_DB_PASS environment variables")
        exit_code = 1
    else:
        try:
            engine = create_mc_engine()
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1 AS test"))
                row = result.fetchone()
                if row and row[0] == 1:
                    print("MobiControlDB: CONNECTED OK")

                    # Test if DevInfo table exists and has data
                    try:
                        result = conn.execute(text("SELECT COUNT(*) FROM dbo.DevInfo"))
                        count = result.fetchone()[0]
                        print(f"  DevInfo table: {count:,} devices found")
                    except Exception as e:
                        print(f"  DevInfo table: QUERY FAILED - {e}")
                else:
                    print("MobiControlDB: UNEXPECTED RESPONSE")
                    exit_code = 1
        except Exception as e:
            print(f"MobiControlDB: FAILED - {e}")
            exit_code = 1

    return exit_code


def verify_backend(check_only: bool) -> int:
    _print_header("Backend")
    if not _check_pytest():
        print("pytest not found. Install dependencies with 'python -m pip install -e .'")
        return 2
    if not _check_backend_import():
        print("device_anomaly not importable. Install with 'python -m pip install -e .'")
        return 2
    if check_only:
        print("Backend check OK (pytest available).")
        return 0
    return _run([sys.executable, "-m", "pytest"], env=_pythonpath_env())


def verify_frontend(check_only: bool) -> int:
    _print_header("Frontend")
    ok, msg = _check_frontend_deps()
    if not ok:
        print(msg)
        return 2
    if check_only:
        print("Frontend check OK (npm + node_modules present).")
        return 0

    rc = _run(["npm", "--prefix", str(FRONTEND_DIR), "run", "lint"])
    if rc != 0:
        return rc
    rc = _run(["npm", "--prefix", str(FRONTEND_DIR), "run", "typecheck"])
    if rc != 0:
        return rc
    rc = _run(["npm", "--prefix", str(FRONTEND_DIR), "run", "test:contracts"])
    if rc != 0:
        return rc
    return _run(["npm", "--prefix", str(FRONTEND_DIR), "run", "build"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify backend + frontend checks.")
    parser.add_argument(
        "--check", action="store_true", help="Only check prerequisites; do not run tests/builds"
    )
    parser.add_argument("--skip-backend", action="store_true", help="Skip backend verification")
    parser.add_argument("--skip-frontend", action="store_true", help="Skip frontend verification")
    parser.add_argument(
        "--db-only", action="store_true", help="Only run database connectivity tests"
    )
    args = parser.parse_args()

    # If db-only, just test database connections
    if args.db_only:
        print("verify: database connectivity")
        return verify_database_connections()

    print("verify: backend + frontend checks")

    exit_codes = []
    if not args.skip_backend:
        exit_codes.append(verify_backend(args.check))
    if not args.skip_frontend:
        exit_codes.append(verify_frontend(args.check))

    return max(exit_codes) if exit_codes else 0


if __name__ == "__main__":
    sys.exit(main())
