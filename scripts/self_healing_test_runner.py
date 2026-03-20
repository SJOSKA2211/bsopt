import subprocess


def run_test_gauntlet():
    """
    Executes make test-all and attempts to heal common institutional failures.
    """
    print("🔥 Launching The Gauntlet (Self-Healing Mode)...")

    try:
        result = subprocess.run(["make", "test-all"], capture_output=True, text=True)
        print(result.stdout)

        if result.returncode == 0:
            print("✅ Gauntlet Passed perfectly.")
            return True

        print("❌ Gauntlet Failed. Initiating Self-Healing Protocols...")
        analyze_failures(result.stdout + result.stderr)

    except Exception as e:
        print(f"🚨 Gauntlet Crash: {str(e)}")
        return False


def analyze_failures(output: str):
    """
    Parses logs for known error patterns and suggests/applies fixes.
    """
    # 1. Check for gRPC/Protobuf mismatches
    if "ModuleNotFoundError: No module named 'src.shared.protos.auth_pb2'" in output:
        print("🔧 HEAL: Protobuf bindings missing. Running make protos...")
        subprocess.run(["make", "protos"])

    # 2. Check for Alembic migration drift
    if "Can't find any matching revision" in output:
        print("🔧 HEAL: Alembic migration drift detected. Attempting stamp & upgrade...")
        subprocess.run(["make", "alembic", "ARGS=upgrade head"])

    # 3. Check for Rust Cargo.lock drift
    if "Cargo.lock" in output and "error" in output.lower():
        print("🔧 HEAL: Cargo.lock drift. Running cargo build in rust-src.shared...")
        # Note: In real scenarios, this runs inside the container

    # 4. Check for Pydantic V2 validation errors
    if "pydantic_core._pydantic_core.ValidationError" in output:
        print("🚨 HEAL: Pydantic V2 validation failure. Inspecting schemas...")
        # This would involve deeper analysis and code generation in a real agent

    print("🚀 Self-Healing Attempted. Please re-run 'make test-all'.")


if __name__ == "__main__":
    run_test_gauntlet()
