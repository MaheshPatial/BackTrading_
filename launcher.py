import sys
from pathlib import Path


def _get_base_dir() -> Path:
    """Return the base directory where app files live.

    - In normal Python: folder containing this launcher.
    - In a PyInstaller onefile EXE: the temporary _MEIPASS folder.
    """
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller extraction folder
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def main() -> None:
    try:
        from streamlit.web import cli as stcli  # type: ignore
    except Exception as e:
        # Show the actual error so we can debug missing modules in the EXE
        print("Error while importing Streamlit:")
        print(repr(e))
        input("Press Enter to exit...")
        return

    base_dir = _get_base_dir()
    app_path = base_dir / "streamlit_app.py"

    if not app_path.exists():
        print(f"Could not find {app_path}.")
        print("Make sure streamlit_app.py is in the same folder as this launcher.")
        input("Press Enter to exit...")
        return

    # Prepare argv for the embedded Streamlit CLI
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port=8501",
        "--server.address=127.0.0.1",
    ]

    # This will start the Streamlit server and block until it stops
    stcli.main()


if __name__ == "__main__":
    main()
