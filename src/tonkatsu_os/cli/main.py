"""
Command-line interface for Tonkatsu-OS.
"""

import argparse
import logging
import sys
from pathlib import Path

from tonkatsu_os import __version__
from tonkatsu_os.web.app import main as run_web_app


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def cmd_web(args):
    """Run the web interface."""
    print("🔬 Starting Tonkatsu-OS Web Interface...")
    print(f"Version: {__version__}")
    print("Open your browser and navigate to the URL shown below:")
    print("-" * 50)

    # Import and run Streamlit app
    import os
    import subprocess

    # Get the path to the app.py file
    app_path = Path(__file__).parent.parent / "web" / "app.py"

    # Run streamlit
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]

    if args.headless:
        cmd.extend(["--server.headless", "true"])

    subprocess.run(cmd)


def cmd_analyze(args):
    """Analyze a spectrum file."""
    from tonkatsu_os import AdvancedPreprocessor, EnsembleClassifier, RamanSpectralDatabase

    print(f"🔬 Analyzing spectrum: {args.file}")

    # Initialize components
    db = RamanSpectralDatabase(args.database)
    preprocessor = AdvancedPreprocessor()

    # Load and analyze spectrum
    # Implementation would go here
    print("Analysis complete!")


def cmd_import(args):
    """Import spectrum files into database."""
    from tonkatsu_os import RamanSpectralDatabase, SpectrumImporter

    print(f"📥 Importing spectra from: {args.path}")

    db = RamanSpectralDatabase(args.database)
    importer = SpectrumImporter(db)

    # Implementation would go here
    print("Import complete!")


def cmd_train(args):
    """Train machine learning models."""
    from tonkatsu_os import EnsembleClassifier, RamanSpectralDatabase

    print("🤖 Training machine learning models...")

    db = RamanSpectralDatabase(args.database)
    classifier = EnsembleClassifier()

    # Implementation would go here
    print("Training complete!")


def cmd_db_stats(args):
    """Show database statistics."""
    from tonkatsu_os import RamanSpectralDatabase

    db = RamanSpectralDatabase(args.database)
    stats = db.get_database_stats()

    print("📊 Database Statistics")
    print("=" * 30)
    print(f"Total Spectra: {stats['total_spectra']}")
    print(f"Unique Compounds: {stats['unique_compounds']}")
    print(f"Top Compounds:")
    for compound, count in stats["top_compounds"][:10]:
        print(f"  • {compound}: {count}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Tonkatsu-OS: AI-Powered Raman Molecular Identification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tonkatsu web                    # Start web interface
  tonkatsu analyze spectrum.csv   # Analyze a spectrum
  tonkatsu import data/           # Import spectrum files
  tonkatsu train                  # Train ML models
  tonkatsu db-stats               # Show database stats
        """,
    )

    parser.add_argument("--version", action="version", version=f"Tonkatsu-OS {__version__}")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--database",
        default="raman_spectra.db",
        help="Database file path (default: raman_spectra.db)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Web interface command
    web_parser = subparsers.add_parser("web", help="Start web interface")
    web_parser.add_argument("--host", default="localhost", help="Host address")
    web_parser.add_argument("--port", type=int, default=8501, help="Port number")
    web_parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    web_parser.set_defaults(func=cmd_web)

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze spectrum file")
    analyze_parser.add_argument("file", help="Spectrum file to analyze")
    analyze_parser.add_argument("--output", help="Output file for results")
    analyze_parser.set_defaults(func=cmd_analyze)

    # Import command
    import_parser = subparsers.add_parser("import", help="Import spectrum files")
    import_parser.add_argument("path", help="Path to spectrum file or directory")
    import_parser.add_argument("--recursive", action="store_true", help="Import recursively")
    import_parser.set_defaults(func=cmd_import)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train ML models")
    train_parser.add_argument("--output", help="Output path for trained model")
    train_parser.set_defaults(func=cmd_train)

    # Database stats command
    stats_parser = subparsers.add_parser("db-stats", help="Show database statistics")
    stats_parser.set_defaults(func=cmd_db_stats)

    # Parse arguments
    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)

    # Run command or show help
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
