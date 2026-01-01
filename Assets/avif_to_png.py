import subprocess
import pathlib
import sys

def convert_avif_to_png(folder="."):
    folder = pathlib.Path(folder).resolve()
    print(f"[DEBUG] Scanning folder: {folder}")

    avif_files = list(folder.rglob("*.avif"))
    print(f"[DEBUG] Found {len(avif_files)} .avif files")

    if not avif_files:
        print("[INFO] No AVIF files found. Exiting.")
        return

    for avif in avif_files:
        png = avif.with_suffix(".png")

        print(f"[DEBUG] Converting: {avif} -> {png}")

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(avif),
            str(png)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"[OK] Converted: {png}")

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to convert: {avif}")
            print("[STDOUT]", e.stdout)
            print("[STDERR]", e.stderr)

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    convert_avif_to_png(folder)