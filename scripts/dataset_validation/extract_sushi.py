from pathlib import Path
import zipfile

archive = Path("data/SUSHI_tiny_1.0.zip")
destination = Path("data/SUSHI_tiny")
bad_member = "p_spike/02/0000004.npy"

destination.mkdir(parents=True, exist_ok=True)

extracted = 0
skipped = []

with zipfile.ZipFile(archive) as zip_file:
    for member in zip_file.infolist():
        if member.filename == bad_member:
            skipped.append(member.filename)
            continue

        try:
            zip_file.extract(member, destination)
            extracted += 1
        except (zipfile.BadZipFile, RuntimeError, OSError) as error:
            skipped.append(member.filename)
            print(f"Skipped {member.filename}: {error}")

print(f"Extracted members: {extracted}")
print(f"Skipped members: {skipped}")