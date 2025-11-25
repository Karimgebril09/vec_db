import zipfile
import os
import shutil
# ----- CONSTANTS -----
INDEX_SIZE = 1
FOLDER_NAME = f"IVF_index"
ZIP_NAME = f"{FOLDER_NAME}_{INDEX_SIZE}M.zip"
DRIVE_FOLDER = "G:/My Drive/Vector_db_indexes/IVF"
# ----------------------

def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, os.path.dirname(folder_path))
                zipf.write(abs_path, rel_path)


def upload_to_google_drive_local(zip_name):
    dest = os.path.join(DRIVE_FOLDER, zip_name)

    # Delete old file if exists
    if os.path.exists(dest):
        os.remove(dest)
        print(f"Deleted old file: {dest}")

    # Copy new file
    shutil.copy(zip_name, dest)
    print(f"Uploaded new file: {dest}")


if __name__ == "__main__":
    # Ensure folder exists
    if not os.path.isdir(FOLDER_NAME):
        os.makedirs(FOLDER_NAME)

    zip_folder(FOLDER_NAME, ZIP_NAME)
    upload_to_google_drive_local(ZIP_NAME)
    print(f"Compressed folder structure only → {ZIP_NAME}")