from PIL import Image
import os

# --- Configuration ---
ROOT_DIR = 'new_results/classes_split/goodware'
BEFORE_DIR = os.path.join(ROOT_DIR, 'before')
AFTER_DIR = os.path.join(ROOT_DIR, 'after')
MERGED_DIR = os.path.join(ROOT_DIR, 'merged')

def merge_images_side_by_side_random():
    """
    Lists files in the 'before' directory and merges them with 
    corresponding files in the 'after' directory.
    """
    
    # 1. Creating the output directory if it doesn't exist
    if not os.path.exists(MERGED_DIR):
        os.makedirs(MERGED_DIR)
        print(f"Created directory: {MERGED_DIR}")

    # 2. Getting the list of all PNG filenames from the 'before' directory
    before_files = [f for f in os.listdir(BEFORE_DIR) if f.endswith('.png')]
    num_files_to_process = len(before_files)
    
    print(f"Found {num_files_to_process} PNG files in '{BEFORE_DIR}'. Starting merge...")
    
    # 3. Iterate through the filenames found in 'before'
    processed_count = 0
    
    for filename in before_files:
        before_path = os.path.join(BEFORE_DIR, filename)
        after_path = os.path.join(AFTER_DIR, filename)
        merged_path = os.path.join(MERGED_DIR, filename)
        if not os.path.exists(after_path):
            print(f"Skipping {filename}: Corresponding file not found in '{AFTER_DIR}'.")
            continue

        try:
            img_before = Image.open(before_path)
            img_after = Image.open(after_path)
            
            # **Image Merging Logic**
            if img_before.height != img_after.height:
                print(f"Warning: Heights for {filename} are different ({img_before.height} vs {img_after.height}). Skipping merge for this file.")
                continue

            new_width = img_before.width + img_after.width
            new_height = img_before.height

            merged_image = Image.new('RGB', (new_width, new_height))
            
            # Pasting side-by-side
            merged_image.paste(img_before, (0, 0))
            merged_image.paste(img_after, (img_before.width, 0))
            merged_image.save(merged_path)
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} files.")

        except Exception as e:
            print(f"An unexpected error occurred while processing {filename}: {e}")

    print(f"Finished! Successfully merged {processed_count} image pairs into the **{MERGED_DIR}** folder.")

if __name__ == "__main__":
    merge_images_side_by_side_random()