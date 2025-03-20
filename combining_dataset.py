output_base_dir = "merged_rbc_dataset"
os.makedirs(output_base_dir, exist_ok=True)

for split in ["train", "validation", "test"]:
    split_dir = os.path.join(output_base_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    for category in categories:
        os.makedirs(os.path.join(split_dir, category), exist_ok=True)

source_dirs = [category1_dir, category2_dir, category3_dir, category4_dir]

for source_dir, category in zip(source_dirs, categories):
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.jpeg')]
    
    train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)
    
    for files, split_name in [(train_files, "train"), (test_files, "test")]:
        for file in files:
            source_path = os.path.join(source_dir, file)
            dest_path = os.path.join(output_base_dir, split_name, category, file)
            shutil.copy(source_path, dest_path)
    
