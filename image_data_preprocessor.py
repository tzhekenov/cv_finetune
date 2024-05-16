import os
import shutil
import pandas as pd

# Function to clean filenames
def clean_filename(filename):
    first_dot = filename.find(".")
    first_parenthesis = filename.find("(")
    cut_off = min([i for i in [first_dot, first_parenthesis] if i != -1])
    if cut_off == -1:
        return filename
    return filename[:cut_off]


# Function to generate the DataFrame
def generate_dataframe(directory):
    filenames = []
    cleaned_names = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                filenames.append(os.path.join(root, file))  # Full path needed here
                cleaned_name = clean_filename(file)
                cleaned_names.append(cleaned_name)

    df = pd.DataFrame({"filename": filenames, "cleaned_name": cleaned_names})
    return df


    
# Function to mark files for testing/training
def mark_test_column(df):
    # Sort the DataFrame by filename
    df = df.sort_values(by="filename")
    # Count occurrences of each cleaned name
    name_counts = df["cleaned_name"].value_counts()
    # Initialize the test column to 0
    df["test"] = 0
    # Mark the first occurrence of each name that has duplicates as 1
    df.loc[
        (~df.duplicated("cleaned_name", keep="first"))
        & (df["cleaned_name"].isin(name_counts.index[name_counts > 1])),
        "test",
    ] = 1
    # Print columns to verify "test" presence
    print("Columns in the DataFrame after marking test:", df.columns)
    return df

# Function to separate files into similar and dissimilar folders
def separate_files(df, base_directory, occurance_threshold):
    similar_dir = os.path.join(base_directory, "similar")
    dissimilar_dir = os.path.join(base_directory, "dissimilar")

    # Create target directories if they don't exist
    os.makedirs(similar_dir, exist_ok=True)
    os.makedirs(dissimilar_dir, exist_ok=True)

    # Count occurrences of each cleaned name
    name_counts = df["cleaned_name"].value_counts()

    for _, row in df.iterrows():
        src_path = row["filename"]
        object_name = row["cleaned_name"]

        if name_counts[object_name] > occurance_threshold:
            dest_dir = os.path.join(similar_dir, object_name)
        else:
            dest_dir = os.path.join(dissimilar_dir, object_name)

        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(src_path, os.path.join(dest_dir, os.path.basename(src_path)))

# Function to organize files into "test" and "train" folders
def organize_files(df, base_directory):
    test_dir = os.path.join(base_directory, "test")
    train_dir = os.path.join(base_directory, "train")

    # Create target directories if they don't exist
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    # Move files based on the 'test' column
    for _, row in df.iterrows():
        src_path = row["filename"]
        if row["test"] == 1:
            dest_dir = test_dir
        else:
            dest_dir = train_dir
        # Move the file to the appropriate directory
        shutil.move(src_path, os.path.join(dest_dir, os.path.basename(src_path)))


# Function to process images and organize them into folders
def process_and_organize_images(directory):
    df = generate_dataframe(directory)
    separate_files(df, directory, 3) # Separate files into similar and dissimilar folders
    # df = mark_test_column(df)
    # organize_files(df, directory) # Organize files into "test" and "train" folders
    return df


# Usage Example
# directory = "/app/data_test"
directory = '/workspaces/finetune/data_558'
process_and_organize_images(directory)
