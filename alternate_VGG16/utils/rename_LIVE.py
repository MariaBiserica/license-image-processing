import os
import pandas as pd

# Paths
folder_path = '../data/LIVE2/databaserelease2/fastfading'
info_path = '../data/LIVE2/databaserelease2/fastfading/info.txt'
initial_dmos_csv = '../data/LIVE_release2/refname_dmos_scores.csv'
updated_dmos_csv = '../data/LIVE_release2/LIVE_image_scores.csvL'

print("Loading initial DMOS scores CSV...")
dmos_df = pd.read_csv(initial_dmos_csv)


def get_and_remove_first_dmos(dmos_df):
    if not dmos_df.empty:
        score = dmos_df.iloc[0]['DMOS_score']
        dmos_df = dmos_df.iloc[1:].reset_index(drop=True)
        return score, dmos_df
    else:
        print("No more DMOS scores available.")
        return None, dmos_df


def process_images_and_update_csv(folder_path, info_path, initial_dmos_csv, updated_dmos_csv):
    print(f"Processing images using info from: {info_path}")
    dmos_df = pd.read_csv(initial_dmos_csv)
    if os.path.exists(updated_dmos_csv):
        updated_df = pd.read_csv(updated_dmos_csv)
    else:
        updated_df = pd.DataFrame(columns=['new_name', 'DMOS_score'])
        print("Creating new updated_image_scores.csv file...")

    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 3:
                    original_name, distorted_name, _ = parts
                    original_name = original_name.replace('.bmp', '')
                    distorted_name = distorted_name.replace('.bmp', '')
                    new_image_name = f"fastfading_{original_name}_{distorted_name}"
                    src_img_path = os.path.join(folder_path, distorted_name + '.bmp')
                    new_img_path = os.path.join(folder_path, new_image_name + '.bmp')
                    os.rename(src_img_path, new_img_path)
                    print(f"Renamed {src_img_path} to {new_img_path}")
                    score, dmos_df = get_and_remove_first_dmos(dmos_df)
                    if score is not None:
                        new_row = {'new_name': new_image_name + '.bmp', 'DMOS_score': score}
                        updated_df = pd.concat([updated_df, pd.DataFrame([new_row])], ignore_index=True)
                        print(f"Updated CSV with: {new_image_name}.bmp, DMOS_score: {score}")

    # Save the updated CSV
    updated_df.to_csv(updated_dmos_csv, index=False)
    print(f"Saved updated scores to {updated_dmos_csv}")

    # Save the remaining DMOS scores back to the initial CSV
    dmos_df.to_csv(initial_dmos_csv, index=False)
    print(f"Saved remaining DMOS scores back to {initial_dmos_csv}")


process_images_and_update_csv(folder_path, info_path, initial_dmos_csv, updated_dmos_csv)
