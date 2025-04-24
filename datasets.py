import os
import json
from torch.utils.data import DataLoader, random_split, Subset
import torch

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

# Bỏ qua các import không cần thiết hoặc đã có sẵn trong môi trường Kaggle
# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.data import create_transform
import medmnist
from medmnist import INFO, Evaluator


import requests
from zipfile import ZipFile
import pandas as pd
import shutil

# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed)

# --- Định nghĩa các thư mục ---
# Thư mục kiểm tra (chỉ đọc) - Nơi dataset có thể đã tồn tại sẵn
INPUT_CHECK_ROOT = '/kaggle/input/data-kan/other/'
# Thư mục làm việc (có thể ghi) - Nơi tải và xử lý nếu chưa có trong input
WORKING_ROOT = '/kaggle/working/data-kan/other/'

# --- Lớp PADatasetDownloader đã chỉnh sửa ---
class PADatasetDownloader:
    def __init__(self, input_root_dir=INPUT_CHECK_ROOT, working_root_dir=WORKING_ROOT, dataset_url='https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip'):
        self.input_root_dir = input_root_dir
        self.working_root_dir = working_root_dir
        self.dataset_url = dataset_url

        # Đường dẫn để kiểm tra sự tồn tại trong input
        self.input_organized_images_dir = os.path.join(self.input_root_dir, 'PAD-Dataset')

        # Đường dẫn cho các thao tác trong working directory
        self.working_dataset_zip_path = os.path.join(self.working_root_dir, 'zr7vgbcyr2-1.zip')
        # Thư mục giải nén chính (thường là working_root_dir)
        self.working_dataset_extracted_dir = self.working_root_dir
        # Thư mục chứa các file zip con và thư mục ảnh con sau khi giải nén zip chính
        self.working_images_base_dir = os.path.join(self.working_root_dir, 'images')
        # Đường dẫn tới các thư mục chứa ảnh gốc (trong working dir)
        self.working_source_images_dirs = [os.path.join(self.working_images_base_dir, f'imgs_part_{i}') for i in range(1, 4)]
        # Đường dẫn tới thư mục ảnh đã được tổ chức (trong working dir)
        self.working_organized_images_dir = os.path.join(self.working_root_dir, 'PAD-Dataset')
        # Đường dẫn file metadata (trong working dir, sau khi giải nén zip chính)
        # Lưu ý: Đường dẫn thực tế có thể khác tuỳ thuộc cấu trúc file zip
        self.working_metadata_file_path = os.path.join(self.working_root_dir, 'metadata.csv') # Giả định nó nằm ở gốc sau khi giải nén zip chính

    def download_dataset(self):
        os.makedirs(self.working_root_dir, exist_ok=True) # Tạo thư mục working nếu chưa có
        if not os.path.exists(self.working_dataset_zip_path):
            print(f"Downloading dataset from {self.dataset_url} to {self.working_dataset_zip_path}...")
            try:
                with requests.get(self.dataset_url, stream=True) as r:
                    r.raise_for_status()
                    with open(self.working_dataset_zip_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192*16): # Tăng chunk size
                            f.write(chunk)
                print("Download complete.")
            except requests.exceptions.RequestException as e:
                print(f"Error downloading dataset: {e}")
                # Xóa file zip nếu tải lỗi để thử lại lần sau
                if os.path.exists(self.working_dataset_zip_path):
                    os.remove(self.working_dataset_zip_path)
                raise # Ném lại lỗi để dừng tiến trình nếu cần
        else:
            print(f"Dataset zip already exists at {self.working_dataset_zip_path}")

    def extract_dataset(self):
         # Kiểm tra xem thư mục 'images' (hoặc file metadata) đã tồn tại trong working_root_dir chưa
        expected_file_after_extract = self.working_metadata_file_path # Hoặc self.working_images_base_dir
        if not os.path.exists(expected_file_after_extract):
            print(f"Extracting main dataset {self.working_dataset_zip_path} to {self.working_root_dir}...")
            try:
                with ZipFile(self.working_dataset_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.working_root_dir) # Giải nén vào thư mục gốc working
                print("Main extraction complete.")
                # Kiểm tra lại sau khi giải nén
                if not os.path.exists(expected_file_after_extract):
                     print(f"Warning: Expected file/dir {expected_file_after_extract} not found after main extraction.")
                # Tìm lại file metadata nếu đường dẫn ban đầu không đúng
                if not os.path.exists(self.working_metadata_file_path):
                     found_meta = False
                     for root, dirs, files in os.walk(self.working_root_dir):
                         if 'metadata.csv' in files:
                             self.working_metadata_file_path = os.path.join(root, 'metadata.csv')
                             print(f"Found metadata file at: {self.working_metadata_file_path}")
                             found_meta = True
                             break
                     if not found_meta:
                          print("Warning: metadata.csv not found anywhere within the extracted files.")

            except (ZipFile.BadZipFile, FileNotFoundError) as e:
                 print(f"Error extracting main dataset: {e}")
                 raise
        else:
            print(f"Main dataset seems already extracted in {self.working_root_dir}")
            # Đảm bảo đường dẫn metadata đúng nếu đã giải nén từ trước
            if not os.path.exists(self.working_metadata_file_path):
                 found_meta = False
                 for root, dirs, files in os.walk(self.working_root_dir):
                     if 'metadata.csv' in files:
                         self.working_metadata_file_path = os.path.join(root, 'metadata.csv')
                         print(f"Found existing metadata file at: {self.working_metadata_file_path}")
                         found_meta = True
                         break
                 if not found_meta:
                      print("Warning: metadata.csv not found anywhere within the working directory.")


    def extract_inner_datasets(self):
        # Kiểm tra sự tồn tại của thư mục images base trước khi giải nén zip con
        if not os.path.exists(self.working_images_base_dir):
            print(f"Base images directory {self.working_images_base_dir} not found. Cannot extract inner zips.")
            return

        for i, source_images_dir in enumerate(self.working_source_images_dirs, start=1):
            # Đường dẫn zip con nằm trong thư mục images base
            inner_zip_path = os.path.join(self.working_images_base_dir, f'imgs_part_{i}.zip')
            # Chỉ giải nén nếu zip con tồn tại VÀ thư mục đích chưa tồn tại
            if os.path.exists(inner_zip_path) and not os.path.exists(source_images_dir):
                print(f"Extracting {inner_zip_path}...")
                try:
                    with ZipFile(inner_zip_path, 'r') as zip_ref:
                        # Giải nén vào thư mục cha của zip con (working_images_base_dir)
                        zip_ref.extractall(self.working_images_base_dir)
                    print(f"Extraction of {inner_zip_path} complete.")
                except (ZipFile.BadZipFile, FileNotFoundError) as e:
                    print(f"Error extracting inner zip {inner_zip_path}: {e}")
            elif not os.path.exists(inner_zip_path):
                print(f"Inner zip {inner_zip_path} not found. Skipping extraction.")
            else:
                print(f"Inner dataset directory {source_images_dir} already exists. Skipping extraction.")

    def organize_images(self):
        # Kiểm tra nếu đã tổ chức trong working dir
        if os.path.exists(self.working_organized_images_dir):
            print(f"Images are already organized in the working directory: {self.working_organized_images_dir}")
            return

        # Kiểm tra file metadata trong working dir
        if not os.path.exists(self.working_metadata_file_path):
            print(f"Metadata file not found at {self.working_metadata_file_path}. Cannot organize images.")
            # Thử tìm lại lần nữa phòng trường hợp đường dẫn bị sai lệch
            found_meta = False
            for root, dirs, files in os.walk(self.working_root_dir):
                 if 'metadata.csv' in files:
                     self.working_metadata_file_path = os.path.join(root, 'metadata.csv')
                     print(f"Found metadata file at: {self.working_metadata_file_path}")
                     found_meta = True
                     break
            if not found_meta:
                 raise FileNotFoundError(f"Metadata file not found at {self.working_metadata_file_path} or elsewhere. Cannot organize images.")


        try:
            metadata = pd.read_csv(self.working_metadata_file_path)
        except Exception as e:
            print(f"Error reading metadata CSV {self.working_metadata_file_path}: {e}")
            raise

        os.makedirs(self.working_organized_images_dir, exist_ok=True) # Tạo thư mục đích trong working

        diagnostic_labels = metadata['diagnostic'].unique()
        for label in diagnostic_labels:
            os.makedirs(os.path.join(self.working_organized_images_dir, str(label)), exist_ok=True) # Tạo thư mục con trong working

        print("Organizing images into subfolders...")
        moved_count = 0
        not_found_count = 0
        missing_sources = []

        # Đảm bảo các thư mục nguồn tồn tại trước khi duyệt
        existing_source_dirs = [d for d in self.working_source_images_dirs if os.path.isdir(d)]
        if not existing_source_dirs:
             print("Warning: None of the source image directories exist. Cannot move images.")
             print(f"Expected source directories: {self.working_source_images_dirs}")
             return # Hoặc raise lỗi tùy theo yêu cầu

        for _, row in metadata.iterrows():
            img_id = row['img_id']
            diagnostic = str(row['diagnostic']) # Đảm bảo label là string
            found_source = False
            for source_dir in existing_source_dirs: # Chỉ duyệt qua các thư mục nguồn thực sự tồn tại
                source_path = os.path.join(source_dir, img_id)
                if os.path.exists(source_path):
                    destination_path = os.path.join(self.working_organized_images_dir, diagnostic, img_id)
                    try:
                        shutil.move(source_path, destination_path)
                        moved_count += 1
                        found_source = True
                        break # Thoát vòng lặp source_dir khi đã tìm thấy và di chuyển
                    except OSError as e:
                        print(f"Error moving {source_path} to {destination_path}: {e}")
                        # Có thể file đích đã tồn tại hoặc lỗi quyền (ít khả năng trong /kaggle/working)
                        not_found_count += 1 # Coi như không tìm thấy nếu không di chuyển được
                        break

            if not found_source:
                not_found_count += 1
                missing_sources.append(img_id)
                # print(f"Warning: Source image {img_id} not found in any source directory.") # Bật nếu cần debug

        print(f"Images moved: {moved_count}. Images listed in metadata but not found in source folders: {not_found_count}.")
        if not_found_count > 0 and moved_count == 0:
            print("Warning: No images were moved. Check if inner datasets were extracted correctly and if metadata matches filenames.")
            print(f"First 10 missing image IDs: {missing_sources[:10]}")
        print("Image organization complete.")

    def get_dataset(self):
        # 1. Kiểm tra trong INPUT_CHECK_ROOT
        if os.path.exists(self.input_organized_images_dir) and os.path.isdir(self.input_organized_images_dir):
             # Kiểm tra xem có thư mục con không (đảm bảo không phải thư mục rỗng)
            if any(os.path.isdir(os.path.join(self.input_organized_images_dir, i)) for i in os.listdir(self.input_organized_images_dir)):
                print(f"Dataset found in INPUT directory: {self.input_organized_images_dir}")
                return self.input_organized_images_dir
            else:
                 print(f"Found directory {self.input_organized_images_dir} in INPUT, but it seems empty. Proceeding to working directory.")

        # 2. Kiểm tra trong WORKING_ROOT
        if os.path.exists(self.working_organized_images_dir) and os.path.isdir(self.working_organized_images_dir):
             # Kiểm tra xem có thư mục con không
            if any(os.path.isdir(os.path.join(self.working_organized_images_dir, i)) for i in os.listdir(self.working_organized_images_dir)):
                print(f"Dataset found in WORKING directory: {self.working_organized_images_dir}")
                return self.working_organized_images_dir
            else:
                 print(f"Found directory {self.working_organized_images_dir} in WORKING, but it seems empty. Attempting processing.")


        # 3. Nếu không có, tải và xử lý vào WORKING_ROOT
        print("Dataset not found pre-organized in input or working directory. Starting download and processing...")
        try:
            self.download_dataset()
            self.extract_dataset()
            self.extract_inner_datasets()
            self.organize_images()
        except Exception as e:
             print(f"An error occurred during dataset preparation: {e}")
             print("Please check the logs for details. Cannot return dataset path.")
             raise # Ném lại lỗi để dừng hẳn

        # 4. Kiểm tra lại sau khi xử lý và trả về đường dẫn working
        if os.path.exists(self.working_organized_images_dir) and os.path.isdir(self.working_organized_images_dir):
             if any(os.path.isdir(os.path.join(self.working_organized_images_dir, i)) for i in os.listdir(self.working_organized_images_dir)):
                 print(f"Dataset processed and available at: {self.working_organized_images_dir}")
                 return self.working_organized_images_dir
             else:
                 raise RuntimeError(f"Dataset processing finished, but the target directory {self.working_organized_images_dir} is empty or invalid.")
        else:
            raise RuntimeError("Dataset processing failed. Organized directory not found in working directory after processing.")

# --- Các lớp Downloader khác cần được chỉnh sửa tương tự ---
# (FetalDatasetDownloader, ISICDatasetManager, CPNDatasetDownloader, KvasirDatasetDownloader)
# Dưới đây là ví dụ cho FetalDatasetDownloader:

class FetalDatasetDownloader:
    def __init__(self, input_root_dir=INPUT_CHECK_ROOT, working_root_dir=WORKING_ROOT, dataset_url='https://zenodo.org/records/3904280/files/FETAL_PLANES_ZENODO.zip'):
        self.input_root_dir = input_root_dir
        self.working_root_dir = working_root_dir
        self.dataset_url = dataset_url

        # Input check path
        self.input_organized_images_dir = os.path.join(self.input_root_dir, 'Fetal-Dataset')

        # Working paths
        self.working_dataset_zip_path = os.path.join(self.working_root_dir, 'FETAL_PLANES_ZENODO.zip')
        self.working_dataset_extracted_dir = self.working_root_dir # Giải nén trực tiếp vào working root
        self.working_organized_images_dir = os.path.join(self.working_root_dir, 'Fetal-Dataset')
        # Đường dẫn file excel và thư mục ảnh gốc sau khi giải nén vào working dir
        self.working_excel_file_path = os.path.join(self.working_dataset_extracted_dir, 'FETAL_PLANES_DB_data.xlsx')
        self.working_source_images_dir = os.path.join(self.working_dataset_extracted_dir, 'Images')

    def download_dataset(self):
        os.makedirs(self.working_root_dir, exist_ok=True)
        if not os.path.exists(self.working_dataset_zip_path):
            print(f"Downloading dataset from {self.dataset_url} to {self.working_dataset_zip_path}...")
            # ... (Giống PADatasetDownloader) ...
            try:
                with requests.get(self.dataset_url, stream=True) as r:
                    r.raise_for_status()
                    with open(self.working_dataset_zip_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192*16):
                            f.write(chunk)
                print("Download complete.")
            except requests.exceptions.RequestException as e:
                print(f"Error downloading dataset: {e}")
                if os.path.exists(self.working_dataset_zip_path): os.remove(self.working_dataset_zip_path)
                raise
        else:
            print(f"Dataset zip already exists at {self.working_dataset_zip_path}")

    def extract_dataset(self):
        # Kiểm tra sự tồn tại của file excel hoặc thư mục Images trong working dir
        if not os.path.exists(self.working_excel_file_path) or not os.path.exists(self.working_source_images_dir):
            print(f"Extracting dataset {self.working_dataset_zip_path} to {self.working_dataset_extracted_dir}...")
            try:
                with ZipFile(self.working_dataset_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.working_dataset_extracted_dir)
                print("Extraction complete.")
                if not os.path.exists(self.working_excel_file_path) or not os.path.exists(self.working_source_images_dir):
                     print("Warning: Expected files/dirs not found after extraction.")
            except (ZipFile.BadZipFile, FileNotFoundError) as e:
                 print(f"Error extracting dataset: {e}")
                 raise
        else:
            print(f"Dataset seems already extracted in {self.working_dataset_extracted_dir}")

    def organize_images(self):
        if os.path.exists(self.working_organized_images_dir):
            print(f"Images are already organized in the working directory: {self.working_organized_images_dir}")
            return

        if not os.path.exists(self.working_excel_file_path):
             print(f"Excel file not found at {self.working_excel_file_path}. Cannot organize images.")
             raise FileNotFoundError(f"Excel file not found at {self.working_excel_file_path}")

        if not os.path.exists(self.working_source_images_dir):
             print(f"Source images directory not found at {self.working_source_images_dir}. Cannot organize images.")
             raise FileNotFoundError(f"Source images directory not found at {self.working_source_images_dir}")


        try:
            df = pd.read_excel(self.working_excel_file_path)
        except Exception as e:
             print(f"Error reading Excel file {self.working_excel_file_path}: {e}")
             raise

        os.makedirs(self.working_organized_images_dir, exist_ok=True)

        plane_labels = df['Plane'].unique()
        for label in plane_labels:
            os.makedirs(os.path.join(self.working_organized_images_dir, str(label)), exist_ok=True)

        print("Organizing images...")
        moved_count = 0
        not_found_count = 0
        for _, row in df.iterrows():
            img_id = row['Image_name']
            plane = str(row['Plane'])
            # Đường dẫn file ảnh gốc trong working dir (thường có đuôi .png)
            source_path = os.path.join(self.working_source_images_dir, f'{img_id}.png')
            destination_path = os.path.join(self.working_organized_images_dir, plane, f'{img_id}.png')

            if os.path.exists(source_path):
                try:
                    shutil.move(source_path, destination_path)
                    moved_count += 1
                except OSError as e:
                    print(f"Error moving {source_path} to {destination_path}: {e}")
                    not_found_count += 1
            else:
                # Có thể file có đuôi khác? hoặc tên file trong excel không khớp?
                # print(f"Warning: Source image {source_path} not found.")
                not_found_count += 1

        print(f"Images moved: {moved_count}. Images listed in Excel but not found: {not_found_count}.")
        print("Images moved successfully.") # Giữ lại log gốc

    def get_dataset(self):
        # 1. Check input
        if os.path.exists(self.input_organized_images_dir) and os.path.isdir(self.input_organized_images_dir):
            if any(os.path.isdir(os.path.join(self.input_organized_images_dir, i)) for i in os.listdir(self.input_organized_images_dir)):
                print(f"Dataset found in INPUT directory: {self.input_organized_images_dir}")
                return self.input_organized_images_dir
            else:
                print(f"Found directory {self.input_organized_images_dir} in INPUT, but it seems empty.")

        # 2. Check working
        if os.path.exists(self.working_organized_images_dir) and os.path.isdir(self.working_organized_images_dir):
            if any(os.path.isdir(os.path.join(self.working_organized_images_dir, i)) for i in os.listdir(self.working_organized_images_dir)):
                print(f"Dataset found in WORKING directory: {self.working_organized_images_dir}")
                return self.working_organized_images_dir
            else:
                 print(f"Found directory {self.working_organized_images_dir} in WORKING, but it seems empty.")


        # 3. Process to working
        print("Dataset not found pre-organized. Starting download and processing...")
        try:
            self.download_dataset()
            self.extract_dataset()
            self.organize_images()
        except Exception as e:
            print(f"An error occurred during dataset preparation: {e}")
            raise

        # 4. Return working path
        if os.path.exists(self.working_organized_images_dir) and os.path.isdir(self.working_organized_images_dir):
             if any(os.path.isdir(os.path.join(self.working_organized_images_dir, i)) for i in os.listdir(self.working_organized_images_dir)):
                 print(f"Dataset processed and available at: {self.working_organized_images_dir}")
                 return self.working_organized_images_dir
             else:
                  raise RuntimeError(f"Dataset processing finished, but the target directory {self.working_organized_images_dir} is empty or invalid.")
        else:
             raise RuntimeError("Dataset processing failed. Organized directory not found in working directory after processing.")


# --- Cần chỉnh sửa tương tự cho ISICDatasetManager, CPNDatasetDownloader, KvasirDatasetDownloader ---
# Ví dụ cho ISICDatasetManager (khái niệm tương tự):
class ISICDatasetManager:
    def __init__(self, input_base_dir=INPUT_CHECK_ROOT, working_base_dir=WORKING_ROOT):
        self.input_base_dir = input_base_dir
        self.working_base_dir = working_base_dir

        # URLs
        self.train_url = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip'
        self.test_url = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_Input.zip'
        self.train_gt_url = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip'
        self.test_gt_url = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_GroundTruth.zip'

        # Input check paths (thư mục cuối cùng sau khi tổ chức)
        self.input_train_categorized = os.path.join(self.input_base_dir, 'ISIC2018_Train', 'Categorized')
        self.input_test_categorized = os.path.join(self.input_base_dir, 'ISIC2018_Test', 'Categorized')

        # Working paths
        self.working_train_path = os.path.join(self.working_base_dir, 'ISIC2018_Train')
        self.working_test_path = os.path.join(self.working_base_dir, 'ISIC2018_Test')
        self.working_train_categorized = os.path.join(self.working_train_path, 'Categorized')
        self.working_test_categorized = os.path.join(self.working_test_path, 'Categorized')

        # Paths for temporary downloads (within working dir)
        self.working_temp_downloads = self.working_base_dir # Tải zip vào thư mục gốc working

        # Ensure working base directory exists
        os.makedirs(self.working_base_dir, exist_ok=True)
        os.makedirs(self.working_train_path, exist_ok=True)
        os.makedirs(self.working_test_path, exist_ok=True)


    def download_and_extract(self, url, extract_to):
        # Tải vào working_temp_downloads
        local_filename = os.path.join(self.working_temp_downloads, url.split('/')[-1])
        if not os.path.exists(local_filename):
            print(f"Downloading {url} to {local_filename}...")
            try:
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(local_filename, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192*16):
                            f.write(chunk)
                print("Download complete.")
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {url}: {e}")
                if os.path.exists(local_filename): os.remove(local_filename)
                raise
        else:
            print(f"Zip file {local_filename} already exists.")

        # Giải nén vào thư mục extract_to (trong working dir)
        # Kiểm tra xem thư mục đích đã chứa nội dung chưa (heuristic)
        # Ví dụ: kiểm tra sự tồn tại của file metadata hoặc một file ảnh bất kỳ nếu biết trước
        # Hoặc đơn giản là kiểm tra thư mục đích không rỗng
        should_extract = True
        if os.path.isdir(extract_to) and len(os.listdir(extract_to)) > 0:
             # Có vẻ đã giải nén, có thể kiểm tra kỹ hơn nếu cần
              print(f"Directory {extract_to} already exists and is not empty. Assuming extracted.")
              should_extract = False # Bỏ qua giải nén

        if should_extract:
            print(f"Extracting {local_filename} to {extract_to}...")
            try:
                with ZipFile(local_filename, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                print("Extraction complete.")
            except (ZipFile.BadZipFile, FileNotFoundError) as e:
                 print(f"Error extracting {local_filename}: {e}")
                 # Không nên xóa zip ở đây vì có thể dùng lại
                 raise
        # Không xóa file zip tải về vì có thể cần giải nén lại hoặc dùng cho test set

    def organize_by_labels(self, metadata_path, image_dir, output_base_dir):
         # Kiểm tra nếu đã tổ chức
        if os.path.exists(output_base_dir) and any(os.path.isdir(os.path.join(output_base_dir, i)) for i in os.listdir(output_base_dir)):
            print(f"Images seem already organized in {output_base_dir}. Skipping organization.")
            return

        if not os.path.exists(metadata_path):
            print(f"Metadata file not found: {metadata_path}. Cannot organize.")
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        if not os.path.exists(image_dir):
            print(f"Image directory not found: {image_dir}. Cannot organize.")
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        try:
            metadata = pd.read_csv(metadata_path)
        except Exception as e:
             print(f"Error reading metadata CSV {metadata_path}: {e}")
             raise

        labels = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        os.makedirs(output_base_dir, exist_ok=True) # Đảm bảo thư mục đích tồn tại
        for label in labels:
            os.makedirs(os.path.join(output_base_dir, label), exist_ok=True)

        print(f"Organizing images from {image_dir} based on {metadata_path} into {output_base_dir}...")
        moved_count = 0
        not_found_count = 0
        error_count = 0

        # Hiệu quả hơn là listdir một lần
        available_images = set(os.listdir(image_dir))

        def move_image(row):
            nonlocal moved_count, not_found_count, error_count
            image_name_base = row['image']
            # Thử cả .jpg và .jpeg vì dataset có thể không nhất quán
            possible_filenames = [f"{image_name_base}.jpg", f"{image_name_base}.jpeg"]
            source_filename = None

            for fname in possible_filenames:
                 if fname in available_images:
                     source_filename = fname
                     break

            if source_filename is None:
                 #print(f"Warning: Image {image_name_base} (jpg/jpeg) not found in {image_dir}")
                 not_found_count += 1
                 return

            source_path = os.path.join(image_dir, source_filename)
            target_label = None
            for label in labels:
                # Kiểm tra xem cột có tồn tại và giá trị là 1.0 hay không
                if label in row and row[label] == 1.0:
                    target_label = label
                    break

            if target_label:
                target_path = os.path.join(output_base_dir, target_label, source_filename)
                try:
                    shutil.move(source_path, target_path)
                    moved_count += 1
                    available_images.remove(source_filename) # Xóa khỏi set để không tìm lại
                except OSError as e:
                    print(f"Error moving {source_path} to {target_path}: {e}")
                    error_count += 1
            else:
                #print(f"Warning: No label found for image {image_name_base}")
                not_found_count +=1 # Hoặc coi là lỗi?

        # Sử dụng apply có thể chậm với dataset lớn, vòng lặp for có thể nhanh hơn
        # metadata.apply(move_image, axis=1)
        for index, row in metadata.iterrows():
             move_image(row)

        print(f"Organization complete for {output_base_dir}. Moved: {moved_count}, Not found/No label: {not_found_count}, Errors: {error_count}")
        # Dọn dẹp thư mục ảnh gốc nếu nó không còn chứa gì (và khác thư mục giải nén gốc)
        if moved_count > 0 and not error_count and not not_found_count:
             try:
                 # Cẩn thận khi xóa, đảm bảo không xóa nhầm
                 # Ví dụ: chỉ xóa nếu image_dir là thư mục con cụ thể như 'ISIC2018_Task3_Training_Input'
                 # if os.path.basename(image_dir) in ['ISIC2018_Task3_Training_Input', 'ISIC2018_Task3_Test_Input'] and not os.listdir(image_dir):
                 #     print(f"Removing empty source directory: {image_dir}")
                 #     shutil.rmtree(image_dir)
                 pass # Tạm thời không xóa tự động
             except OSError as e:
                 print(f"Could not remove source directory {image_dir}: {e}")


    def setup_dataset(self):
        # 1. Check input categorized paths
        train_ready = os.path.exists(self.input_train_categorized) and os.path.isdir(self.input_train_categorized) and any(os.path.isdir(os.path.join(self.input_train_categorized, i)) for i in os.listdir(self.input_train_categorized))
        test_ready = os.path.exists(self.input_test_categorized) and os.path.isdir(self.input_test_categorized) and any(os.path.isdir(os.path.join(self.input_test_categorized, i)) for i in os.listdir(self.input_test_categorized))

        if train_ready and test_ready:
             print(f"Found organized datasets in INPUT: {self.input_train_categorized} and {self.input_test_categorized}")
             return self.input_train_categorized, self.input_test_categorized

        # 2. Check working categorized paths
        train_ready_work = os.path.exists(self.working_train_categorized) and os.path.isdir(self.working_train_categorized) and any(os.path.isdir(os.path.join(self.working_train_categorized, i)) for i in os.listdir(self.working_train_categorized))
        test_ready_work = os.path.exists(self.working_test_categorized) and os.path.isdir(self.working_test_categorized) and any(os.path.isdir(os.path.join(self.working_test_categorized, i)) for i in os.listdir(self.working_test_categorized))

        if train_ready_work and test_ready_work:
            print(f"Found organized datasets in WORKING: {self.working_train_categorized} and {self.working_test_categorized}")
            return self.working_train_categorized, self.working_test_categorized

        # 3. Process to working directory
        print("Organized ISIC datasets not found. Starting download, extraction, and organization...")

        try:
            # Download and extract all necessary parts into their respective working directories
            print("\n--- Processing Training Data ---")
            self.download_and_extract(self.train_url, self.working_train_path) # Extract input to working_train_path
            self.download_and_extract(self.train_gt_url, self.working_train_path) # Extract GT to working_train_path

            print("\n--- Processing Test Data ---")
            self.download_and_extract(self.test_url, self.working_test_path)   # Extract input to working_test_path
            self.download_and_extract(self.test_gt_url, self.working_test_path)  # Extract GT to working_test_path

            print("\n--- Organizing Training Images ---")
            # Đường dẫn sau khi giải nén trong working dir
            train_gt_csv = os.path.join(self.working_train_path, 'ISIC2018_Task3_Training_GroundTruth', 'ISIC2018_Task3_Training_GroundTruth.csv')
            train_img_dir = os.path.join(self.working_train_path, 'ISIC2018_Task3_Training_Input')
            self.organize_by_labels(train_gt_csv, train_img_dir, self.working_train_categorized)

            print("\n--- Organizing Test Images ---")
             # Đường dẫn sau khi giải nén trong working dir
            test_gt_csv = os.path.join(self.working_test_path, 'ISIC2018_Task3_Test_GroundTruth', 'ISIC2018_Task3_Test_GroundTruth.csv')
            test_img_dir = os.path.join(self.working_test_path, 'ISIC2018_Task3_Test_Input')
            self.organize_by_labels(test_gt_csv, test_img_dir, self.working_test_categorized)

        except Exception as e:
             print(f"An error occurred during ISIC dataset setup: {e}")
             raise

        # 4. Return working paths after successful processing
        train_ok = os.path.exists(self.working_train_categorized) and os.path.isdir(self.working_train_categorized) and any(os.path.isdir(os.path.join(self.working_train_categorized, i)) for i in os.listdir(self.working_train_categorized))
        test_ok = os.path.exists(self.working_test_categorized) and os.path.isdir(self.working_test_categorized) and any(os.path.isdir(os.path.join(self.working_test_categorized, i)) for i in os.listdir(self.working_test_categorized))

        if train_ok and test_ok:
             print(f"ISIC datasets processed and available at: {self.working_train_categorized} and {self.working_test_categorized}")
             return self.working_train_categorized, self.working_test_categorized
        else:
             raise RuntimeError("ISIC dataset processing failed. Categorized directories not found or are empty in working directory.")


# --- Tương tự cho CPN và Kvasir ---
class CPNDatasetDownloader:
    def __init__(self, input_root_dir=INPUT_CHECK_ROOT, working_root_dir=WORKING_ROOT, dataset_url='https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/dvntn9yhd2-1.zip'):
        self.input_root_dir = input_root_dir
        self.working_root_dir = working_root_dir
        self.dataset_url = dataset_url

        # Input check path (thư mục cuối cùng chứa các class con)
        self.input_organized_images_dir = os.path.join(self.input_root_dir, 'CPN-Dataset', 'Covid19-Pneumonia-Normal Chest X-Ray Images') # Đường dẫn có thể cần điều chỉnh

        # Working paths
        self.working_dataset_zip_path = os.path.join(self.working_root_dir, 'dvntn9yhd2-1.zip')
        # Thư mục chứa zip chính và có thể là thư mục giải nén zip chính
        self.working_outer_extract_dir = os.path.join(self.working_root_dir, 'dvntn9yhd2-1')
        # Zip con bên trong
        self.working_inner_zip_path = os.path.join(self.working_outer_extract_dir, 'Covid19-Pneumonia-Normal Chest X-Ray Images Dataset.zip')
        # Thư mục đích cuối cùng cho ảnh đã tổ chức (nơi giải nén zip con)
        self.working_organized_images_dir_base = os.path.join(self.working_root_dir, 'CPN-Dataset') # Thư mục gốc cho dataset này
        # Thư mục chứa ảnh thực tế sau khi giải nén zip con
        # Tên thư mục này phụ thuộc vào cấu trúc bên trong inner zip
        self.working_organized_images_dir = os.path.join(self.working_organized_images_dir_base, 'Covid19-Pneumonia-Normal Chest X-Ray Images') # Giả định tên thư mục

    def download_dataset(self):
        os.makedirs(self.working_root_dir, exist_ok=True)
        if not os.path.exists(self.working_dataset_zip_path):
            print(f"Downloading dataset from {self.dataset_url} to {self.working_dataset_zip_path}...")
            # ... (Download logic) ...
            try:
                with requests.get(self.dataset_url, stream=True) as r:
                    r.raise_for_status()
                    with open(self.working_dataset_zip_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192*16): f.write(chunk)
                print("Download complete.")
            except requests.exceptions.RequestException as e:
                print(f"Error downloading: {e}"); os.remove(self.working_dataset_zip_path); raise
        else:
            print(f"Dataset zip already exists at {self.working_dataset_zip_path}")

    def extract_dataset(self):
        # Giải nén zip chính vào working_root_dir để lộ ra thư mục dvntn9yhd2-1 và zip con bên trong
        if not os.path.exists(self.working_outer_extract_dir):
            print(f"Extracting main dataset {self.working_dataset_zip_path} to {self.working_root_dir}...")
            try:
                with ZipFile(self.working_dataset_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.working_root_dir)
                print("Main extraction complete.")
                if not os.path.exists(self.working_inner_zip_path):
                     print(f"Warning: Inner zip {self.working_inner_zip_path} not found after main extraction.")
            except (ZipFile.BadZipFile, FileNotFoundError) as e:
                 print(f"Error extracting main dataset: {e}"); raise
        else:
            print(f"Main dataset seems already extracted (found {self.working_outer_extract_dir}).")

    def extract_inner_dataset(self):
        # Giải nén zip con vào thư mục đích cuối cùng (working_organized_images_dir_base)
         # Chỉ giải nén nếu zip con tồn tại và thư mục đích chưa tồn tại (hoặc rỗng)
        inner_zip_exists = os.path.exists(self.working_inner_zip_path)
        target_dir_exists = os.path.exists(self.working_organized_images_dir)
        target_dir_not_empty = target_dir_exists and any(os.path.isdir(os.path.join(self.working_organized_images_dir, i)) for i in os.listdir(self.working_organized_images_dir))

        if inner_zip_exists and not target_dir_not_empty:
             os.makedirs(self.working_organized_images_dir_base, exist_ok=True) # Đảm bảo thư mục cha tồn tại
             print(f"Extracting inner dataset {self.working_inner_zip_path} to {self.working_organized_images_dir_base}...")
             try:
                 with ZipFile(self.working_inner_zip_path, 'r') as zip_ref:
                     zip_ref.extractall(self.working_organized_images_dir_base) # Giải nén vào thư mục gốc CPN
                 print("Inner extraction complete.")
                 if not os.path.exists(self.working_organized_images_dir):
                      print(f"Warning: Expected final image directory {self.working_organized_images_dir} not found after inner extraction.")
             except (ZipFile.BadZipFile, FileNotFoundError) as e:
                 print(f"Error extracting inner dataset: {e}"); raise
        elif not inner_zip_exists:
            print(f"Inner zip {self.working_inner_zip_path} not found. Cannot extract inner dataset.")
        elif target_dir_not_empty:
            print(f"Inner dataset seems already extracted and organized in {self.working_organized_images_dir}.")


    def get_dataset(self):
         # 1. Check input
        input_path_to_check = self.input_organized_images_dir # Điều chỉnh nếu cần
        if os.path.exists(input_path_to_check) and os.path.isdir(input_path_to_check):
            if any(os.path.isdir(os.path.join(input_path_to_check, i)) for i in os.listdir(input_path_to_check)):
                print(f"Dataset found in INPUT directory: {input_path_to_check}")
                return input_path_to_check

        # 2. Check working
        working_path_to_check = self.working_organized_images_dir # Thư mục cuối cùng chứa class
        if os.path.exists(working_path_to_check) and os.path.isdir(working_path_to_check):
            if any(os.path.isdir(os.path.join(working_path_to_check, i)) for i in os.listdir(working_path_to_check)):
                print(f"Dataset found in WORKING directory: {working_path_to_check}")
                return working_path_to_check

        # 3. Process to working
        print("CPN Dataset not found pre-organized. Starting download and processing...")
        try:
            self.download_dataset()
            self.extract_dataset()
            self.extract_inner_dataset()
        except Exception as e:
             print(f"An error occurred during CPN dataset preparation: {e}"); raise

        # 4. Return working path
        if os.path.exists(working_path_to_check) and os.path.isdir(working_path_to_check):
             if any(os.path.isdir(os.path.join(working_path_to_check, i)) for i in os.listdir(working_path_to_check)):
                 print(f"Dataset processed and available at: {working_path_to_check}")
                 return working_path_to_check
             else:
                  raise RuntimeError(f"CPN Dataset processing finished, but the target directory {working_path_to_check} is empty or invalid.")
        else:
             raise RuntimeError("CPN Dataset processing failed. Organized directory not found in working directory after processing.")


class KvasirDatasetDownloader:
    def __init__(self, input_root_dir=INPUT_CHECK_ROOT, working_root_dir=WORKING_ROOT, dataset_url='https://datasets.simula.no/downloads/kvasir/kvasir-dataset.zip'):
        self.input_root_dir = input_root_dir
        self.working_root_dir = working_root_dir
        self.dataset_url = dataset_url

        # Input check path
        self.input_dataset_dir = os.path.join(self.input_root_dir, 'kvasir-dataset')

        # Working paths
        self.working_dataset_zip_path = os.path.join(self.working_root_dir, 'kvasir-dataset.zip')
        # Thư mục giải nén cuối cùng trong working dir
        self.working_dataset_dir = os.path.join(self.working_root_dir, 'kvasir-dataset')

    def download_dataset(self):
        os.makedirs(self.working_root_dir, exist_ok=True)
        if not os.path.exists(self.working_dataset_zip_path):
            print(f"Downloading dataset from {self.dataset_url} to {self.working_dataset_zip_path}...")
            # ... (Download logic) ...
            try:
                with requests.get(self.dataset_url, stream=True) as r:
                    r.raise_for_status()
                    with open(self.working_dataset_zip_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192*16): f.write(chunk)
                print("Download complete.")
            except requests.exceptions.RequestException as e:
                print(f"Error downloading: {e}"); os.remove(self.working_dataset_zip_path); raise
        else:
            print(f"Dataset zip already exists at {self.working_dataset_zip_path}")

    def extract_dataset(self):
        # Giải nén vào working_root_dir, nó sẽ tạo ra thư mục kvasir-dataset
        if not os.path.exists(self.working_dataset_dir):
            print(f"Extracting dataset {self.working_dataset_zip_path} to {self.working_root_dir}...")
            try:
                with ZipFile(self.working_dataset_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.working_root_dir)
                print("Extraction complete.")
                if not os.path.exists(self.working_dataset_dir):
                     print(f"Warning: Expected directory {self.working_dataset_dir} not found after extraction.")
            except (ZipFile.BadZipFile, FileNotFoundError) as e:
                 print(f"Error extracting dataset: {e}"); raise
        else:
            print(f"Dataset directory {self.working_dataset_dir} already exists.")

    def get_dataset(self):
        # 1. Check input
        if os.path.exists(self.input_dataset_dir) and os.path.isdir(self.input_dataset_dir):
            if any(os.path.isdir(os.path.join(self.input_dataset_dir, i)) for i in os.listdir(self.input_dataset_dir)):
                print(f"Dataset found in INPUT directory: {self.input_dataset_dir}")
                return self.input_dataset_dir

        # 2. Check working
        if os.path.exists(self.working_dataset_dir) and os.path.isdir(self.working_dataset_dir):
            if any(os.path.isdir(os.path.join(self.working_dataset_dir, i)) for i in os.listdir(self.working_dataset_dir)):
                print(f"Dataset found in WORKING directory: {self.working_dataset_dir}")
                return self.working_dataset_dir

        # 3. Process to working
        print("Kvasir Dataset not found pre-organized. Starting download and processing...")
        try:
            self.download_dataset()
            self.extract_dataset()
        except Exception as e:
             print(f"An error occurred during Kvasir dataset preparation: {e}"); raise

        # 4. Return working path
        if os.path.exists(self.working_dataset_dir) and os.path.isdir(self.working_dataset_dir):
            if any(os.path.isdir(os.path.join(self.working_dataset_dir, i)) for i in os.listdir(self.working_dataset_dir)):
                 print(f"Dataset processed and available at: {self.working_dataset_dir}")
                 return self.working_dataset_dir
            else:
                raise RuntimeError(f"Kvasir Dataset processing finished, but the target directory {self.working_dataset_dir} is empty or invalid.")
        else:
             raise RuntimeError("Kvasir Dataset processing failed. Organized directory not found in working directory after processing.")



# --- Hàm build_transform (giữ nguyên hoặc điều chỉnh nếu cần) ---
def build_transform(args):
    t_train = []
    t_train.append(transforms.RandomResizedCrop(args.input_size)) # Sử dụng args.input_size
    # t_train.append(transforms.AugMix(alpha= 0.4)) # AugMix có thể cần cài đặt riêng
    t_train.append(transforms.RandomHorizontalFlip(p=0.5)) # P=0.5 thường phổ biến hơn
    t_train.append(transforms.ToTensor())
    t_train.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])) # Normalize 3 kênh nếu ảnh RGB

    t_test = []
    t_test.append(transforms.Resize((args.input_size, args.input_size))) # Sử dụng args.input_size
    t_test.append(transforms.ToTensor())
    t_test.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])) # Normalize 3 kênh

    # Kiểm tra nếu dataset là grayscale (ví dụ: một số MedMNIST)
    if args.channels == 1:
         print("Adjusting transforms for grayscale images.")
         # Thay đổi Normalize cho 1 kênh
         t_train[-1] = transforms.Normalize(mean=[0.5], std=[0.5])
         t_test[-1] = transforms.Normalize(mean=[0.5], std=[0.5])
         # Có thể thêm chuyển đổi sang RGB nếu mô hình yêu cầu 3 kênh đầu vào
         # t_train.insert(-2, transforms.Lambda(lambda x: x.repeat(3, 1, 1))) # Lặp lại kênh grayscale 3 lần
         # t_test.insert(-2, transforms.Lambda(lambda x: x.repeat(3, 1, 1)))

    return transforms.Compose(t_train), transforms.Compose(t_test)


# --- Hàm build_dataset đã chỉnh sửa ---
def build_dataset(args):
    args.input_size = getattr(args, 'input_size', 224) # Đảm bảo có input_size
    args.channels = getattr(args, 'channels', 3) # Mặc định là 3 kênh

    train_transform, test_transform = build_transform(args)

    # Khởi tạo các downloader với cả input và working paths
    pa_downloader = PADatasetDownloader(input_root_dir=INPUT_CHECK_ROOT, working_root_dir=WORKING_ROOT)
    fetal_downloader = FetalDatasetDownloader(input_root_dir=INPUT_CHECK_ROOT, working_root_dir=WORKING_ROOT)
    cpn_downloader = CPNDatasetDownloader(input_root_dir=INPUT_CHECK_ROOT, working_root_dir=WORKING_ROOT)
    kvasir_downloader = KvasirDatasetDownloader(input_root_dir=INPUT_CHECK_ROOT, working_root_dir=WORKING_ROOT)
    isic_manager = ISICDatasetManager(input_base_dir=INPUT_CHECK_ROOT, working_base_dir=WORKING_ROOT)

    # Biến để lưu đường dẫn data cuối cùng
    data_dir = None
    train_path = None
    test_path = None

    if args.dataset == 'Kvasir':
        train_size = 2408
        val_size = 392
        test_size = 1200
        nb_classes = 8
        args.channels = 3
        data_dir = kvasir_downloader.get_dataset()
        print(f"Kvasir dataset is available at: {data_dir}")
    elif args.dataset == 'CPN':
        train_size = 3140
        val_size = 521
        test_size = 1567
        nb_classes = 3
        args.channels = 3 # Hoặc 1 nếu là X-ray grayscale? Cần xác nhận
        data_dir = cpn_downloader.get_dataset()
        print(f"CPN dataset is available at: {data_dir}")
    elif args.dataset == 'Fetal':
        train_size = 7446
        val_size = 1237
        test_size = 3717
        nb_classes = 6
        args.channels = 1 # Thường là ảnh siêu âm grayscale
        data_dir = fetal_downloader.get_dataset()
        print(f"Fetal dataset is available at: {data_dir}")
    elif args.dataset == 'PAD':
        train_size = 1384
        val_size = 227
        test_size = 687
        nb_classes = 6
        args.channels = 3
        data_dir = pa_downloader.get_dataset()
        print(f"PAD dataset is available at: {data_dir}")
    elif args.dataset == 'ISIC2018':
        nb_classes = 7
        args.channels = 3
        train_path, test_path = isic_manager.setup_dataset()
        print(f"ISIC train dataset at: {train_path}")
        print(f"ISIC test dataset at: {test_path}")
        # ISIC đã được chia sẵn train/test
        train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
        # Tạo validation set từ training set của ISIC nếu cần
        # Ví dụ: chia 80/20
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(0.2 * num_train) # 20% cho validation
        val_idx, train_idx = indices[:split], indices[split:]
        # Tạo SubsetRandomSampler hoặc Subset cho DataLoader
        # Ở đây ta trả về train_dataset đầy đủ và test_dataset riêng
        test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)
        return train_dataset, test_dataset, nb_classes # Trả về train và test riêng biệt
    elif args.dataset.endswith('mnist'):
        # MedMNIST xử lý download riêng, dùng working directory làm root
        medmnist_root = '/kaggle/working/data-kan/medmnist/' # Thư mục riêng cho medmnist trong working
        # Kiểm tra nếu có sẵn trong input
        medmnist_input_root = '/kaggle/input/data-kan/medmnist/' # Đường dẫn giả định trong input
        if os.path.exists(medmnist_input_root) and os.path.isdir(medmnist_input_root):
             print(f"MedMNIST root found in input: {medmnist_input_root}")
             medmnist_root_to_use = medmnist_input_root
        else:
             print(f"MedMNIST root not found in input, using working directory: {medmnist_root}")
             os.makedirs(medmnist_root, exist_ok=True)
             medmnist_root_to_use = medmnist_root

        info = INFO[args.dataset]
        task = info['task']
        n_channels = info['n_channels']
        nb_classes = len(info['label'])
        args.channels = n_channels # Cập nhật số kênh cho transform
        DataClass = getattr(medmnist, info['python_class'])
        print(f"Dataset: {args.dataset}")
        print("Number of channels: ", n_channels)
        print("Number of classes: ", nb_classes)
        print(f"Task: {task}")

        # Tạo lại transform với số kênh đúng
        train_transform, test_transform = build_transform(args)

        # Tải vào thư mục đã chọn (input hoặc working)
        # size=224 có thể không được hỗ trợ bởi mọi dataset MedMNIST, cần kiểm tra INFO
        img_size = args.input_size if 'size' not in info else info['size'][0] # Lấy size từ info nếu có
        print(f"Using image size: {img_size} for MedMNIST")

        train_dataset = DataClass(split='train', transform=train_transform, download=True, as_rgb=(args.channels == 3), root=medmnist_root_to_use, size=img_size)
        # Có thể cần validation split từ train
        val_dataset = DataClass(split='val', transform=test_transform, download=True, as_rgb=(args.channels == 3), root=medmnist_root_to_use, size=img_size)
        test_dataset = DataClass(split='test', transform=test_transform, download=True, as_rgb=(args.channels == 3), root=medmnist_root_to_use, size=img_size)

        # MedMNIST thường có sẵn val split, nên trả về cả 3
        return train_dataset, val_dataset, test_dataset, nb_classes
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not implemented.")

    # Xử lý split cho các dataset Kvasir, CPN, Fetal, PAD
    if data_dir:
        print(f"Loading full dataset from: {data_dir}")
        # Tạo full_dataset với transform test trước để lấy thông tin (không dùng để train)
        full_dataset_info = datasets.ImageFolder(root=data_dir, transform=test_transform)
        total_size = len(full_dataset_info)
        print(f"Total images found: {total_size}")
        expected_total = train_size + val_size + test_size
        if total_size != expected_total:
             print(f"Warning: Total images found ({total_size}) does not match expected sum of splits ({expected_total}). Adjusting split sizes proportionally.")
             # Điều chỉnh lại kích thước split dựa trên tỷ lệ
             ratio = total_size / expected_total
             train_size = int(train_size * ratio)
             val_size = int(val_size * ratio)
             # Đảm bảo test_size lấy phần còn lại để tổng khớp total_size
             test_size = total_size - train_size - val_size
             print(f"Adjusted sizes: Train={train_size}, Val={val_size}, Test={test_size}")

        # Kiểm tra lại tổng sau khi điều chỉnh
        assert train_size + val_size + test_size == total_size, "Adjusted splits still don't sum to total!"

        # Tạo dataset gốc không có transform để chia
        full_dataset_no_transform = datasets.ImageFolder(root=data_dir)

        # Chia dataset gốc
        print("Splitting dataset...")
        try:
             train_subset_indices, val_subset_indices, test_subset_indices = random_split(
                 range(total_size), [train_size, val_size, test_size],
                 generator=torch.Generator().manual_seed(seed) # Đảm bảo seed được áp dụng
             )
        except ValueError as e:
             print(f"Error during random_split: {e}")
             print(f"Check if sizes are valid: Train={train_size}, Val={val_size}, Test={test_size}, Total={total_size}")
             raise

        # Tạo các đối tượng Subset với transform tương ứng
        # Lưu ý: Cần tạo lại ImageFolder với transform *bên trong* Subset hoặc dùng wrapper class
        # Cách 1: Dùng wrapper (phức tạp hơn)
        # Cách 2: Tạo ImageFolder riêng cho từng split (dễ hơn nhưng đọc data 3 lần)
        # Cách 3: Dùng Subset và áp dụng transform trong vòng lặp dataloader (không hiệu quả)
        # Cách 4: Dùng Subset với dataset gốc và một transform riêng cho subset đó (khuyến nghị)

        # Dataset gốc với transform train
        train_ready_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
        # Dataset gốc với transform test
        test_ready_dataset = datasets.ImageFolder(root=data_dir, transform=test_transform)

        train_dataset = Subset(train_ready_dataset, train_subset_indices)
        val_dataset = Subset(test_ready_dataset, val_subset_indices) # Validation dùng test transform
        test_dataset = Subset(test_ready_dataset, test_subset_indices)

        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
        print("Number of classes = %d" % nb_classes)

        # Trả về train, val, test
        return train_dataset, val_dataset, test_dataset, nb_classes

    # Trường hợp không xác định được dataset (lỗi logic)
    raise RuntimeError("Could not determine dataset paths or type.")


# --- Ví dụ cách sử dụng ---
if __name__ == '__main__':
    # Tạo một đối tượng args giả lập cho việc test
    class Args:
        def __init__(self, dataset_name):
            self.dataset = dataset_name
            self.input_size = 224 # Kích thước ảnh đầu vào model
            # self.channels = 3 # Sẽ được cập nhật trong build_dataset

    # --- Test từng dataset ---
    # dataset_to_test = 'PAD'
    # dataset_to_test = 'Fetal'
    # dataset_to_test = 'CPN'
    # dataset_to_test = 'Kvasir'
    dataset_to_test = 'ISIC2018'
    # dataset_to_test = 'pathmnist' # Ví dụ MedMNIST
    # dataset_to_test = 'dermamnist' # Ví dụ MedMNIST

    print(f"\n===== Testing Dataset: {dataset_to_test} =====")
    args = Args(dataset_name=dataset_to_test)

    try:
        results = build_dataset(args)

        if len(results) == 4: # train, val, test, nb_classes (MedMNIST hoặc split)
            train_ds, val_ds, test_ds, num_classes = results
            print(f"Successfully built dataset '{args.dataset}'")
            print(f"Number of classes: {num_classes}")
            print(f"Train dataset size: {len(train_ds)}")
            print(f"Validation dataset size: {len(val_ds)}")
            print(f"Test dataset size: {len(test_ds)}")
        elif len(results) == 3: # train, test, nb_classes (ISIC2018)
            train_ds, test_ds, num_classes = results
            print(f"Successfully built dataset '{args.dataset}'")
            print(f"Number of classes: {num_classes}")
            print(f"Train dataset size: {len(train_ds)}")
            print(f"Test dataset size: {len(test_ds)}") # Không có validation set riêng từ hàm này
        else:
            print("build_dataset returned an unexpected number of values.")

        # Có thể thêm code để lấy một vài mẫu từ DataLoader để kiểm tra shape, etc.
        # loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        # images, labels = next(iter(loader))
        # print("Sample batch shape:", images.shape)
        # print("Sample labels:", labels)

    except Exception as e:
        print(f"\n !!!!! An error occurred while testing {args.dataset}: {e} !!!!!")
        import traceback
        traceback.print_exc()
