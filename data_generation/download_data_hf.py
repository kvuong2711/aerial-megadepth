import os
import argparse
import zipfile
import glob
from huggingface_hub import snapshot_download


def download_from_huggingface(repo_id: str, local_dir: str, repo_type: str = "dataset", 
                             allow_patterns: list = None, max_workers: int = 8, 
                             resume_download: bool = True, unzip: bool = False, 
                             remove_zip: bool = False, create_subdirs: bool = False):
    """Downloads data from Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face repository ID (e.g., "username/dataset-name")
        local_dir: Local directory to save the downloaded data
        repo_type: Type of repository ("dataset", "model", "space")
        allow_patterns: List of file patterns to download (e.g., ["**.zip"])
        max_workers: Maximum number of workers for parallel download
        resume_download: Whether to resume interrupted downloads
        unzip: Whether to unzip the downloaded zip files
        remove_zip: Whether to remove zip files after extraction
        create_subdirs: Whether to create subdirectories based on zip file names
    """
    # Create output directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    print(f'>>>>> Downloading from Hugging Face: {repo_id}')
    print(f'Repository type: {repo_type}')
    print(f'Local directory: {local_dir}')
    if allow_patterns:
        print(f'Allowed patterns: {allow_patterns}')
    
    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=local_dir,
            resume_download=resume_download,
            max_workers=max_workers,
            allow_patterns=allow_patterns,
        )
        print(f'Successfully downloaded to: {downloaded_path}')
        
        # Unzip files if requested
        if unzip:
            print(f'Unzipping files...')
            zip_files = glob.glob(os.path.join(downloaded_path, "**/*.zip"), recursive=True)
            
            for zip_file in zip_files:
                print(f'Unzipping {zip_file}...')
                if create_subdirs:
                    # Create a subdirectory based on the zip file name (without extension)
                    zip_name = os.path.splitext(os.path.basename(zip_file))[0]
                    extract_path = os.path.join(os.path.dirname(zip_file), zip_name)
                    os.makedirs(extract_path, exist_ok=True)
                else:
                    # Extract directly to the same directory as the zip file
                    extract_path = os.path.dirname(zip_file)
                
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                print(f'Unzipped to {extract_path}')
                
                # Remove the zip file if requested
                if remove_zip:
                    os.remove(zip_file)
                    print(f'Removed zip file {zip_file}')
        
        return downloaded_path
    except Exception as e:
        print(f'Error downloading from Hugging Face: {e}')
        raise

def main():
    parser = argparse.ArgumentParser(description='Download aerial-megadepth data from Hugging Face Hub')
    
    # Required arguments
    parser.add_argument('--repo_id', type=str, required=True,
                       default='kvuong2711/aerialmegadepth',
                       help='Hugging Face repository ID (e.g., "username/dataset-name")')
    parser.add_argument('--local_dir', type=str, required=True,
                       default='/mnt/slarge2/megadepth_aerial_data/data',
                       help='Local directory to save the downloaded data')
    
    # Pattern options
    parser.add_argument('--allow_patterns', type=str, nargs='*',
                       default=["**.zip"],
                       help='File patterns to download (default: ["**.zip"])')
    
    # Unzip options
    parser.add_argument('--unzip', action='store_true',
                       default=False,
                       help='Unzip the downloaded zip files')
    parser.add_argument('--remove_zip', action='store_true',
                       default=False,
                       help='Remove the zip files after extraction')
    parser.add_argument('--create_subdirs', action='store_true',
                       default=False,
                       help='Create subdirectories based on zip file names')
    
    args = parser.parse_args()
    
    # Download from Hugging Face
    download_from_huggingface(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        repo_type='dataset',
        allow_patterns=args.allow_patterns,
        max_workers=8,
        resume_download=True,
        unzip=args.unzip,
        remove_zip=args.remove_zip,
        create_subdirs=args.create_subdirs
    )

if __name__ == '__main__':
    main()