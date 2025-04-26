import os
import gdown
import zipfile
import argparse

def parse_download_links(links_file: str = None, gdrive_link: str = None) -> list[str]:
    """Parses input (either a file or single link) into a list of Google Drive links.
    
    Args:
        links_file: Path to text file containing filename and Google Drive links
        gdrive_link: Single Google Drive link to download
        
    Returns:
        List of Google Drive links
    """
    if links_file is None and gdrive_link is None:
        raise ValueError("Either links_file or gdrive_link must be provided")
        
    if gdrive_link:
        return [gdrive_link]
        
    # Parse links file
    with open(links_file, 'r') as f:
        links = f.readlines()
    
    urls = []
    for line in links:
        if not line.strip():
            continue
        _, url = line.strip().split(': ')
        urls.append(url)
        
    return urls

def download_and_extract_data(output_path: str, urls: list[str], unzip: bool = True, remove_zip: bool = True):
    """Downloads and extracts data from Google Drive links.
    
    Args:
        output_path: Path where the extracted data should be saved
        urls: List of Google Drive links to download
        unzip: Whether to unzip the downloaded file
        remove_zip: Whether to remove the zip file after extraction
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    # Ensure output path is a directory and ends with a slash
    if not output_path.endswith('/'):
        output_path += '/'
    
    # Process each file
    for url in urls:
        # Download file
        print(f'>>>>> Downloading {url} to {output_path}')
        output = gdown.download(url=url, output=output_path, fuzzy=True, quiet=False)
        print(f'Downloaded to {output}')
        
        # Unzip the downloaded file
        if unzip:
            print(f'Unzipping file...')
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall(output_path)
            print(f'Unzipped to {output_path}')
        
        # Remove the zip file
        if remove_zip:
            os.remove(output)
            print(f'Removed zip file {output}')

def main():
    parser = argparse.ArgumentParser(description='Download and extract aerial-megadepth data')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path where the extracted data should be saved')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--gdrive_links_file', type=str,
                        help='Path to text file containing filename and Google Drive links')
    group.add_argument('--gdrive_link', type=str,
                        help='Single Google Drive link to download')
    
    parser.add_argument('--unzip', action='store_true',
                        default=True,
                        help='Unzip the downloaded files')
    parser.add_argument('--remove_zip', action='store_true',
                        default=True,
                        help='Remove the zip files after extraction')
    args = parser.parse_args()

    # Parse the links first
    urls = parse_download_links(args.gdrive_links_file, args.gdrive_link)

    # Then download and extract
    download_and_extract_data(
        args.output_path,
        urls,
        args.unzip,
        args.remove_zip
    )

if __name__ == '__main__':
    main()
