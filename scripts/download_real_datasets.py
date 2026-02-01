"""
STEP 1: Download real datasets from official sources
- OULAD: UCI official zip
- UCI697: UCI official zip  
- HarvardX Person-Course: Harvard Dataverse API
"""
import sys
import io
import os
import zipfile
import requests
from pathlib import Path
import json

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def download_file(url: str, dest_path: Path, chunk_size: int = 8192):
    """Download a file with progress"""
    print(f"Downloading from: {url}")
    print(f"Destination: {dest_path}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
    
    print()  # New line after progress
    print(f"Downloaded: {dest_path} ({downloaded} bytes)")
    return dest_path

def unzip_file(zip_path: Path, extract_to: Path):
    """Unzip a file"""
    print(f"Extracting {zip_path} to {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"Extracted to: {extract_to}")
    return extract_to

def download_oulad(data_dir: Path):
    """Download OULAD dataset from UCI"""
    print("\n" + "="*80)
    print("A) Downloading OULAD dataset")
    print("="*80)
    
    url = "https://archive.ics.uci.edu/static/public/349/open%2Buniversity%2Blearning%2Banalytics%2Bdataset.zip"
    zip_path = data_dir / "raw" / "oulad.zip"
    extract_to = data_dir / "raw" / "oulad"
    
    try:
        download_file(url, zip_path)
        unzip_file(zip_path, extract_to)
        
        # Check if studentInfo.csv exists
        student_info = extract_to / "studentInfo.csv"
        if not student_info.exists():
            # Try to find it in subdirectories
            for csv_file in extract_to.rglob("studentInfo.csv"):
                student_info = csv_file
                break
        
        if student_info.exists():
            print(f"SUCCESS: Found studentInfo.csv at {student_info}")
            return True
        else:
            print(f"WARNING: studentInfo.csv not found in {extract_to}")
            print("Listing extracted files:")
            for f in extract_to.rglob("*"):
                if f.is_file():
                    print(f"  {f}")
            return False
    except Exception as e:
        print(f"ERROR downloading OULAD: {e}")
        return False

def download_uci697(data_dir: Path):
    """Download UCI697 dataset from UCI"""
    print("\n" + "="*80)
    print("B) Downloading UCI697 dataset")
    print("="*80)
    
    url = "https://cdn.uci-ics-mlr-prod.aws.uci.edu/697/predict%2Bstudents%2Bdropout%2Band%2Bacademic%2Bsuccess.zip"
    zip_path = data_dir / "raw" / "uci697.zip"
    extract_to = data_dir / "raw" / "uci697"
    
    try:
        download_file(url, zip_path)
        unzip_file(zip_path, extract_to)
        
        # Check if student-mat.csv exists
        student_mat = extract_to / "student-mat.csv"
        if not student_mat.exists():
            # Try to find it in subdirectories
            for csv_file in extract_to.rglob("student-mat.csv"):
                student_mat = csv_file
                break
        
        if student_mat.exists():
            print(f"SUCCESS: Found student-mat.csv at {student_mat}")
            return True
        else:
            print(f"WARNING: student-mat.csv not found in {extract_to}")
            print("Listing extracted files:")
            for f in extract_to.rglob("*"):
                if f.is_file():
                    print(f"  {f}")
            return False
    except Exception as e:
        print(f"ERROR downloading UCI697: {e}")
        return False

def download_harvardx_dataverse(data_dir: Path):
    """Download HarvardX Person-Course dataset from Dataverse"""
    print("\n" + "="*80)
    print("C) Downloading HarvardX Person-Course dataset from Dataverse")
    print("="*80)
    
    persistent_id = "doi:10.7910/DVN/26147"
    
    # Method 1: Try Data Access API
    print("\nTrying Method 1: Dataverse Data Access API")
    try:
        url = f"https://dataverse.harvard.edu/api/access/dataset/:persistentId?persistentId={persistent_id}"
        print(f"Requesting: {url}")
        
        response = requests.get(url, allow_redirects=True)
        response.raise_for_status()
        
        # Check if response is a zip file
        content_type = response.headers.get('content-type', '')
        if 'zip' in content_type or response.content[:2] == b'PK':
            zip_path = data_dir / "raw" / "harvardx_dvn26147.zip"
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            print(f"Downloaded zip file: {zip_path} ({len(response.content)} bytes)")
            extract_to = data_dir / "raw" / "harvardx"
            unzip_file(zip_path, extract_to)
            
            # Check for the CSV file
            csv_file = extract_to / "HMXPC13_DI_v2_5-14-14.csv"
            if not csv_file.exists():
                for f in extract_to.rglob("*.csv"):
                    csv_file = f
                    break
            
            if csv_file.exists():
                print(f"SUCCESS: Found CSV file at {csv_file}")
                return True
        else:
            print(f"Response is not a zip file. Content-Type: {content_type}")
            print("Trying Method 2...")
    except Exception as e:
        print(f"Method 1 failed: {e}")
        print("Trying Method 2...")
    
    # Method 2: Use Native API to get file IDs
    print("\nTrying Method 2: Dataverse Native API")
    try:
        # Get dataset metadata
        metadata_url = f"https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId={persistent_id}"
        print(f"Requesting metadata: {metadata_url}")
        
        response = requests.get(metadata_url)
        response.raise_for_status()
        metadata = response.json()
        
        # Extract file IDs
        files = []
        if 'data' in metadata and 'latestVersion' in metadata['data']:
            files = metadata['data']['latestVersion'].get('files', [])
        
        if not files:
            print("No files found in metadata")
            return False
        
        print(f"Found {len(files)} file(s)")
        
        # Download each file
        extract_to = data_dir / "raw" / "harvardx"
        extract_to.mkdir(parents=True, exist_ok=True)
        
        for file_info in files:
            file_id = file_info.get('dataFile', {}).get('id')
            if not file_id:
                continue
            
            file_name = file_info.get('dataFile', {}).get('filename', f'file_{file_id}')
            print(f"\nDownloading file: {file_name} (ID: {file_id})")
            
            file_url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
            file_path = extract_to / file_name
            
            download_file(file_url, file_path)
        
        # Check for CSV file
        csv_file = extract_to / "HMXPC13_DI_v2_5-14-14.csv"
        if not csv_file.exists():
            for f in extract_to.rglob("*.csv"):
                csv_file = f
                break
        
        if csv_file.exists():
            print(f"SUCCESS: Found CSV file at {csv_file}")
            return True
        else:
            print("WARNING: CSV file not found")
            print("Listing downloaded files:")
            for f in extract_to.iterdir():
                if f.is_file():
                    print(f"  {f}")
            return False
            
    except Exception as e:
        print(f"Method 2 failed: {e}")
        return False

def main():
    """Main download function"""
    print("="*80)
    print("STEP 1: Download Real Datasets")
    print("="*80)
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    (data_dir / "raw").mkdir(exist_ok=True)
    
    results = {}
    
    # Download OULAD
    results['OULAD'] = download_oulad(data_dir)
    
    # Download UCI697
    results['UCI697'] = download_uci697(data_dir)
    
    # Download HarvardX
    results['HarvardX_PersonCourse'] = download_harvardx_dataverse(data_dir)
    
    # Summary
    print("\n" + "="*80)
    print("Download Summary:")
    print("="*80)
    for dataset, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"{dataset:<30} {status}")
    
    all_success = all(results.values())
    
    if all_success:
        print("\nAll datasets downloaded successfully!")
        print("\nNext steps:")
        print("1. Run: python scripts/verify_synthetic_usage.py")
        print("2. If verification passes, proceed to STEP 2 (disable synthetic fallback)")
    else:
        print("\nSome downloads failed. Please check errors above.")
        print("Do NOT proceed until all datasets are downloaded.")
    
    return all_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
