# ============================================================================
# Download Media Assets from GitHub
# ============================================================================
# This module downloads pre-generated figures from GitHub to ensure consistent
# display in both Google Colab and local environments.

import os
import urllib.request
from typing import List, Optional


def download_github_file(
    filename: str,
    media_root: str,
    github_user: str,
    github_repo: str,
    github_branch: str,
    media_folder_url: str,
    force: bool = False
) -> Optional[str]:
    """
    Download a single file from GitHub raw content.
    
    Args:
        filename: Name of the file to download
        media_root: Local directory to save the file
        github_user: GitHub username
        github_repo: GitHub repository name
        github_branch: Git branch (e.g., 'main')
        media_folder_url: Path within the repo (forward slash separated)
        force: If True, re-download even if file exists
        
    Returns:
        Local path to the downloaded file, or None on failure
    """
    local_path = os.path.join(media_root, filename)
    
    # Skip if file exists and not forcing re-download
    if os.path.exists(local_path) and not force:
        print(f"  ✓ {filename} (already exists)")
        return local_path
    
    # URL always uses forward slashes
    url = f"https://raw.githubusercontent.com/{github_user}/{github_repo}/{github_branch}/{media_folder_url}/{filename}"
    
    try:
        urllib.request.urlretrieve(url, local_path)
        print(f"  ✓ {filename} (downloaded)")
        return local_path
    except Exception as e:
        print(f"  ✗ {filename} - Error: {e}")
        return None


def download_media_assets(
    github_user: str,
    github_repo: str,
    github_branch: str,
    media_root: str,
    media_files: List[str],
    force: bool = False
) -> str:
    """
    Download media assets from a GitHub repository.
    
    Args:
        github_user: GitHub username
        github_repo: GitHub repository name
        github_branch: Git branch (e.g., 'main')
        media_root: Local directory to save files
        media_files: List of filenames to download
        force: If True, re-download even if files exist
        
    Returns:
        The media_root path where files are stored
    """
    # Extract media folder path from media_root for URL construction
    # Convert backslashes to forward slashes for URL
    media_folder_url = media_root.replace("\\", "/").lstrip("./")
    
    # Create media directory if it doesn't exist
    os.makedirs(media_root, exist_ok=True)
    
    print(f"Media folder: {os.path.abspath(media_root)}")
    print(f"Downloading media from: {github_user}/{github_repo}")
    print("-" * 50)
    
    for filename in media_files:
        download_github_file(
            filename=filename,
            media_root=media_root,
            github_user=github_user,
            github_repo=github_repo,
            github_branch=github_branch,
            media_folder_url=media_folder_url,
            force=force
        )
    
    print("-" * 50)
    print(f"Done! Media available at: {media_root}")
    
    return media_root
