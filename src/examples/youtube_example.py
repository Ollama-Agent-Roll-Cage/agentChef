"""Example demonstrating the YouTubeDownloader class from oarc-crawlers."""

import asyncio
from pathlib import Path
from oarc_crawlers import YouTubeDownloader
from agentChef.logs.agentchef_logging import log

logger = log

async def youtube_downloader_example():
    """Demonstrate YouTube downloading capabilities."""
    
    # Create output directory if it doesn't exist
    output_dir = Path("./output/youtube")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the downloader
    downloader = YouTubeDownloader(data_dir=str(output_dir))
    
    # 1. Search for videos
    print("Searching for videos about 'transformer neural networks'...")
    search_results = await downloader.search_videos("transformer neural networks", limit=5)
    
    # Show search results
    if "results" in search_results:
        print(f"Found {len(search_results['results'])} videos")
        for i, video in enumerate(search_results['results'][:3]):  # Show first 3
            print(f"\n{i+1}. {video['title']}")
            print(f"   URL: {video['url']}")
            print(f"   Author: {video['author']}")
            
        # Get first video URL for downloading demo
        if search_results['results']:
            video_url = search_results['results'][0]['url']
            
            # 2. Extract captions from the video
            print("\nExtracting captions from the first video...")
            captions = await downloader.extract_captions(video_url)
            
            if "captions" in captions:
                print(f"Found captions in {len(captions['captions'])} languages")
                for lang, path in captions['captions'].items():
                    print(f"Language: {lang}, Saved to: {path}")
            else:
                print("No captions found or error extracting captions")
                
            # 3. Download only audio from the video (to avoid large downloads)
            print("\nDownloading audio from the video...")
            result = await downloader.download_video(
                video_url, 
                format="mp4", 
                extract_audio=True
            )
            
            if "file_path" in result:
                print(f"Downloaded to: {result['file_path']}")
                print(f"File size: {result.get('file_size', 'unknown')} bytes")
            else:
                print(f"Error downloading: {result.get('error', 'unknown error')}")
    else:
        print(f"Error searching: {search_results.get('error', 'unknown error')}")

if __name__ == "__main__":
    asyncio.run(youtube_downloader_example())