import yt_dlp
import asyncio
import aiohttp
import aiofiles
import os
import time
import multiprocessing
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("youtube-downloader")

app = FastAPI(title="High-Performance YouTube Downloader API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class DownloadRequest(BaseModel):
    url: str
    format: Optional[str] = "best"
    output_dir: Optional[str] = "./downloads"
    max_connections: Optional[int] = 16
    chunk_size: Optional[int] = 1024 * 1024  # 1MB chunks by default

class DownloadResponse(BaseModel):
    download_id: str
    estimated_size: Optional[int]
    formats_available: List[Dict[str, Any]]
    status: str = "queued"

class DownloadStatus(BaseModel):
    download_id: str
    status: str
    progress: float
    speed: Optional[str]
    eta: Optional[str]
    file_path: Optional[str]

# In-memory storage for active downloads (would use a proper database in production)
active_downloads = {}

# Optimization: Use a thread pool for CPU-bound operations
cpu_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

async def get_video_info(url: str) -> Dict[str, Any]:
    """Extract video information without downloading."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'format': 'best',
    }
    
    loop = asyncio.get_event_loop()
    
    def _extract_info():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=False)
    
    # Run in a separate thread to avoid blocking
    return await loop.run_in_executor(None, _extract_info)

async def download_segment(session, url, start_byte, end_byte, output_file, segment_id):
    """Download a specific byte range of the file."""
    headers = {'Range': f'bytes={start_byte}-{end_byte}'}
    
    try:
        async with session.get(url, headers=headers) as response:
            if response.status == 206:  # Partial Content
                # Create a temporary file for this segment
                temp_file = f"{output_file}.part{segment_id}"
                async with aiofiles.open(temp_file, 'wb') as f:
                    async for chunk in response.content.iter_chunked(1024 * 1024):  # 1MB chunks
                        await f.write(chunk)
                return temp_file, segment_id
            else:
                logger.error(f"Failed to download segment {segment_id}: HTTP {response.status}")
                return None, segment_id
    except Exception as e:
        logger.error(f"Error downloading segment {segment_id}: {str(e)}")
        return None, segment_id

async def parallel_download(url, output_file, file_size, max_connections=16):
    """Download a file in parallel segments."""
    if file_size is None:
        # Fall back to single connection if size is unknown
        return await single_connection_download(url, output_file)
    
    # Calculate segment size
    segment_size = max(file_size // max_connections, 1024 * 1024)  # At least 1MB per segment
    segments = []
    
    for i in range(max_connections):
        start_byte = i * segment_size
        end_byte = min((i + 1) * segment_size - 1, file_size - 1)
        if start_byte <= end_byte:
            segments.append((start_byte, end_byte, i))
    
    # Create an aiohttp session for all downloads
    async with aiohttp.ClientSession() as session:
        # Download all segments in parallel
        tasks = [
            download_segment(session, url, start, end, output_file, idx)
            for start, end, idx in segments
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Filter out failed segments
        successful_segments = [r[0] for r in results if r[0] is not None]
        
        if len(successful_segments) != len(segments):
            logger.warning(f"Some segments failed to download: {len(segments) - len(successful_segments)} of {len(segments)}")
            
        # Combine segments into the final file
        async with aiofiles.open(output_file, 'wb') as outfile:
            for segment_file, _ in sorted(results, key=lambda x: x[1]):
                if segment_file:
                    async with aiofiles.open(segment_file, 'rb') as infile:
                        await outfile.write(await infile.read())
                    # Remove the temporary segment file
                    os.remove(segment_file)
                    
        return output_file

async def single_connection_download(url, output_file):
    """Fallback method for when parallel download isn't possible."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                async with aiofiles.open(output_file, 'wb') as f:
                    async for chunk in response.content.iter_chunked(1024 * 1024):  # 1MB chunks
                        await f.write(chunk)
                return output_file
            else:
                raise Exception(f"Failed to download: HTTP {response.status}")

async def get_direct_url(video_info, format_id="best"):
    """Extract the direct URL for the video file."""
    if format_id == "best":
        # Get the best format with both video and audio
        formats = video_info.get('formats', [])
        formats_with_video_audio = [f for f in formats if f.get('vcodec') != 'none' and f.get('acodec') != 'none']
        
        if formats_with_video_audio:
            # Sort by quality (resolution, bitrate)
            best_format = max(formats_with_video_audio, 
                              key=lambda x: (x.get('height', 0) or 0, x.get('tbr', 0) or 0))
            return best_format.get('url'), best_format.get('filesize'), best_format
    
    # If specific format or fallback needed
    for fmt in video_info.get('formats', []):
        if fmt.get('format_id') == format_id:
            return fmt.get('url'), fmt.get('filesize'), fmt
    
    # Last resort
    return video_info.get('url'), video_info.get('filesize'), None

async def process_download(download_id: str, request: DownloadRequest):
    """Process a download request asynchronously."""
    try:
        # Update status to processing
        active_downloads[download_id]["status"] = "processing"
        
        # Get video info
        video_info = await get_video_info(request.url)
        
        if not video_info:
            raise Exception("Failed to extract video information")
        
        # Get filename and create directories
        video_title = video_info.get('title', 'video').replace('/', '_')
        os.makedirs(request.output_dir, exist_ok=True)
        
        # Get the direct URL and file size
        direct_url, file_size, format_info = await get_direct_url(video_info, request.format)
        
        if not direct_url:
            raise Exception("Could not extract direct download URL")
        
        # Update download info
        active_downloads[download_id].update({
            "title": video_title,
            "estimated_size": file_size,
            "format_info": format_info
        })
        
        # Determine output file path
        ext = format_info.get('ext', 'mp4') if format_info else 'mp4'
        output_file = os.path.join(request.output_dir, f"{video_title}.{ext}")
        
        # Start time for speed calculation
        start_time = time.time()
        
        # Update status to downloading
        active_downloads[download_id]["status"] = "downloading"
        active_downloads[download_id]["start_time"] = start_time
        
        # Check if we can use parallel download
        if file_size and file_size > 10 * 1024 * 1024:  # Only for files > 10MB
            logger.info(f"Using parallel download for {video_title}")
            await parallel_download(
                direct_url, 
                output_file, 
                file_size, 
                max_connections=request.max_connections
            )
        else:
            logger.info(f"Using single connection download for {video_title}")
            await single_connection_download(direct_url, output_file)
        
        # Calculate download stats
        end_time = time.time()
        duration = end_time - start_time
        actual_size = os.path.getsize(output_file)
        speed = actual_size / duration if duration > 0 else 0
        
        # Update download status to completed
        active_downloads[download_id].update({
            "status": "completed",
            "progress": 1.0,
            "file_path": output_file,
            "actual_size": actual_size,
            "duration": duration,
            "speed": f"{speed / (1024 * 1024):.2f} MB/s"
        })
        
        logger.info(f"Download completed: {video_title}")
        logger.info(f"Speed: {speed / (1024 * 1024):.2f} MB/s")
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        active_downloads[download_id].update({
            "status": "failed",
            "error": str(e)
        })

@app.post("/download", response_model=DownloadResponse)
async def start_download(request: DownloadRequest):
    """Initialize a new download."""
    download_id = f"dl_{int(time.time())}_{os.urandom(4).hex()}"
    
    # Initialize download info
    active_downloads[download_id] = {
        "download_id": download_id,
        "status": "queued",
        "progress": 0,
        "request": request.dict()
    }
    
    try:
        # Get video info to return available formats
        video_info = await get_video_info(request.url)
        formats_available = [
            {
                "format_id": fmt.get("format_id"),
                "ext": fmt.get("ext"),
                "resolution": f"{fmt.get('width', 0)}x{fmt.get('height', 0)}",
                "filesize": fmt.get("filesize"),
                "vcodec": fmt.get("vcodec"),
                "acodec": fmt.get("acodec"),
            }
            for fmt in video_info.get("formats", [])
            if fmt.get("vcodec") != "none" or fmt.get("acodec") != "none"
        ]
        
        # Store formats for later use
        active_downloads[download_id]["formats_available"] = formats_available
        
        # Start download process in background
        asyncio.create_task(process_download(download_id, request))
        
        # Return initial response
        return DownloadResponse(
            download_id=download_id,
            estimated_size=video_info.get("filesize"),
            formats_available=formats_available,
            status="queued"
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize download: {str(e)}")
        active_downloads[download_id]["status"] = "failed"
        active_downloads[download_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{download_id}", response_model=DownloadStatus)
async def get_download_status(download_id: str):
    """Get the status of a download."""
    if download_id not in active_downloads:
        raise HTTPException(status_code=404, detail="Download not found")
    
    download_info = active_downloads[download_id]
    
    # Calculate progress and speed for active downloads
    if download_info["status"] == "downloading":
        if "start_time" in download_info and "estimated_size" in download_info:
            elapsed = time.time() - download_info["start_time"]
            
            # We'd need to check actual downloaded bytes, but this is simplified
            # In a real app, we'd track actual bytes downloaded
            progress = min(elapsed / 60, 0.99)  # Estimate based on time, max 99%
            
            estimated_size = download_info["estimated_size"] or 10 * 1024 * 1024  # Default 10MB
            estimated_downloaded = estimated_size * progress
            
            if elapsed > 0:
                speed = f"{(estimated_downloaded / elapsed) / (1024 * 1024):.2f} MB/s"
                remaining = (estimated_size - estimated_downloaded) / (estimated_downloaded / elapsed)
                eta = f"{int(remaining // 60)}m {int(remaining % 60)}s"
            else:
                speed = "calculating..."
                eta = "calculating..."
                
            download_info["progress"] = progress
            download_info["speed"] = speed
            download_info["eta"] = eta
    
    return DownloadStatus(
        download_id=download_id,
        status=download_info["status"],
        progress=download_info.get("progress", 0),
        speed=download_info.get("speed"),
        eta=download_info.get("eta"),
        file_path=download_info.get("file_path")
    )

@app.get("/formats", response_model=List[Dict[str, Any]])
async def get_available_formats(url: str = Query(..., description="YouTube URL")):
    """Get available formats for a video without starting a download."""
    try:
        video_info = await get_video_info(url)
        
        formats = [
            {
                "format_id": fmt.get("format_id"),
                "ext": fmt.get("ext"),
                "resolution": f"{fmt.get('width', 0)}x{fmt.get('height', 0)}",
                "filesize": fmt.get("filesize", 0),
                "filesize_approx": fmt.get("filesize_approx", 0),
                "vcodec": fmt.get("vcodec"),
                "acodec": fmt.get("acodec"),
                "fps": fmt.get("fps"),
                "tbr": fmt.get("tbr")
            }
            for fmt in video_info.get("formats", [])
            if fmt.get("vcodec") != "none" or fmt.get("acodec") != "none"
        ]
        
        return formats
    except Exception as e:
        logger.error(f"Failed to get formats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/download/{download_id}")
async def cancel_download(download_id: str):
    """Cancel an ongoing download."""
    if download_id not in active_downloads:
        raise HTTPException(status_code=404, detail="Download not found")
    
    # In a real application, we would need to properly cancel the download task
    # This is a simplified version
    active_downloads[download_id]["status"] = "cancelled"
    
    return {"status": "cancelled", "download_id": download_id}

if __name__ == "__main__":
    # Run the FastAPI app with Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)