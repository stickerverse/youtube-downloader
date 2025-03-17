import yt_dlp
import asyncio
import os
import time
import json
import tempfile
import logging
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Set, Union
import uuid
import psutil
from datetime import datetime

from advanced_downloader import AdaptiveDownloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("youtube-downloader")

app = FastAPI(title="Advanced YouTube Downloader API")

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
    audio_only: Optional[bool] = False
    video_only: Optional[bool] = False
    quality: Optional[str] = "best"  # best, medium, worst
    convert_format: Optional[str] = None  # mp4, mp3, etc.

class DownloadResponse(BaseModel):
    download_id: str
    estimated_size: Optional[int]
    formats_available: List[Dict[str, Any]]
    video_info: Dict[str, Any]
    status: str = "queued"

class DownloadStatus(BaseModel):
    download_id: str
    status: str
    progress: float
    downloaded_bytes: int
    total_bytes: int
    speed: Optional[str]
    eta: Optional[str]
    file_path: Optional[str]
    active_connections: Optional[int]
    format_details: Optional[Dict[str, Any]]
    start_time: Optional[float]
    elapsed: Optional[float]

class SystemStatus(BaseModel):
    cpu_percent: float
    memory_percent: float
    download_count: int
    active_downloads: int
    completed_downloads: int
    failed_downloads: int
    disk_usage_percent: float
    uptime: float

# In-memory storage for active downloads
active_downloads = {}
connected_clients: Set[WebSocket] = set()
start_time = time.time()

# Thread pool for CPU-bound operations
thread_pool = ThreadPoolExecutor(max_workers=min(os.cpu_count() * 2, 8))

# Cache directory for temporary files
cache_dir = os.path.join(tempfile.gettempdir(), "youtube-downloader")
os.makedirs(cache_dir, exist_ok=True)

# Create downloads directory
downloads_dir = "./downloads"
os.makedirs(downloads_dir, exist_ok=True)

def get_system_status() -> SystemStatus:
    """Get system status information"""
    download_count = len(active_downloads)
    active_count = sum(1 for info in active_downloads.values() if info["status"] in ["downloading", "processing", "queued"])
    completed_count = sum(1 for info in active_downloads.values() if info["status"] == "completed")
    failed_count = sum(1 for info in active_downloads.values() if info["status"] == "failed")
    
    disk = psutil.disk_usage(os.path.abspath(downloads_dir))
    
    return SystemStatus(
        cpu_percent=psutil.cpu_percent(),
        memory_percent=psutil.virtual_memory().percent,
        download_count=download_count,
        active_downloads=active_count,
        completed_downloads=completed_count,
        failed_downloads=failed_count,
        disk_usage_percent=disk.percent,
        uptime=time.time() - start_time
    )

def run_in_thread(func):
    """Run a function in a thread pool to avoid blocking the event loop"""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(thread_pool, func)

async def extract_video_info(url: str) -> Dict[str, Any]:
    """Extract detailed video information using yt-dlp"""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'format': 'best',
        'writeinfojson': False,
    }
    
    def _extract_info():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            # Clean up info to make it serializable
            if info.get('formats'):
                for fmt in info['formats']:
                    # Remove non-serializable objects
                    for key in list(fmt.keys()):
                        if not isinstance(fmt[key], (str, int, float, bool, type(None), list, dict)):
                            fmt[key] = str(fmt[key])
            return info
    
    # Run in thread pool to avoid blocking
    return await run_in_thread(_extract_info)

async def get_best_download_url(video_info: Dict[str, Any], format_id: str = "best", 
                              audio_only: bool = False, video_only: bool = False,
                              quality: str = "best") -> Dict[str, Any]:
    """
    Get the best download URL based on format preferences.
    Returns a dict with url, filesize, and format info.
    """
    formats = video_info.get('formats', [])
    
    # Filter formats based on preferences
    if audio_only:
        filtered_formats = [f for f in formats if f.get('acodec') != 'none' and (f.get('vcodec') == 'none' or quality != "best")]
    elif video_only:
        filtered_formats = [f for f in formats if f.get('vcodec') != 'none' and f.get('acodec') == 'none']
    else:
        filtered_formats = [f for f in formats if f.get('vcodec') != 'none' and f.get('acodec') != 'none']
    
    # If specific format requested
    if format_id != "best":
        for fmt in formats:
            if fmt.get('format_id') == format_id:
                return {
                    'url': fmt.get('url'),
                    'filesize': fmt.get('filesize'),
                    'format_info': fmt
                }
    
    # Sort by quality
    if quality == "best":
        # Sort by resolution and bitrate
        if filtered_formats:
            selected_format = max(filtered_formats, 
                              key=lambda x: (x.get('height', 0) or 0, x.get('tbr', 0) or 0))
            return {
                'url': selected_format.get('url'),
                'filesize': selected_format.get('filesize'),
                'format_info': selected_format
            }
    elif quality == "worst":
        # Sort by lowest resolution and bitrate
        if filtered_formats:
            selected_format = min(filtered_formats, 
                              key=lambda x: (x.get('height', 0) or float('inf'), x.get('tbr', 0) or float('inf')))
            return {
                'url': selected_format.get('url'),
                'filesize': selected_format.get('filesize'),
                'format_info': selected_format
            }
    elif quality == "medium":
        # Try to find a medium quality format
        if filtered_formats:
            # Sort by resolution
            sorted_formats = sorted(filtered_formats, 
                                   key=lambda x: (x.get('height', 0) or 0))
            
            # Pick format in the middle
            mid_idx = len(sorted_formats) // 2
            selected_format = sorted_formats[mid_idx]
            return {
                'url': selected_format.get('url'),
                'filesize': selected_format.get('filesize'),
                'format_info': selected_format
            }
    
    # Fallback to default best format
    default_format = video_info.get('requested_formats', [{}])[0] if video_info.get('requested_formats') else {}
    return {
        'url': default_format.get('url') or video_info.get('url'),
        'filesize': default_format.get('filesize') or video_info.get('filesize'),
        'format_info': default_format or {}
    }

async def process_websocket_progress(websocket: WebSocket, progress_info: Dict[str, Any]):
    """Send progress updates to websocket clients"""
    try:
        await websocket.send_json(progress_info)
    except Exception as e:
        logger.error(f"Error sending websocket update: {str(e)}")
        # Client might have disconnected
        if websocket in connected_clients:
            connected_clients.remove(websocket)

async def download_progress_callback(download_id: str, progress_info: Dict[str, Any]):
    """Callback for download progress updates"""
    if download_id in active_downloads:
        # Update the download status
        active_downloads[download_id].update({
            "progress": progress_info["progress"],
            "downloaded_bytes": progress_info["downloaded_bytes"],
            "speed": f"{progress_info['current_speed_mbs']:.2f} MB/s",
            "active_connections": progress_info["active_connections"],
            "optimal_connections": progress_info["optimal_connections"],
            "eta": f"{int(progress_info['eta_seconds'] // 60)}m {int(progress_info['eta_seconds'] % 60)}s" if progress_info.get("eta_seconds") else None,
        })
        
        # Send update to connected websockets
        update = {
            "type": "download_progress",
            "download_id": download_id,
            "data": {
                **progress_info,
                "status": active_downloads[download_id]["status"]
            }
        }
        
        for client in connected_clients:
            asyncio.create_task(process_websocket_progress(client, update))

async def process_download(download_id: str, request: DownloadRequest):
    """Process a download request asynchronously using the advanced downloader"""
    try:
        # Update status to processing
        active_downloads[download_id]["status"] = "processing"
        
        # Extract video information
        video_info = await extract_video_info(request.url)
        
        if not video_info:
            raise Exception("Failed to extract video information")
        
        # Get best download format based on preferences
        download_format = await get_best_download_url(
            video_info,
            format_id=request.format,
            audio_only=request.audio_only,
            video_only=request.video_only,
            quality=request.quality
        )
        
        if not download_format.get('url'):
            raise Exception("Could not determine download URL")
        
        # Create safe filename from video title
        video_title = video_info.get('title', 'video')
        safe_title = "".join([c if c.isalnum() or c in " ._-" else "_" for c in video_title])
        
        # Ensure output directory exists
        os.makedirs(request.output_dir, exist_ok=True)
        
        # Determine output file extension
        ext = download_format.get('format_info', {}).get('ext', 'mp4')
        if request.convert_format:
            ext = request.convert_format
            
        # Create full output path
        output_file = os.path.join(request.output_dir, f"{safe_title}.{ext}")
        
        # Update download info
        active_downloads[download_id].update({
            "title": video_title,
            "estimated_size": download_format.get('filesize'),
            "format_details": download_format.get('format_info'),
            "output_file": output_file,
            "status": "downloading",
            "start_time": time.time()
        })
        
        # Create a callback for this specific download
        async def progress_handler(progress_data):
            await download_progress_callback(download_id, progress_data)
        
        # Initialize and start the adaptive downloader
        downloader = AdaptiveDownloader(
            url=download_format['url'],
            output_file=output_file,
            file_size=download_format.get('filesize'),
            initial_connections=request.max_connections,
            max_connections=min(32, request.max_connections * 2),
            progress_callback=progress_handler
        )
        
        async with downloader:
            final_path = await downloader.start()
            
            # Process final file if needed (e.g., convert format)
            if request.convert_format and download_format.get('format_info', {}).get('ext') != request.convert_format:
                # Update status
                active_downloads[download_id]["status"] = "converting"
                
                # Conversion will be implemented here
                # For now, just rename the file
                final_path = output_file
        
        # Update download status to completed
        active_downloads[download_id].update({
            "status": "completed",
            "progress": 1.0,
            "file_path": final_path,
            "elapsed": time.time() - active_downloads[download_id]["start_time"]
        })
        
        logger.info(f"Download completed: {video_title}")
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        active_downloads[download_id].update({
            "status": "failed",
            "error": str(e)
        })
        
        # Send error notification to websockets
        update = {
            "type": "download_error",
            "download_id": download_id,
            "error": str(e)
        }
        
        for client in connected_clients:
            asyncio.create_task(process_websocket_progress(client, update))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    connected_clients.add(websocket)
    
    try:
        # Send initial system status
        await websocket.send_json({
            "type": "system_status",
            "data": get_system_status().dict()
        })
        
        # Send current downloads
        for download_id, download_info in active_downloads.items():
            await websocket.send_json({
                "type": "download_info",
                "download_id": download_id,
                "data": download_info
            })
        
        # Keep connection alive and handle client messages
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                
                # Handle client requests
                if message.get("type") == "get_download_status" and message.get("download_id"):
                    download_id = message["download_id"]
                    if download_id in active_downloads:
                        await websocket.send_json({
                            "type": "download_info",
                            "download_id": download_id,
                            "data": active_downloads[download_id]
                        })
                
                elif message.get("type") == "get_system_status":
                    await websocket.send_json({
                        "type": "system_status",
                        "data": get_system_status().dict()
                    })
                    
            except Exception as e:
                logger.error(f"Error processing websocket message: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error processing request: {str(e)}"
                })
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)

@app.post("/download", response_model=DownloadResponse)
async def start_download(request: DownloadRequest, background_tasks: BackgroundTasks):
    """Initialize a new download with the advanced downloader"""
    download_id = f"dl_{uuid.uuid4().hex}"
    
    # Initialize download info
    active_downloads[download_id] = {
        "download_id": download_id,
        "status": "queued",
        "progress": 0,
        "downloaded_bytes": 0,
        "total_bytes": 0,
        "speed": "0 MB/s",
        "eta": "calculating...",
        "request": request.dict()
    }
    
    try:
        # Get video info
        video_info = await extract_video_info(request.url)
        
        if not video_info:
            raise Exception("Failed to extract video information")
            
        # Get available formats
        formats_available = [
            {
                "format_id": fmt.get("format_id"),
                "ext": fmt.get("ext"),
                "resolution": f"{fmt.get('width', 0)}x{fmt.get('height', 0)}",
                "filesize": fmt.get("filesize"),
                "vcodec": fmt.get("vcodec"),
                "acodec": fmt.get("acodec"),
                "fps": fmt.get("fps"),
                "tbr": fmt.get("tbr", 0)
            }
            for fmt in video_info.get("formats", [])
            if isinstance(fmt, dict)
        ]
        
        # Update download info
        active_downloads[download_id].update({
            "formats_available": formats_available,
            "video_info": {
                "title": video_info.get("title", ""),
                "channel": video_info.get("channel", ""),
                "duration": video_info.get("duration"),
                "upload_date": video_info.get("upload_date"),
                "thumbnail": video_info.get("thumbnail"),
                "webpage_url": video_info.get("webpage_url")
            }
        })
        
        # Start download process in background
        background_tasks.add_task(process_download, download_id, request)
        
        # Create response
        return DownloadResponse(
            download_id=download_id,
            estimated_size=video_info.get("filesize"),
            formats_available=formats_available,
            video_info=active_downloads[download_id]["video_info"],
            status="queued"
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize download: {str(e)}")
        active_downloads[download_id]["status"] = "failed"
        active_downloads[download_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{download_id}", response_model=DownloadStatus)
async def get_download_status(download_id: str):
    """Get the status of a download"""
    if download_id not in active_downloads:
        raise HTTPException(status_code=404, detail="Download not found")
    
    download_info = active_downloads[download_id]
    
    # Calculate elapsed time if download is active
    elapsed = None
    if "start_time" in download_info:
        elapsed = time.time() - download_info["start_time"]
    
    return DownloadStatus(
        download_id=download_id,
        status=download_info["status"],
        progress=download_info.get("progress", 0),
        downloaded_bytes=download_info.get("downloaded_bytes", 0),
        total_bytes=download_info.get("estimated_size", 0),
        speed=download_info.get("speed"),
        eta=download_info.get("eta"),
        file_path=download_info.get("file_path"),
        active_connections=download_info.get("active_connections"),
        format_details=download_info.get("format_details"),
        start_time=download_info.get("start_time"),
        elapsed=elapsed
    )

@app.get("/formats", response_model=List[Dict[str, Any]])
async def get_available_formats(url: str = Query(..., description="YouTube URL")):
    """Get available formats for a video without starting a download"""
    try:
        video_info = await extract_video_info(url)
        
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
                "tbr": fmt.get("tbr"),
                "format_note": fmt.get("format_note", ""),
                "quality": fmt.get("quality", 0),
                "quality_label": fmt.get("quality_label", "")
            }
            for fmt in video_info.get("formats", [])
            if fmt.get("vcodec") != "none" or fmt.get("acodec") != "none"
        ]
        
        # Also return basic video info
        video_details = {
            "title": video_info.get("title", ""),
            "channel": video_info.get("channel", ""),
            "duration": video_info.get("duration"),
            "upload_date": video_info.get("upload_date"),
            "thumbnail": video_info.get("thumbnail"),
            "description": video_info.get("description", "")[:500]  # Truncate long descriptions
        }
        
        return formats
    except Exception as e:
        logger.error(f"Failed to get formats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/download/{download_id}")
async def cancel_download(download_id: str):
    """Cancel an ongoing download"""
    if download_id not in active_downloads:
        raise HTTPException(status_code=404, detail="Download not found")
    
    # Only cancel active downloads
    if active_downloads[download_id]["status"] in ["queued", "processing", "downloading"]:
        active_downloads[download_id]["status"] = "cancelled"
        
        # Notify websocket clients
        update = {
            "type": "download_cancelled",
            "download_id": download_id
        }
        
        for client in connected_clients:
            asyncio.create_task(process_websocket_progress(client, update))
    
    return {"status": "cancelled", "download_id": download_id}

@app.get("/download/{download_id}/file")
async def download_file(download_id: str):
    """Download a completed file"""
    if download_id not in active_downloads:
        raise HTTPException(status_code=404, detail="Download not found")
    
    download_info = active_downloads[download_id]
    
    if download_info["status"] != "completed" or not download_info.get("file_path"):
        raise HTTPException(status_code=400, detail="Download not completed")
    
    file_path = download_info["file_path"]
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),
        media_type="application/octet-stream"
    )

@app.get("/system")
async def get_system_info():
    """Get system status information"""
    return get_system_status()

@app.on_event("startup")
async def startup_event():
    """Initialize app on startup"""
    logger.info("Starting YouTube Downloader API")
    
    # Ensure downloads directory exists
    os.makedirs(downloads_dir, exist_ok=True)
    
    # Start system status broadcasting task
    asyncio.create_task(broadcast_system_status())

async def broadcast_system_status():
    """Periodically broadcast system status to all connected websocket clients"""
    while True:
        try:
            if connected_clients:
                status = get_system_status()
                update = {
                    "type": "system_status",
                    "data": status.dict()
                }
                
                for client in connected_clients:
                    asyncio.create_task(process_websocket_progress(client, update))
        except Exception as e:
            logger.error(f"Error broadcasting system status: {str(e)}")
        
        await asyncio.sleep(5)  # Update every 5 seconds

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
