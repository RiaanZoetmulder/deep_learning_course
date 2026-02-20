# Helper function to display images and videos from the media folder
from IPython.display import display, Image as IPyImage, HTML
import base64
import os 


def show_figure(filename: str, width: int = 900, media_root: str = "media") -> None:
    """
    Display a figure from the media folder.
    
    Args:
        filename: Name of the image file
        width: Display width in pixels
        media_root: Root directory containing the media files
    """
    filepath = os.path.join(media_root, filename)
    if os.path.exists(filepath):
        display(IPyImage(filename=filepath, width=width))
    else:
        print(f"Image not found: {filepath}")
        print(f"Run the media download cell first, or check the filename.")


def show_video(filename: str, width: int = 900, media_root: str = "media") -> None:
    """
    Display a video from the media folder using HTML5 video element.
    
    Args:
        filename: Name of the video file
        width: Display width in pixels
        media_root: Root directory containing the media files
    """
    filepath = os.path.join(media_root, filename)
    if os.path.exists(filepath):
        # Read video file and encode as base64 for inline display
        with open(filepath, "rb") as f:
            video_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Determine video type from extension
        ext = filename.split(".")[-1].lower()
        mime_type = {
            "mp4": "video/mp4",
            "webm": "video/webm",
            "ogg": "video/ogg",
        }.get(ext, "video/mp4")
        
        # Create HTML5 video element
        html = f'''
        <video width="{width}" controls>
            <source src="data:{mime_type};base64,{video_data}" type="{mime_type}">
            Your browser does not support the video tag.
        </video>
        '''
        display(HTML(html))
    else:
        print(f"Video not found: {filepath}")
        print(f"Run the media download cell first, or check the filename.")
