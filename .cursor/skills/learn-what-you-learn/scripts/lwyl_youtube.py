#!/usr/bin/env python3
"""Fetch YouTube transcript and print to stdout."""
import sys
import re

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    sys.exit("Missing: pip install youtube-transcript-api")

url = " ".join(sys.argv[1:]).strip()
match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
if not match:
    sys.exit(f"Could not extract video ID from: {url}")

video_id = match.group(1)
try:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
except Exception as e:
    sys.exit(f"Could not fetch transcript: {e}")

text = " ".join(entry["text"] for entry in transcript)
print(text)
