#!/usr/bin/env python3
"""Append a note to the current learning session's Notion page."""
import os
import sys

try:
    from notion_client import Client
except ImportError:
    sys.exit("Missing: pip install notion-client")

if len(sys.argv) < 2:
    sys.exit("Usage: lwyl_notion_save.py <note> [label]")

note = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else "Note"

try:
    with open("/tmp/lwyl_session.txt") as f:
        page_id = f.read().strip()
except FileNotFoundError:
    sys.exit("No active session. Start one with /learn-what-you-learn first.")

notion = Client(auth=os.environ["NOTION_API_KEY"])

notion.blocks.children.append(
    block_id=page_id,
    children=[
        {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [
                    {"type": "text", "text": {"content": f"[{label}] {note}"}}
                ]
            },
        }
    ],
)

print("Saved.")
