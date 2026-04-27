#!/usr/bin/env python3
"""Create a new Notion page for this learning session and store its ID."""
import os
import sys
import datetime

try:
    from notion_client import Client
except ImportError:
    sys.exit("Missing: pip install notion-client")

title = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else "Learning Session"
date_str = datetime.date.today().isoformat()
page_title = f"{date_str} — {title}"

notion = Client(auth=os.environ["NOTION_API_KEY"])

page = notion.pages.create(
    parent={"page_id": os.environ["NOTION_LEARNING_PAGE_ID"]},
    properties={
        "title": {"title": [{"text": {"content": page_title}}]}
    },
    children=[
        {
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": [{"type": "text", "text": {"content": "Learning session notes"}}],
                "icon": {"emoji": "📚"},
            },
        }
    ],
)

# Store page ID for this session
with open("/tmp/lwyl_session.txt", "w") as f:
    f.write(page["id"])

print(f"Session started: {page_title}")
