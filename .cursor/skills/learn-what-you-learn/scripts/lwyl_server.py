#!/usr/bin/env python3
"""
Local Notion bridge server for learn-what-you-learn.
Runs on localhost:7842. Start once, leave it running.
"""

import os
import json
import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

from notion_client import Client

NOTION_API_KEY = os.environ.get("NOTION_API_KEY")
NOTION_PARENT_ID = os.environ.get("NOTION_LEARNING_PAGE_ID")

if not NOTION_API_KEY or not NOTION_PARENT_ID:
    raise SystemExit("Missing NOTION_API_KEY or NOTION_LEARNING_PAGE_ID env vars.")

notion = Client(auth=NOTION_API_KEY)

# Holds the current session page ID in memory
state = {"page_id": None, "page_title": None}


def create_page(title: str) -> str:
    date_str = datetime.date.today().isoformat()
    page_title = f"{date_str} — {title}"
    page = notion.pages.create(
        parent={"page_id": NOTION_PARENT_ID},
        properties={"title": {"title": [{"text": {"content": page_title}}]}},
        children=[{
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": [{"type": "text", "text": {"content": "Learning session notes"}}],
                "icon": {"emoji": "📚"},
            },
        }],
    )
    return page["id"], page_title


def append_note(page_id: str, note: str, label: str):
    notion.blocks.children.append(
        block_id=page_id,
        children=[{
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"type": "text", "text": {"content": f"[{label}] {note}"}}]
            },
        }],
    )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress request logs

    def send_json(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        # POST /init  {"title": "..."}
        if self.path == "/init":
            title = body.get("title", "Learning Session")
            page_id, page_title = create_page(title)
            state["page_id"] = page_id
            state["page_title"] = page_title
            self.send_json(200, {"ok": True, "page_title": page_title})

        # POST /save  {"note": "...", "label": "Idea"}
        elif self.path == "/save":
            if not state["page_id"]:
                self.send_json(400, {"ok": False, "error": "No active session. Call /init first."})
                return
            note = body.get("note", "")
            label = body.get("label", "Note")
            append_note(state["page_id"], note, label)
            self.send_json(200, {"ok": True})

        # POST /status
        elif self.path == "/status":
            self.send_json(200, {
                "ok": True,
                "active": state["page_id"] is not None,
                "page_title": state["page_title"],
            })

        else:
            self.send_json(404, {"ok": False, "error": "Unknown endpoint"})


if __name__ == "__main__":
    port = 7842
    print(f"learn-what-you-learn Notion bridge running on http://localhost:{port}")
    print("Keep this terminal open while using the skill in Claude Desktop.")
    HTTPServer(("localhost", port), Handler).serve_forever()
