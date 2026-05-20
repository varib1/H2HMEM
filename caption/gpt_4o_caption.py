#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPT-4o Batch Image Description Tool (with token limit)
Uses GPT-4o to generate image descriptions with automatic truncation to specified token limit.
"""

import os
import json
import argparse
import base64
import re
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
from openai import OpenAI
import tiktoken


class GPT4oImageDescriber:
    """Batch image describer using GPT-4o with token limit and structured output"""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self._test_connection()

    def _test_connection(self) -> bool:
        try:
            models = self.client.models.list()
            print(f"API connection successful, available models: {len(models.data)}")
            return True
        except Exception as e:
            print(f"API connection failed: {e}")
            return False

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated = self.tokenizer.decode(tokens[:max_tokens])
        return truncated + "... [truncated]"

    def describe_image(self, image_path: str, caption_max_tokens: int = 256,
                       prompt: Optional[str] = None,
                       max_tokens: int = 1200, detail: str = "auto") -> Dict[str, Any]:
        img_path = Path(image_path)
        if not img_path.exists():
            return self._error_result(f"Image file not found: {image_path}", image_path)

        with open(img_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')

        ext = img_path.suffix.lower().replace('.', '')
        mime_map = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png',
                    'gif': 'gif', 'webp': 'webp', 'bmp': 'bmp'}
        mime = mime_map.get(ext, 'jpeg')

        if prompt is None:
            prompt = f"""Please describe this image in a concise manner, ensuring the description does not exceed {caption_max_tokens} tokens.

Please include:
- Core scene
- Main elements (objects/persons/animals)
- Visible text (if any)
- Actions or interactions (if any)
- Overall mood or atmosphere

Please provide a clear and informative description based on the image content. If the description exceeds the token limit, it will be truncated."""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{mime};base64,{base64_image}",
                                "detail": detail
                            }
                        }
                    ]
                }
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )

        raw_text = response.choices[0].message.content
        orig_tokens = self.count_tokens(raw_text)

        if orig_tokens > caption_max_tokens:
            final_text = self.truncate_to_token_limit(raw_text, caption_max_tokens)
            was_truncated = True
            final_tokens = caption_max_tokens
        else:
            final_text = raw_text
            was_truncated = False
            final_tokens = orig_tokens

        return {
            "id": str(uuid.uuid4()),
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "image_info": {
                "path": str(img_path.absolute()),
                "filename": img_path.name,
                "size": os.path.getsize(image_path),
                "modified_time": datetime.fromtimestamp(os.path.getmtime(image_path)).isoformat()
            },
            "model_info": {
                "model": "gpt-4o",
                "detail_level": detail,
                "caption_max_tokens": caption_max_tokens
            },
            "description": {
                "full_text": raw_text,
                "final_text": final_text,
                "token_stats": {
                    "original_tokens": orig_tokens,
                    "final_tokens": final_tokens,
                    "was_truncated": was_truncated,
                    "limit": caption_max_tokens
                }
            },
            "raw_response": {
                "completion_id": response.id,
                "created": response.created,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                    "completion_tokens": response.usage.completion_tokens if response.usage else None,
                    "total_tokens": response.usage.total_tokens if response.usage else None
                }
            }
        }

    def _error_result(self, msg: str, path: str) -> Dict:
        return {
            "id": str(uuid.uuid4()),
            "success": False,
            "error": True,
            "error_message": msg,
            "timestamp": datetime.now().isoformat(),
            "image_path": path
        }

    def process_single_session(self, session_path: Path, caption_max_tokens: int = 256) -> Dict:
        image_dir = session_path / "image"
        if not image_dir.exists():
            return {"session": session_path.name, "status": "failed", "error": "No image directory"}

        caption_dir = session_path / "caption"
        caption_dir.mkdir(exist_ok=True)

        image_files = []
        for ext in ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp']:
            image_files.extend(image_dir.glob(f"*.{ext}"))
            image_files.extend(image_dir.glob(f"*.{ext.upper()}"))

        def num_key(p: Path):
            m = re.search(r'\d+', p.stem)
            return int(m.group()) if m else 0
        image_files.sort(key=num_key)

        if not image_files:
            return {"session": session_path.name, "status": "failed", "error": "No image files"}

        result = {
            "session": session_path.name,
            "image_dir": str(image_dir),
            "caption_dir": str(caption_dir),
            "caption_max_tokens": caption_max_tokens,
            "total_images": len(image_files),
            "successful": 0,
            "failed": 0,
            "processed_files": []
        }

        for img in image_files:
            out_json = caption_dir / f"{img.stem}.json"
            out_txt = caption_dir / f"{img.stem}.txt"
            desc = self.describe_image(str(img), caption_max_tokens=caption_max_tokens)

            if desc.get("success"):
                with open(out_json, 'w', encoding='utf-8') as f:
                    json.dump(desc, f, ensure_ascii=False, indent=2)
                with open(out_txt, 'w', encoding='utf-8') as f:
                    f.write(desc["description"]["final_text"])
                result["successful"] += 1
                result["processed_files"].append({
                    "image": img.name,
                    "json_file": out_json.name,
                    "txt_file": out_txt.name,
                    "success": True,
                    "token_stats": desc["description"]["token_stats"]
                })
            else:
                result["failed"] += 1
                result["processed_files"].append({
                    "image": img.name,
                    "success": False,
                    "error": desc.get("error_message", "Unknown")
                })
        return result

    def process_dialogue(self, dialogue_path: Path, caption_max_tokens: int = 256) -> Dict:
        print(f"  📁 Processing {dialogue_path.name}...")
        scenes_dir = dialogue_path / "scenes"
        if not scenes_dir.exists():
            return {"dialogue": dialogue_path.name, "status": "failed", "error": "No scenes directory"}

        session_folders = [d for d in scenes_dir.iterdir() if d.is_dir() and re.match(r'^session\d+$', d.name)]
        session_folders.sort(key=lambda x: int(re.search(r'\d+', x.name).group()))

        if not session_folders:
            return {"dialogue": dialogue_path.name, "status": "failed", "error": "No session folders"}

        result = {
            "dialogue": dialogue_path.name,
            "scenes_dir": str(scenes_dir),
            "caption_max_tokens": caption_max_tokens,
            "total_sessions": len(session_folders),
            "total_images": 0,
            "total_successful": 0,
            "total_failed": 0,
            "total_truncated": 0,
            "session_results": []
        }

        for sess in session_folders:
            print(f"  📁 Processing {dialogue_path.name}:{sess.name}...")
            sess_res = self.process_single_session(sess, caption_max_tokens)
            result["session_results"].append(sess_res)
            if "error" not in sess_res:
                result["total_images"] += sess_res["total_images"]
                result["total_successful"] += sess_res["successful"]
                result["total_failed"] += sess_res["failed"]
                for f in sess_res.get("processed_files", []):
                    if f.get("success") and f.get("token_stats", {}).get("was_truncated"):
                        result["total_truncated"] += 1
            else:
                print(f"    ⚠️ {sess.name} skipped: {sess_res.get('error')}")

        summary_file = dialogue_path / "caption_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result


def main():
    parser = argparse.ArgumentParser(description="GPT-4o Batch Image Description Tool (with token limit)")
    parser.add_argument("--base_path", required=True, help="Root directory containing dialogue folders")
    parser.add_argument("--api_key", required=True, help="OpenAI API key or compatible service key")
    parser.add_argument("--base_url", required=True, help="API base URL")
    parser.add_argument("--caption_max_tokens", type=int, default=256, help="Maximum tokens for description")
    parser.add_argument("--dialogue", help="Process only a single dialogue folder name")
    parser.add_argument("--dialogue_pattern", default="dialogue*", help="Dialogue folder matching pattern")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    base = Path(args.base_path)
    if not base.exists():
        print(f"Error: Base path does not exist: {base}")
        return 1

    describer = GPT4oImageDescriber(api_key=args.api_key, base_url=args.base_url)

    if args.dialogue:
        dialogue_path = base / args.dialogue
        if not dialogue_path.is_dir():
            print(f"Error: Dialogue directory does not exist: {dialogue_path}")
            return 1
        describer.process_dialogue(dialogue_path, args.caption_max_tokens)
    else:
        pattern = args.dialogue_pattern
        if '*' not in pattern:
            pattern = pattern + '*'
        dialogues = sorted(base.glob(pattern))
        if not dialogues:
            print(f"No dialogue folders matching '{args.dialogue_pattern}' found")
            return 1
        for dlg in dialogues:
            describer.process_dialogue(dlg, args.caption_max_tokens)

    print("\n🎉 All tasks completed!")
    return 0


if __name__ == "__main__":
    exit(main())