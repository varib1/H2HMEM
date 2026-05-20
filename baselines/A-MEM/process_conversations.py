#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
process_conversations.py
Process conversation data, create memory system - supports hybrid retriever
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from natsort import natsorted
import sys
import re
import time
sys.path.insert(0, str(Path(__file__).parent))

from memory_system import AgenticMemorySystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process_conversations.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ConversationProcessor:
    """Conversation processor - supports hybrid retriever"""
    
    def __init__(self, 
                 dialogue_path: Path,
                 memoryconstruct_model: str = "gpt-4o-mini",
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 retriever_type: str = "hybrid",
                 hybrid_alpha: float = 0.5):
        
        self.dialogue_path = Path(dialogue_path)
        self.dialogue_name = self.dialogue_path.name
        self.scenes_dir = self.dialogue_path / "scenes"
        
        # Create memory system
        self.memory_system = AgenticMemorySystem(
            dialogue_name=self.dialogue_name,
            embedding_model_name=embedding_model_name,
            memoryconstruct_model=memoryconstruct_model,
            evo_threshold=evo_threshold,
            api_key=api_key,
            base_url=base_url,
            retriever_type=retriever_type,
            hybrid_alpha=hybrid_alpha
        )
        
        # Statistics
        self.stats = {
            "total_sessions": 0,
            "total_dialogues": 0,
            "processed_dialogues": 0,
            "evolution_count": 0,
            "sessions": {}
        }
        
        # Timing statistics
        self.timing_stats = {
            "start_time": None,
            "end_time": None,
            "total_duration_seconds": 0,
            "total_duration_formatted": "",
            "scan_sessions_time": 0,
            "load_session_time": 0,
            "process_dialogues_time": 0,
            "process_dialogue_time": 0,
            "save_memory_time": 0,
            "per_dialogue_times": [],
            "session_times": {}
        }
        
        logger.info(f"Initializing conversation processor: {self.dialogue_name}")
        logger.info(f"  Path: {self.dialogue_path}")
        logger.info(f"  Retriever type: {retriever_type}")
        if retriever_type == "hybrid":
            logger.info(f"  Hybrid weight alpha: {hybrid_alpha}")
        logger.info(f"  Evolution threshold: {evo_threshold}")
    
    def scan_sessions(self) -> List[Path]:
        """Scan all session directories"""
        start_time = time.time()
        
        if not self.scenes_dir.exists():
            logger.error(f"Scenes directory does not exist: {self.scenes_dir}")
            return []
        
        session_dirs = [
            d for d in self.scenes_dir.iterdir() 
            if d.is_dir() and (d / "session.json").exists()
        ]
        
        session_dirs = natsorted(session_dirs, key=lambda x: x.name)
        
        logger.info(f"Found {len(session_dirs)} session directories:")
        for d in session_dirs:
            logger.info(f"  - {d.name}")
        
        self.stats["total_sessions"] = len(session_dirs)
        
        self.timing_stats["scan_sessions_time"] = time.time() - start_time
        logger.debug(f"Session scanning time: {self.timing_stats['scan_sessions_time']:.2f} seconds")
        
        return session_dirs
    
    def load_session_data(self, session_dir: Path) -> Optional[Dict]:
        """Load session data"""
        start_time = time.time()
        
        conv_file = session_dir / "session.json"
        
        try:
            with open(conv_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            session_id = session_dir.name
            session_title = data.get("session_title", "")
            dialogues = data.get("dialogue", [])
            timestamp = data.get("timeline_date", "")
            
            load_time = time.time() - start_time
            self.timing_stats["load_session_time"] += load_time
            
            return {
                "session_id": session_id,
                "session_dir": session_dir,
                "session_title": session_title,
                "timeline_date": timestamp,
                "dialogues": dialogues,
                "dialogue_count": len(dialogues),
                "load_time": load_time
            }
            
        except Exception as e:
            logger.error(f"Failed to load {conv_file}: {e}")
            return None
    
    def process_dialogue(self, 
                        dialogue: Dict,
                        session_id: str,
                        session_timestamp: str,
                        dialogue_index: int,
                        session_dir: Path) -> tuple[bool, float]:
        """Process a single dialogue, returns (success, processing_time)"""
        start_time = time.time()
        
        try:
            role = dialogue.get("role", "")
            content = dialogue.get("content", {})
            
            text = session_timestamp + ":" + content.get("text", "")
            image = content.get("image", "")
            caption_text = ""
            
            # If there is an image, extract the number from the image filename
            if image:
                caption_json = Path(image).stem + ".json"
                image_caption_path = session_dir / "caption" / caption_json
                print(image_caption_path)
                if image_caption_path.exists():
                    with open(image_caption_path, 'r', encoding='utf-8') as f:
                        caption_data = json.load(f)
                    description = caption_data.get("description", "")
                    caption_text = description.get("final_text", "")
            
            full_content = f"[{role}]: {text}"
            if image:
                full_content += f" [image: {image}]" + (f" [image_description: {caption_text}]" if caption_text else "")
            
            dialogue_timestamp = f"{session_timestamp}_{dialogue_index:03d}"

            memory_id = self.memory_system.add_note(
                content=full_content,
                time=dialogue_timestamp,
                dialogue_name=self.dialogue_name,
                session_id=session_id,
                dialogue_index=dialogue_index,
                role=role,
                has_image=bool(image),
                image_filename=image if image else ""
            )
            
            self.stats["processed_dialogues"] += 1
            
            process_time = time.time() - start_time
            self.timing_stats["process_dialogue_time"] += process_time
            self.timing_stats["per_dialogue_times"].append(process_time)
            
            if self.stats["processed_dialogues"] % 10 == 0:
                avg_time = sum(self.timing_stats["per_dialogue_times"][-10:]) / min(10, len(self.timing_stats["per_dialogue_times"]))
                logger.info(f"  Processed {self.stats['processed_dialogues']} dialogues (last 10 avg: {avg_time:.3f}s/dialogue)")
            
            return True, process_time
            
        except Exception as e:
            logger.error(f"Failed to process dialogue: {e}")
            return False, time.time() - start_time
    
    def process_session(self, session_dir: Path) -> Dict:
        """Process a single session"""
        session_start_time = time.time()
        
        session_data = self.load_session_data(session_dir)
        if not session_data:
            return {"status": "failed", "error": "Failed to load"}
        
        session_id = session_data["session_id"]
        dialogues = session_data["dialogues"]
        session_timestamp = session_data["timeline_date"]
        
        logger.info(f"Processing session {session_id} ({len(dialogues)} dialogues)")
        
        success_count = 0
        dialogue_times = []
        
        for idx, dialogue in enumerate(dialogues, 1):
            success, proc_time = self.process_dialogue(dialogue, session_id, session_timestamp, idx, session_data["session_dir"])
            if success:
                success_count += 1
                dialogue_times.append(proc_time)
        
        session_total_time = time.time() - session_start_time
        
        self.stats["sessions"][session_id] = {
            "session_dir": str(session_data["session_dir"]),
            "session_title": session_data["session_title"],
            "total_dialogues": len(dialogues),
            "processed": success_count,
            "success_rate": f"{success_count/len(dialogues)*100:.1f}%" if dialogues else "0%",
            "processing_time_seconds": session_total_time,
            "processing_time_formatted": self._format_time(session_total_time),
            "avg_time_per_dialogue": session_total_time / len(dialogues) if dialogues else 0
        }
        
        self.timing_stats["session_times"][session_id] = {
            "total_time": session_total_time,
            "dialogue_count": len(dialogues),
            "avg_time": session_total_time / len(dialogues) if dialogues else 0
        }
        
        self.stats["total_dialogues"] += len(dialogues)
        
        logger.info(f"  Session complete: {success_count}/{len(dialogues)} dialogues")
        logger.info(f"  Time: {self._format_time(session_total_time)} (avg: {session_total_time/len(dialogues):.3f}s/dialogue)")
        
        return {
            "status": "success",
            "session_id": session_id,
            "processed": success_count,
            "total": len(dialogues),
            "processing_time": session_total_time
        }
    
    def _format_time(self, seconds: float) -> str:
        """Format time for display"""
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes} min {secs:.2f} sec"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours} hr {minutes} min {secs:.2f} sec"
    
    def process_all(self) -> bool:
        """Process all sessions"""
        self.timing_stats["start_time"] = datetime.now()
        overall_start = time.time()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting to process conversation: {self.dialogue_name}")
        logger.info(f"Start time: {self.timing_stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*60}")
        
        # Scan sessions
        scan_start = time.time()
        session_dirs = self.scan_sessions()
        self.timing_stats["scan_sessions_time"] = time.time() - scan_start
        
        if not session_dirs:
            logger.error("No sessions found")
            return False
        
        logger.info(f"\nStep 1: Processing {len(session_dirs)} sessions in order")
        for i, session_dir in enumerate(session_dirs, 1):
            logger.info(f"\n[{i}/{len(session_dirs)}] Processing {session_dir.name}")
            result = self.process_session(session_dir)
            
            if result["status"] == "success":
                logger.info(f"  ✅ Complete: {result['processed']}/{result['total']} dialogues")
            else:
                logger.warning(f"  ⚠️ Processing failed: {result.get('error', 'Unknown error')}")
        
        # Save memory system
        save_start = time.time()
        memory_stats = self.memory_system.get_statistics()
        self.stats["evolution_count"] = memory_stats.get("evolution_count", 0)
        self.timing_stats["save_memory_time"] = time.time() - save_start
        
        # Calculate total time
        overall_time = time.time() - overall_start
        self.timing_stats["total_duration_seconds"] = overall_time
        self.timing_stats["total_duration_formatted"] = self._format_time(overall_time)
        self.timing_stats["end_time"] = datetime.now()
        self.timing_stats["process_dialogues_time"] = self.timing_stats["process_dialogue_time"]
        
        # Output processing statistics
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing complete: {self.dialogue_name}")
        logger.info(f"Completion time: {self.timing_stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total time: {self.timing_stats['total_duration_formatted']}")
        logger.info(f"  Total sessions: {self.stats['total_sessions']}")
        logger.info(f"  Total dialogues: {self.stats['total_dialogues']}")
        logger.info(f"  Successfully processed: {self.stats['processed_dialogues']}")
        logger.info(f"  Total memories: {memory_stats['total_memories']}")
        logger.info(f"  Evolution count: {self.stats['evolution_count']}")
        
        # Display detailed timing statistics
        logger.info(f"\nDetailed timing statistics:")
        logger.info(f"  Scan sessions: {self._format_time(self.timing_stats['scan_sessions_time'])}")
        logger.info(f"  Load data: {self._format_time(self.timing_stats['load_session_time'])}")
        logger.info(f"  Process dialogues: {self._format_time(self.timing_stats['process_dialogues_time'])}")
        logger.info(f"  Save memory: {self._format_time(self.timing_stats['save_memory_time'])}")
        
        if self.timing_stats["per_dialogue_times"]:
            avg_time = sum(self.timing_stats["per_dialogue_times"]) / len(self.timing_stats["per_dialogue_times"])
            min_time = min(self.timing_stats["per_dialogue_times"])
            max_time = max(self.timing_stats["per_dialogue_times"])
            logger.info(f"\nDialogue processing statistics:")
            logger.info(f"  Average time/dialogue: {avg_time:.3f} seconds")
            logger.info(f"  Fastest dialogue: {min_time:.3f} seconds")
            logger.info(f"  Slowest dialogue: {max_time:.3f} seconds")
        
        # Display retriever information
        retriever_stats = memory_stats.get('retriever', {})
        logger.info(f"\nRetriever information:")
        logger.info(f"  Retriever type: {retriever_stats.get('retriever_type', 'unknown')}")
        if 'alpha' in retriever_stats:
            logger.info(f"  Hybrid weight alpha: {retriever_stats['alpha']}")
        logger.info(f"{'='*60}")
        
        return True
    
    def save(self):
        """Save memory system"""
        save_start = time.time()
        
        memory_dir = self.dialogue_path / "memory_data"
        self.memory_system.save(memory_dir)
        
        # Add timing statistics to stats file
        stats_file = memory_dir / "processing_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                "dialogue_name": self.dialogue_name,
                "processing_time": datetime.now().isoformat(),
                "stats": self.stats,
                "memory_stats": self.memory_system.get_statistics(),
                "timing_stats": {
                    "start_time": self.timing_stats["start_time"].isoformat() if self.timing_stats["start_time"] else None,
                    "end_time": self.timing_stats["end_time"].isoformat() if self.timing_stats["end_time"] else None,
                    "total_duration_seconds": self.timing_stats["total_duration_seconds"],
                    "total_duration_formatted": self.timing_stats["total_duration_formatted"],
                    "scan_sessions_time": self.timing_stats["scan_sessions_time"],
                    "load_session_time": self.timing_stats["load_session_time"],
                    "process_dialogues_time": self.timing_stats["process_dialogues_time"],
                    "save_memory_time": self.timing_stats["save_memory_time"],
                    "avg_time_per_dialogue": sum(self.timing_stats["per_dialogue_times"]) / len(self.timing_stats["per_dialogue_times"]) if self.timing_stats["per_dialogue_times"] else 0,
                    "min_dialogue_time": min(self.timing_stats["per_dialogue_times"]) if self.timing_stats["per_dialogue_times"] else 0,
                    "max_dialogue_time": max(self.timing_stats["per_dialogue_times"]) if self.timing_stats["per_dialogue_times"] else 0,
                    "session_times": self.timing_stats["session_times"]
                }
            }, f, ensure_ascii=False, indent=2)
        
        save_time = time.time() - save_start
        logger.info(f"Statistics saved to: {stats_file}")
        logger.info(f"Save time: {self._format_time(save_time)}")


def main():
    parser = argparse.ArgumentParser(description="Process conversation data and create memory system")
    
    parser.add_argument("--base_dir", type=str, required=True,
                       help="Base directory containing all conversation folders")
    
    parser.add_argument("--dialogue", type=str, required=True,
                       help="Specify the dialogue name to process")
    
    parser.add_argument("--memoryconstruct_model", type=str, required=True,
                       help="Model name used for memory construction")
    
    parser.add_argument("--embedding_model_name", type=str, required=True,
                       help="Embedding model name")
    
    # API key is optional but recommended to read from environment variable; no hardcoded default
    parser.add_argument("--api_key", type=str, required=True,
                       help="OpenAI API key (can also be set via environment variable OPENAI_API_KEY)")
    
    parser.add_argument("--base_url", type=str, required=True,
                       help="OpenAI API base URL")
    
    parser.add_argument("--evo_threshold", type=int, default=100,
                       help="Evolution threshold to trigger memory consolidation")
    
    # Retriever configuration
    parser.add_argument("--retriever_type", type=str, default="hybrid",
                       choices=["simple", "hybrid"],
                       help="Retriever type: simple (semantic only) or hybrid (mixed retrieval)")
    parser.add_argument("--hybrid_alpha", type=float, default=0.5,
                       help="Hybrid retrieval weight (0=BM25 only, 1=semantic only)")
    
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose logging output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Process API key: prioritize command line argument, otherwise read from environment variable
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("API key not provided")
    
    print("\n" + "="*70)
    print("Conversation Memory Processing Tool - Supports Hybrid Retrieval")
    print("="*70)
    print(f"Base directory: {args.base_dir}")
    print(f"Embedding model: {args.embedding_model_name}")
    print(f"Retriever type: {args.retriever_type}")
    if args.retriever_type == "hybrid":
        print(f"Hybrid weight alpha: {args.hybrid_alpha}")
    print(f"Evolution threshold: {args.evo_threshold}")
    if args.dialogue:
        print(f"Specified dialogue: {args.dialogue}")
    print("="*70 + "\n")
    
    base_path = Path(args.base_dir)
    if not base_path.exists():
        print(f"❌ Base directory does not exist: {base_path}")
        return 1
    
    # Process single dialogue
    if args.dialogue:
        dialogue_path = base_path / args.dialogue
        if not dialogue_path.exists():
            print(f"❌ Dialogue directory does not exist: {dialogue_path}")
            return 1
        
        processor = ConversationProcessor(
            dialogue_path=dialogue_path,
            memoryconstruct_model=args.memoryconstruct_model,
            embedding_model_name=args.embedding_model_name,
            evo_threshold=args.evo_threshold,
            api_key=api_key,
            retriever_type=args.retriever_type,
            hybrid_alpha=args.hybrid_alpha,
            base_url=args.base_url
        )
        
        if processor.process_all():
            processor.save()
            print(f"\n✅ Dialogue {args.dialogue} processed successfully!")
            print(f"⏱️  Total time: {processor.timing_stats['total_duration_formatted']}")
            return 0
        else:
            print(f"\n❌ Dialogue {args.dialogue} processing failed")
            return 1
    
    else:
        dialogue_dirs = [
            d for d in base_path.iterdir() 
            if d.is_dir() and d.name.startswith("dialogue")
        ]
        dialogue_dirs = natsorted(dialogue_dirs)
        
        if not dialogue_dirs:
            print("❌ No folders starting with 'dialogue' found")
            return 1
        
        print(f"Found {len(dialogue_dirs)} dialogue folders:")
        for d in dialogue_dirs:
            print(f"  - {d.name}")
        print()
        
        results = {}
        success_count = 0
        all_timing_stats = {}
        
        for dialogue_dir in dialogue_dirs:
            print(f"\n{'#'*70}")
            print(f"Processing: {dialogue_dir.name}")
            print(f"{'#'*70}")
            
            try:
                processor = ConversationProcessor(
                    dialogue_path=dialogue_dir,
                    memoryconstruct_model=args.memoryconstruct_model,
                    embedding_model_name=args.embedding_model_name,
                    evo_threshold=args.evo_threshold,
                    api_key=api_key,
                    retriever_type=args.retriever_type,
                    hybrid_alpha=args.hybrid_alpha,
                    base_url=args.base_url
                )
                
                if processor.process_all():
                    processor.save()
                    results[dialogue_dir.name] = True
                    success_count += 1
                    all_timing_stats[dialogue_dir.name] = processor.timing_stats
                    print(f"\n✅ {dialogue_dir.name} processed successfully")
                    print(f"⏱️  Time: {processor.timing_stats['total_duration_formatted']}")
                else:
                    results[dialogue_dir.name] = False
                    print(f"\n❌ {dialogue_dir.name} processing failed")
                    
            except Exception as e:
                logger.error(f"Error processing {dialogue_dir.name}: {e}")
                results[dialogue_dir.name] = False
        
        print("\n" + "="*70)
        print("Processing Results Summary")
        print("="*70)
        for name, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            time_info = f" ({all_timing_stats[name]['total_duration_formatted']})" if name in all_timing_stats else ""
            print(f"{status}: {name}{time_info}")
        print("-"*70)
        print(f"Total: {len(results)} dialogues, successful: {success_count}")
        
        if all_timing_stats:
            total_time = sum(stat['total_duration_seconds'] for stat in all_timing_stats.values())
            if 'processor' in locals():
                print(f"Total processing time: {processor._format_time(total_time)}")
            else:
                print(f"Total processing time: {total_time:.2f} seconds")
        print("="*70)
        
        return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)