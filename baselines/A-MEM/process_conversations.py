#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
process_conversations.py
处理对话数据，创建记忆系统 - 支持混合检索器
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
    """对话处理器 - 支持混合检索器"""
    
    def __init__(self, 
                 dialogue_path: Path,
                 llm_model: str = "gpt-4o-mini",
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 retriever_type: str = "hybrid",
                 hybrid_alpha: float = 0.5):
        
        self.dialogue_path = Path(dialogue_path)
        self.dialogue_name = self.dialogue_path.name
        self.scenes_dir = self.dialogue_path / "scenes"
        
        # 创建记忆系统
        self.memory_system = AgenticMemorySystem(
            dialogue_name=self.dialogue_name,
            embedding_model_name=embedding_model_name,
            llm_model=llm_model,
            evo_threshold=evo_threshold,
            api_key=api_key,
            base_url=base_url,
            retriever_type=retriever_type,
            hybrid_alpha=hybrid_alpha
        )
        
        # 统计信息
        self.stats = {
            "total_sessions": 0,
            "total_dialogues": 0,
            "processed_dialogues": 0,
            "evolution_count": 0,
            "sessions": {}
        }
        
        # 时间统计
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
        
        logger.info(f"初始化对话处理器: {self.dialogue_name}")
        logger.info(f"  路径: {self.dialogue_path}")
        logger.info(f"  检索器类型: {retriever_type}")
        if retriever_type == "hybrid":
            logger.info(f"  混合权重 alpha: {hybrid_alpha}")
        logger.info(f"  进化阈值: {evo_threshold}")
    
    def scan_sessions(self) -> List[Path]:
        """扫描所有session目录"""
        start_time = time.time()
        
        if not self.scenes_dir.exists():
            logger.error(f"scenes目录不存在: {self.scenes_dir}")
            return []
        
        session_dirs = [
            d for d in self.scenes_dir.iterdir() 
            if d.is_dir() and (d / "conversation.json").exists()
        ]
        
        session_dirs = natsorted(session_dirs, key=lambda x: x.name)
        
        logger.info(f"找到 {len(session_dirs)} 个session目录:")
        for d in session_dirs:
            logger.info(f"  - {d.name}")
        
        self.stats["total_sessions"] = len(session_dirs)
        
        self.timing_stats["scan_sessions_time"] = time.time() - start_time
        logger.debug(f"扫描sessions耗时: {self.timing_stats['scan_sessions_time']:.2f}秒")
        
        return session_dirs
    
    def load_session_data(self, session_dir: Path) -> Optional[Dict]:
        """加载session数据"""
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
            logger.error(f"加载 {conv_file} 失败: {e}")
            return None
    
    def process_dialogue(self, 
                        dialogue: Dict,
                        session_id: str,
                        session_timestamp: str,
                        dialogue_index: int,
                        session_dir: Path) -> tuple[bool, float]:
        """处理单条对话，返回(是否成功, 耗时)"""
        start_time = time.time()
        
        try:
            role = dialogue.get("role", "")
            content = dialogue.get("content", {})
            
            text = session_timestamp + ":" + content.get("text", "")
            image = content.get("image", "")
            caption_text = ""
            
            # 如果有图片，提取image中的数字
            if image:
                caption_json = Path(image).stem + ".json"
                image_number = image_number.group(1) if image_number else None
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
                logger.info(f"  已处理 {self.stats['processed_dialogues']} 条对话 (最近10条平均: {avg_time:.3f}秒/条)")
            
            return True, process_time
            
        except Exception as e:
            logger.error(f"处理对话失败: {e}")
            return False, time.time() - start_time
    
    def process_session(self, session_dir: Path) -> Dict:
        """处理单个session"""
        session_start_time = time.time()
        
        session_data = self.load_session_data(session_dir)
        if not session_data:
            return {"status": "failed", "error": "加载失败"}
        
        session_id = session_data["session_id"]
        dialogues = session_data["dialogues"]
        session_timestamp = session_data["timeline_date"]
        
        logger.info(f"处理session {session_id} ({len(dialogues)} 条对话)")
        
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
        
        logger.info(f"  session完成: {success_count}/{len(dialogues)} 条对话")
        logger.info(f"  耗时: {self._format_time(session_total_time)} (平均: {session_total_time/len(dialogues):.3f}秒/条)")
        
        return {
            "status": "success",
            "session_id": session_id,
            "processed": success_count,
            "total": len(dialogues),
            "processing_time": session_total_time
        }
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.2f}秒"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}分{secs:.2f}秒"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}小时{minutes}分{secs:.2f}秒"
    
    def process_all(self) -> bool:
        """处理所有session"""
        self.timing_stats["start_time"] = datetime.now()
        overall_start = time.time()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"开始处理对话: {self.dialogue_name}")
        logger.info(f"开始时间: {self.timing_stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*60}")
        
        # 扫描sessions
        scan_start = time.time()
        session_dirs = self.scan_sessions()
        self.timing_stats["scan_sessions_time"] = time.time() - scan_start
        
        if not session_dirs:
            logger.error("没有找到任何session")
            return False
        
        logger.info(f"\n步骤1: 按顺序处理 {len(session_dirs)} 个session")
        for i, session_dir in enumerate(session_dirs, 1):
            logger.info(f"\n[{i}/{len(session_dirs)}] 处理 {session_dir.name}")
            result = self.process_session(session_dir)
            
            if result["status"] == "success":
                logger.info(f"  ✅ 完成: {result['processed']}/{result['total']} 条对话")
            else:
                logger.warning(f"  ⚠️ 处理失败: {result.get('error', '未知错误')}")
        
        # 保存记忆系统
        save_start = time.time()
        memory_stats = self.memory_system.get_statistics()
        self.stats["evolution_count"] = memory_stats.get("evolution_count", 0)
        self.timing_stats["save_memory_time"] = time.time() - save_start
        
        # 计算总时间
        overall_time = time.time() - overall_start
        self.timing_stats["total_duration_seconds"] = overall_time
        self.timing_stats["total_duration_formatted"] = self._format_time(overall_time)
        self.timing_stats["end_time"] = datetime.now()
        self.timing_stats["process_dialogues_time"] = self.timing_stats["process_dialogue_time"]
        
        # 输出处理统计
        logger.info(f"\n{'='*60}")
        logger.info(f"处理完成: {self.dialogue_name}")
        logger.info(f"完成时间: {self.timing_stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"总耗时: {self.timing_stats['total_duration_formatted']}")
        logger.info(f"  总session数: {self.stats['total_sessions']}")
        logger.info(f"  总对话数: {self.stats['total_dialogues']}")
        logger.info(f"  成功处理: {self.stats['processed_dialogues']}")
        logger.info(f"  记忆总数: {memory_stats['total_memories']}")
        logger.info(f"  进化次数: {self.stats['evolution_count']}")
        
        # 显示时间统计详情
        logger.info(f"\n时间统计详情:")
        logger.info(f"  扫描sessions: {self._format_time(self.timing_stats['scan_sessions_time'])}")
        logger.info(f"  加载数据: {self._format_time(self.timing_stats['load_session_time'])}")
        logger.info(f"  处理对话: {self._format_time(self.timing_stats['process_dialogues_time'])}")
        logger.info(f"  保存记忆: {self._format_time(self.timing_stats['save_memory_time'])}")
        
        if self.timing_stats["per_dialogue_times"]:
            avg_time = sum(self.timing_stats["per_dialogue_times"]) / len(self.timing_stats["per_dialogue_times"])
            min_time = min(self.timing_stats["per_dialogue_times"])
            max_time = max(self.timing_stats["per_dialogue_times"])
            logger.info(f"\n对话处理统计:")
            logger.info(f"  平均耗时/对话: {avg_time:.3f}秒")
            logger.info(f"  最快对话: {min_time:.3f}秒")
            logger.info(f"  最慢对话: {max_time:.3f}秒")
        
        # 显示检索器统计
        retriever_stats = memory_stats.get('retriever', {})
        logger.info(f"\n检索器信息:")
        logger.info(f"  检索器类型: {retriever_stats.get('retriever_type', 'unknown')}")
        if 'alpha' in retriever_stats:
            logger.info(f"  混合权重 alpha: {retriever_stats['alpha']}")
        logger.info(f"{'='*60}")
        
        return True
    
    def save(self):
        """保存记忆系统"""
        save_start = time.time()
        
        memory_dir = self.dialogue_path / "memory_data"
        self.memory_system.save(memory_dir)
        
        # 添加时间统计到统计文件
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
        logger.info(f"统计数据已保存: {stats_file}")
        logger.info(f"保存耗时: {self._format_time(save_time)}")


def main():
    parser = argparse.ArgumentParser(description="处理对话数据并创建记忆系统")
    
    parser.add_argument("--base_dir", type=str, required=True,
                       help="基础目录，包含所有对话文件夹")
    
    parser.add_argument("--dialogue", type=str, required=True,
                       help="指定要处理的对话名称（可选）")
    
    parser.add_argument("--llm_model", type=str, required=True,
                       help="LLM模型名称")
    
    parser.add_argument("--embedding_model_name", type=str, required=True,
                       help="嵌入模型名称")
    
    # API 密钥改为可选，但建议从环境变量读取；不设硬编码默认值
    parser.add_argument("--api_key", type=str,required=True,
                       help="OpenAI API密钥(也可通过环境变量 OPENAI_API_KEY 设置)")
    
    parser.add_argument("--evo_threshold", type=int, default=100,
                       help="触发记忆合并的进化次数阈值")
    
    # 检索器配置
    parser.add_argument("--retriever_type", type=str, default="hybrid",
                       choices=["simple", "hybrid"],
                       help="检索器类型: simple(仅语义) 或 hybrid(混合检索)")
    parser.add_argument("--hybrid_alpha", type=float, default=0.5,
                       help="混合检索权重 (0=仅BM25, 1=仅语义)")
    
    parser.add_argument("--verbose", action="store_true",
                       help="详细日志输出")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 处理 API 密钥：优先使用命令行参数，否则从环境变量读取
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("未提供 API 密钥")
    
    print("\n" + "="*70)
    print("对话记忆处理工具 - 支持混合检索")
    print("="*70)
    print(f"基础目录: {args.base_dir}")
    print(f"嵌入模型: {args.embedding_model_name}")
    print(f"检索器类型: {args.retriever_type}")
    if args.retriever_type == "hybrid":
        print(f"混合权重 alpha: {args.hybrid_alpha}")
    print(f"进化阈值: {args.evo_threshold}")
    if args.dialogue:
        print(f"指定对话: {args.dialogue}")
    print("="*70 + "\n")
    
    base_path = Path(args.base_dir)
    if not base_path.exists():
        print(f"❌ 基础目录不存在: {base_path}")
        return 1
    
    # 处理单个对话
    if args.dialogue:
        dialogue_path = base_path / args.dialogue
        if not dialogue_path.exists():
            print(f"❌ 对话目录不存在: {dialogue_path}")
            return 1
        
        processor = ConversationProcessor(
            dialogue_path=dialogue_path,
            llm_model=args.llm_model,
            embedding_model_name=args.embedding_model_name,
            evo_threshold=args.evo_threshold,
            api_key=api_key,
            retriever_type=args.retriever_type,
            hybrid_alpha=args.hybrid_alpha
        )
        
        if processor.process_all():
            processor.save()
            print(f"\n✅ 对话 {args.dialogue} 处理成功!")
            print(f"⏱️  总耗时: {processor.timing_stats['total_duration_formatted']}")
            return 0
        else:
            print(f"\n❌ 对话 {args.dialogue} 处理失败")
            return 1
    
    else:
        dialogue_dirs = [
            d for d in base_path.iterdir() 
            if d.is_dir() and d.name.startswith("dialogue")
        ]
        dialogue_dirs = natsorted(dialogue_dirs)
        
        if not dialogue_dirs:
            print("❌ 未找到任何以“dialogue”开头的文件夹")
            return 1
        
        print(f"找到 {len(dialogue_dirs)} 个对话文件夹:")
        for d in dialogue_dirs:
            print(f"  - {d.name}")
        print()
        
        results = {}
        success_count = 0
        all_timing_stats = {}
        
        for dialogue_dir in dialogue_dirs:
            print(f"\n{'#'*70}")
            print(f"处理: {dialogue_dir.name}")
            print(f"{'#'*70}")
            
            try:
                processor = ConversationProcessor(
                    dialogue_path=dialogue_dir,
                    llm_model=args.llm_model,
                    embedding_model_name=args.embedding_model_name,
                    evo_threshold=args.evo_threshold,
                    api_key=api_key,
                    retriever_type=args.retriever_type,
                    hybrid_alpha=args.hybrid_alpha
                )
                
                if processor.process_all():
                    processor.save()
                    results[dialogue_dir.name] = True
                    success_count += 1
                    all_timing_stats[dialogue_dir.name] = processor.timing_stats
                    print(f"\n✅ {dialogue_dir.name} 处理成功")
                    print(f"⏱️  耗时: {processor.timing_stats['total_duration_formatted']}")
                else:
                    results[dialogue_dir.name] = False
                    print(f"\n❌ {dialogue_dir.name} 处理失败")
                    
            except Exception as e:
                logger.error(f"处理 {dialogue_dir.name} 出错: {e}")
                results[dialogue_dir.name] = False
        
        print("\n" + "="*70)
        print("处理结果汇总")
        print("="*70)
        for name, success in results.items():
            status = "✅ 成功" if success else "❌ 失败"
            time_info = f" ({all_timing_stats[name]['total_duration_formatted']})" if name in all_timing_stats else ""
            print(f"{status}: {name}{time_info}")
        print("-"*70)
        print(f"总计: {len(results)} 个对话，成功: {success_count} 个")
        
        if all_timing_stats:
            total_time = sum(stat['total_duration_seconds'] for stat in all_timing_stats.values())
            if 'processor' in locals():
                print(f"总处理时间: {processor._format_time(total_time)}")
            else:
                print(f"总处理时间: {total_time:.2f}秒")
        print("="*70)
        
        return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)