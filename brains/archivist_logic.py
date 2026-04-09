import os
from datetime import datetime
from langchain_ollama import OllamaLLM

class ArchivistAgent:
    """
    Archivist: The Data Scribe.
    Responsible for large-context local AI inference, Markdown synthesis, and persisting to data lake.
    """
    def __init__(self):
        # Initialize Local LLM connection targeting GPU
        self.local_llm = OllamaLLM(
            base_url="http://localhost:11434",
            model="qwen3.5:27b",
            temperature=0.2,
            num_gpu=99
        )
        
    def synthesize_to_obsidian(self, ticker: str, raw_text: str) -> str:
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        ollama_prompt = f"""You are the Archivist, a meticulous data scientist.
I have raw data on {ticker} from an external source. Your job is to synthesize this into a polished, comprehensive research report.
Your output MUST be formatted as an Obsidian-ready Markdown string.
It MUST include the following exact YAML frontmatter at the very top:
---
tags: [macro, {ticker}]
date: {current_date}
---

Here is the raw research:
{raw_text}

Output the fully formatted Markdown report now. Do not include introductory conversational text.
"""
        try:
            obsidian_report = self.local_llm.invoke(ollama_prompt)
        except Exception as e:
            obsidian_report = f"---\ntags: [macro, {ticker}, fallback]\ndate: {current_date}\n---\n\n# Fallback Raw Report\nLocal LLM failed to synthesize. Error: {str(e)}\n\n{raw_text}"
            
        # Execute File I/O
        self._save_to_data_lake(ticker, current_date, obsidian_report)
        
        return obsidian_report

    def _save_to_data_lake(self, ticker: str, current_date: str, content: str):
        # We are inside 'brains' dir. data_lake is in root dir.
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        obsidian_dir = os.path.join(root_dir, 'data_lake', 'obsidian_sync')
        os.makedirs(obsidian_dir, exist_ok=True)
        
        file_name = f"{ticker}_{current_date}_research.md"
        file_path = os.path.join(obsidian_dir, file_name)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
