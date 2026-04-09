import os
import requests
import json
from datetime import datetime
from langchain_ollama import OllamaLLM

class ZenithAgent:
    """
    Zenith: The Grand Strategist.
    Focuses on macro-economic research, global supply chains, and fundamental catalysts using Perplexity.
    Synthesizes massive data into Obsidian-ready markdown using Local LLM (Ollama).
    """
    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY not found in environment. Zenith remains dormant.")
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "sonar-pro"
        
        # Initialize Local LLM connection
        self.local_llm = OllamaLLM(
            base_url="http://localhost:11434",
            model="qwen3.5:27b",
            temperature=0.2
        )

    def get_system_prompt(self, ticker: str) -> str:
        return f"""You are 'Zenith', the Grand Strategist and Lead Macro Researcher for Google Antigravity.
Your mission is to provide an elite, hedge-fund-level macro analysis on the ticker '{ticker}'.
Your analysis MUST focus on:
1. Global Supply Chain Impact: How global events affect this exact company's logistics, manufacturing, and distribution.
2. Long-term Fundamental Catalysts: Earnings growth, new market expansions, or visionary technological leaps.
3. Macroeconomic Headwinds/Tailwinds: Interest rates, inflation, or geopolitical shifts directly impacting '{ticker}'.

Deliver your findings as a concise, structured intelligence briefing. No fluff. Be decisive. Provide actionable insights.
"""

    def generate_macro_report(self, ticker: str) -> str:
        # Step 1: Fetch raw macro data from Perplexity (Cloud)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.get_system_prompt(ticker)},
                {"role": "user", "content": f"Provide the latest macro and fundamental intelligence briefing on {ticker}."}
            ],
            "max_tokens": 1500,
            "temperature": 0.2
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            raw_macro = data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            return f"Zenith Error: Failed to gather macro data for {ticker}. Exception: {str(e)}"

        # Step 2: Synthesize heavy context via Local LLM (Ollama)
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        ollama_prompt = f"""You are Zenith, the Grand Strategist.
I have raw macro research on {ticker} from an external source. Your job is to synthesize this into a polished, comprehensive research report.
Your output MUST be formatted as an Obsidian-ready Markdown string.
It MUST include the following exact YAML frontmatter at the very top:
---
tags: [macro, Zenith, {ticker}]
date: {current_date}
---

Here is the raw research:
{raw_macro}

Output the fully formatted Markdown report now. Do not include introductory conversational text.
"""
        try:
            obsidian_report = self.local_llm.invoke(ollama_prompt)
            return obsidian_report
        except Exception as e:
            # Fallback if local LLM fails
            fallback_report = f"---\ntags: [macro, Zenith, {ticker}, fallback]\ndate: {current_date}\n---\n\n# Fallback Raw Report\nLocal LLM failed to synthesize. Error: {str(e)}\n\n{raw_macro}"
            return fallback_report
