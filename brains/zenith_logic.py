import os
import requests
import json

class ZenithAgent:
    """
    Zenith: The Grand Strategist.
    Focuses on macro-economic research, global supply chains, and fundamental catalysts using Perplexity.
    """
    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY not found in environment. Zenith remains dormant.")
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "sonar-pro"

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
        # Fetch raw macro data from Perplexity (Cloud)
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
            return raw_macro
        except requests.exceptions.RequestException as e:
            return f"Zenith Error: Failed to gather macro data for {ticker}. Exception: {str(e)}"
