import os
import yfinance as yf
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

class SigmaAgent:
    """
    Sigma: The Sniper. Tactical Swing Trading Agent.
    Focuses on short-term swing trading, risk management, and quantitative data analysis.
    Executes the "Triple Filter Strategy".
    """
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment. Sigma remains dormant.")
        
        # Initialize Gemini 3 Flash
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=self.api_key,
            temperature=0.1
        )

    def fetch_technical_data(self, ticker: str) -> dict:
        """Fetches last 3 months of daily data and computes key indicators."""
        stock = yf.Ticker(ticker)
        df = stock.history(period="3mo")
        
        if df.empty:
            return {"error": "No historical data found for ticker"}

        # Calculate Indicators
        # 1. Simple Moving Averages
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        
        # 2. Average True Range (ATR)
        df['High-Low'] = df['High'] - df['Low']
        df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
        df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
        df['ATR14'] = df['TR'].rolling(window=14).mean()
        
        # 3. Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI14'] = 100 - (100 / (1 + rs))

        # Output the most recent valid row
        latest = df.iloc[-1]
        
        return {
            "current_price": round(latest['Close'], 2),
            "sma_20": round(latest['SMA20'], 2) if not pd.isna(latest['SMA20']) else None,
            "sma_50": round(latest['SMA50'], 2) if not pd.isna(latest['SMA50']) else None,
            "atr_14": round(latest['ATR14'], 2) if not pd.isna(latest['ATR14']) else None,
            "rsi_14": round(latest['RSI14'], 2) if not pd.isna(latest['RSI14']) else None,
            "volume": int(latest['Volume'])
        }

    def get_system_prompt(self, ticker: str) -> str:
        return f"""You are 'Sigma', a ruthless, disciplined tactical agent and Lead Quant Engineer for Google Antigravity.
Your expertise lies in short-term swing trading, flawless risk management, and purely objective technical analysis.

You strictly adhere to the "Triple Filter Strategy":
1. Market/Trend Filter: Is the broader trend (e.g., SMA20 vs SMA50) supporting the trade direction?
2. Momentum Filter: Are oscillators (like RSI) showing overbought/oversold extremes but turning favorably?
3. Entry/Risk Filter: What is the optimal entry price? What is the strict stop-loss defined by volatility metrics like ATR?

Your Task:
Analyze the provided Macro Report (from Zenith) AND the Raw Technical Data.
You must synthesize both data points to make the FINAL Swing Trade Decision for '{ticker}'.

Deliver your output with military precision. Provide:
- Verdict: [LONG / SHORT / CASH (Do nothing)]
- Entry Price: [Specific Target]
- Stop Loss: [Specific Target based on ATR]
- Take Profit: [Specific Target based on Risk/Reward]
- Rationale: [Brief bullet points combining Macro & Technical justification]
"""

    def evaluate_trade(self, ticker: str, macro_report: str, technical_data: dict) -> str:
        system_prompt = self.get_system_prompt(ticker)
        
        user_prompt = f"""
--- ZENITH'S OBSIDIAN MACRO REPORT ON {ticker} ---
{macro_report}

--- SIGMA'S TECHNICAL DATA SUMMARY ON {ticker} ---
Current Price: {technical_data.get('current_price')}
SMA 20: {technical_data.get('sma_20')}
SMA 50: {technical_data.get('sma_50')}
ATR (14): {technical_data.get('atr_14')}
RSI (14): {technical_data.get('rsi_14')}
Volume: {technical_data.get('volume')}

Based on this complete profile, execute your Triple Filter Strategy and declare your final tactical operation. Let cold, hard math guide you.
"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            if isinstance(response.content, list):
                text_parts = [block.get("text", "") for block in response.content if isinstance(block, dict)]
                return "\n".join(text_parts)
            return response.content
        except Exception as e:
            return f"Sigma Error: Tactical analysis failed for {ticker}. Exception: {str(e)}"
