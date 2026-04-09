import sys
import os
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from brains.zenith_logic import ZenithAgent
from brains.sigma_logic import SigmaAgent

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), 'configs', '.env'))

console = Console()

def main():
    if len(sys.argv) < 2:
        console.print("[bold red]Usage: python main_orchestrator.py <TICKER>[/bold red]")
        sys.exit(1)
        
    ticker = sys.argv[1].upper()
    console.print(Panel.fit(f"[bold cyan]Initiating Multi-Agent Trading System for {ticker}[/bold cyan]", border_style="cyan"))

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            
            # Step 1: Initialize Agents
            task_init = progress.add_task(description="[yellow]Booting Agents (Zenith & Sigma)...", total=None)
            zenith = ZenithAgent()
            sigma = SigmaAgent()
            progress.update(task_init, completed=100)
            
            # Step 2: Zenith Macro Research
            task_zenith = progress.add_task(description="[magenta]Zenith: Conducting Grand Strategy Macro Research...", total=None)
            macro_report = zenith.generate_macro_report(ticker)
            progress.update(task_zenith, completed=100)
            
            # Step 2.5: Save to Obsidian Data Lake
            task_obsidian = progress.add_task(description="[green]Saving Research to Obsidian Data Lake...", total=None)
            current_date = datetime.now().strftime("%Y-%m-%d")
            obsidian_dir = os.path.join(os.path.dirname(__file__), 'data_lake', 'obsidian_sync')
            os.makedirs(obsidian_dir, exist_ok=True)
            
            file_name = f"{ticker}_{current_date}_research.md"
            file_path = os.path.join(obsidian_dir, file_name)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(macro_report)
            progress.update(task_obsidian, completed=100)
            
            # Step 3: Sigma Technical Analysis Gathering
            task_tech = progress.add_task(description="[blue]Sigma: Fetching Technical Data and computing indicators (SMA, ATR, RSI)...", total=None)
            technical_data = sigma.fetch_technical_data(ticker)
            progress.update(task_tech, completed=100)
            
            # Step 4: Sigma Final Evaluation
            task_sigma = progress.add_task(description="[red]Sigma: Synthesizing Zenith's Report and Technical Data for Final Evaluation...", total=None)
            if "error" in technical_data:
                console.print(f"[bold red]Sigma Error: {technical_data['error']}[/bold red]")
                sys.exit(1)
                
            final_decision = sigma.evaluate_trade(ticker, macro_report, technical_data)
            progress.update(task_sigma, completed=100)
            
    except Exception as e:
        console.print(f"[bold red]Fatal Error: {str(e)}[/bold red]")
        sys.exit(1)

    # Output Zenith's Results
    console.print(Panel(macro_report, title=f"[bold magenta]Zenith: The Grand Strategist - Macro Briefing ({ticker})[/bold magenta]", border_style="magenta"))
    
    # Output Sigma's Technicals
    tech_str = "\n".join([f"{k.upper()}: {v}" for k, v in technical_data.items()])
    console.print(Panel(tech_str, title=f"[bold blue]Sigma: Raw Technical Data ({ticker})[/bold blue]", border_style="blue"))
    
    # Output Final Decision
    console.print(Panel(final_decision, title=f"[bold red]Sigma: The Sniper - Tactical Evaluation ({ticker})[/bold red]", border_style="red"))

if __name__ == "__main__":
    main()
