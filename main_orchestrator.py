import sys
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from brains.zenith_logic import ZenithAgent
from brains.sigma_logic import SigmaAgent
from brains.archivist_logic import ArchivistAgent

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
            task_init = progress.add_task(description="[yellow]Booting Agents (Zenith, Archivist & Sigma)...", total=None)
            zenith = ZenithAgent()
            archivist = ArchivistAgent()
            sigma = SigmaAgent()
            progress.update(task_init, completed=100)
            
            # Step 2: Zenith Macro Research (Cloud / Perplexity)
            task_zenith = progress.add_task(description="[magenta]Zenith: Fetching Grand Strategy Macro Raw Data via Cloud...", total=None)
            raw_macro = zenith.generate_macro_report(ticker)
            progress.update(task_zenith, completed=100)
            
            # Step 3: Archivist Formatting & I/O (Local GPU / Ollama)
            task_archivist = progress.add_task(description="[green]Archivist: Synthesizing raw data into Obsidian Markdown via Local GPU...", total=None)
            macro_report = archivist.synthesize_to_obsidian(ticker, raw_macro)
            progress.update(task_archivist, completed=100)
            
            # Step 4: Sigma Technical Analysis Gathering
            task_tech = progress.add_task(description="[blue]Sigma: Fetching Technical Data and computing indicators (SMA, ATR, RSI)...", total=None)
            technical_data = sigma.fetch_technical_data(ticker)
            progress.update(task_tech, completed=100)
            
            # Step 5: Sigma Final Evaluation
            task_sigma = progress.add_task(description="[red]Sigma: Synthesizing Zenith's Report and Technical Data for Final Evaluation...", total=None)
            if "error" in technical_data:
                console.print(f"[bold red]Sigma Error: {technical_data['error']}[/bold red]")
                sys.exit(1)
                
            final_decision = sigma.evaluate_trade(ticker, macro_report, technical_data)
            progress.update(task_sigma, completed=100)
            
    except Exception as e:
        console.print(f"[bold red]Fatal Error: {str(e)}[/bold red]")
        sys.exit(1)

    # Output Archivist's Results (The Obsidian output)
    console.print(Panel(macro_report, title=f"[bold magenta]Archivist: Obsidian Macro Briefing ({ticker})[/bold magenta]", border_style="magenta"))
    
    # Output Sigma's Technicals
    tech_str = "\n".join([f"{k.upper()}: {v}" for k, v in technical_data.items()])
    console.print(Panel(tech_str, title=f"[bold blue]Sigma: Raw Technical Data ({ticker})[/bold blue]", border_style="blue"))
    
    # Output Final Decision
    console.print(Panel(final_decision, title=f"[bold red]Sigma: The Sniper - Tactical Evaluation ({ticker})[/bold red]", border_style="red"))

if __name__ == "__main__":
    main()
