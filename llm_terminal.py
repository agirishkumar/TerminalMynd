import subprocess
import argparse
from langchain_ollama import ChatOllama
from rich.console import Console
from rich.prompt import Prompt
import json
import os
import re
from typing import Optional, List, Dict, Tuple

class CommandProcessor:
    """Handles command processing and sanitization"""
    
    @staticmethod
    def extract_command(llm_response: str) -> str:
        """Extract actual command from LLM response"""
        # Remove markdown code blocks and explanatory text
        command = re.sub(r'```(?:bash|shell)?\n?(.*?)\n?```', r'\1', llm_response, flags=re.DOTALL)
        command = re.sub(r'Sure!.*?following command[s]?:?\s*', '', command, flags=re.DOTALL|re.IGNORECASE)
        command = re.sub(r'Here.*?command[s]?:?\s*', '', command, flags=re.DOTALL|re.IGNORECASE)
        command = re.sub(r'I can help.*?\s*', '', command, flags=re.DOTALL|re.IGNORECASE)
        command = re.sub(r'Please.*?\s*', '', command, flags=re.DOTALL|re.IGNORECASE)
        
        # Get the first line that looks like a command
        lines = command.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//') and ':' not in line:
                # Remove numbered lists (e.g., "1. command")
                line = re.sub(r'^\d+\.\s*', '', line)
                command = line
                break
        
        # Clean up the command
        command = command.strip('`$ ')
        return command
    
    @staticmethod
    def sanitize_command(command: str) -> str:
        """Sanitize command for safe execution"""
        # Remove potentially dangerous characters and sequences
        command = command.replace('$(', '').replace('`', '')
        command = command.replace('\n', ' ').strip()
        return command

    @staticmethod
    def is_package_installation_command(command: str) -> bool:
        """Check if the command is a package installation command"""
        install_patterns = [
            r'^pip\s+install',
            r'^python\s+-m\s+pip\s+install',
            r'^apt\s+install',
            r'^apt-get\s+install',
            r'^conda\s+install'
        ]
        return any(re.match(pattern, command) for pattern in install_patterns)


class TerminalMind:
    def __init__(self, interactive_mode: bool = False):
        self.console = Console()
        self.interactive_mode = interactive_mode
        self.llm = ChatOllama(model="llama2:7b")
        self.command_history = []
        self.context = {}
        self.command_processor = CommandProcessor()
        self.environment_context = self._get_environment_context()

    def _get_environment_context(self) -> Dict[str, str]:
        """Get information about the current Python environment"""
        try:
            python_path = subprocess.run(['which', 'python'], capture_output=True, text=True).stdout.strip()
            pip_path = subprocess.run(['which', 'pip'], capture_output=True, text=True).stdout.strip()
            
            return {
                'python_path': python_path,
                'pip_path': pip_path,
                'is_venv': os.environ.get('VIRTUAL_ENV') is not None
            }
        except Exception:
            return {}

    def execute_command(self, command: str) -> Tuple[bool, str]:
        """Execute a shell command and return the result"""
        try:
            # Sanitize command
            command = self.command_processor.sanitize_command(command)
            
            # Basic validation
            if not command:
                return False, "Empty command"
            
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            output = result.stdout.strip()
            self.command_history.append((command, output))
            return True, output
        except subprocess.CalledProcessError as e:
            return False, f"Error: {e.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return False, "Command timed out after 30 seconds"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    def get_llm_response(self, prompt: str, context: Optional[Dict] = None) -> Optional[str]:
        """Get command suggestions from LLM with context awareness"""
        try:
            # Build context-aware prompt
            context_str = ""
            if context:
                context_str = f"\nPrevious command output: {json.dumps(context)}\n"
                
            # Add environment context for package installation commands
            if any(keyword in prompt.lower() for keyword in ['install', 'pip', 'package']):
                env_str = f"\nEnvironment: Using {'virtual environment' if self.environment_context.get('is_venv') else 'system Python'}"
                context_str += env_str
                
            full_prompt = f"""Generate a bash command for the following task: {prompt}
            {context_str}
            Requirements:
            - Return ONLY the command itself, no explanations or comments
            - Do not include backticks, dollar signs, or markdown formatting
            - Do not include any explanatory text or numbered lists
            - Never include phrases like "Here's the command" or "I can help"
            - Use relative paths unless absolute paths are necessary
            - For package installation, use the appropriate pip command with --upgrade
            
            Example good responses:
            - pip install --upgrade numpy pandas torch
            - ls -la --color=auto
            - find . -type d -not -path "./.*"
            
            Example bad responses:
            - Here's the command to install packages: pip install numpy
            - 1. First run: pip install numpy
            - Sure! Use this command: `pip install numpy`
            """
            
            response = self.llm.invoke(full_prompt)
            command = self.command_processor.extract_command(response.content)
            
            # For package installation, ensure we use the correct pip
            if self.command_processor.is_package_installation_command(command):
                if self.environment_context.get('is_venv'):
                    command = f"pip install --upgrade {' '.join(command.split()[2:])}"
                else:
                    command = f"python -m pip install --upgrade {' '.join(command.split()[2:])}"
            
            return command
            
        except Exception as e:
            self.console.print(f"[red]Error getting LLM response: {e}[/red]")
            return None

    def review_command(self, command: str) -> Optional[str]:
        """Allow user to review, edit, or reject a command"""
        while True:
            choice = Prompt.ask(
                f"\nExecute this command?\n[bold cyan]{command}[/bold cyan]",
                choices=["y", "n", "edit"],
                default="n"
            )
            
            if choice == 'y':
                return command
            elif choice == 'n':
                return None
            else:
                edited_command = Prompt.ask("Edit command", default=command)
                verify = Prompt.ask(
                    f"\nExecute edited command?\n[bold cyan]{edited_command}[/bold cyan]",
                    choices=["y", "n"],
                    default="n"
                )
                if verify == 'y':
                    return edited_command
                elif verify == 'n':
                    continue

    def process_task(self, user_input: str) -> None:
        """Process user task with context awareness and command chaining"""
        # Get initial command suggestion
        command = self.get_llm_response(user_input, self.context)
        if not command:
            return

        # Review command if in interactive mode
        if self.interactive_mode:
            command = self.review_command(command)
            if not command:
                self.console.print("[yellow]Command aborted.[/yellow]")
                return

        # Execute command
        self.console.print("[yellow]Executing command...[/yellow]")
        success, output = self.execute_command(command)
        
        if success:
            self.console.print("[green]Command executed successfully:[/green]")
            self.console.print(output)
            
            # Update context with command output
            self.context = {
                'last_command': command,
                'last_output': output,
                'success': True
            }
            
            # Ask LLM if additional commands are needed based on the output
            follow_up = self.get_llm_response(
                f"Based on the output '{output}', determine if additional commands are needed to complete the task '{user_input}'. "
                "If yes, generate the next command. If no, respond with 'DONE'."
            )
            
            if follow_up and follow_up != "DONE":
                self.console.print("\n[yellow]Additional command suggested:[/yellow]")
                self.process_task(f"Continue with: {follow_up}")
        else:
            self.console.print(f"[red]Error:[/red] {output}")
            self.context = {
                'last_command': command,
                'last_output': output,
                'success': False
            }

    def show_history(self) -> None:
        """Display command history"""
        if not self.command_history:
            self.console.print("[yellow]No commands in history[/yellow]")
            return
        
        self.console.print("\n[bold]Command History:[/bold]")
        for idx, (cmd, output) in enumerate(self.command_history, 1):
            self.console.print(f"\n[cyan]{idx}.[/cyan] [bold]Command:[/bold] {cmd}")
            self.console.print(f"[bold]Output:[/bold]\n{output}")

    def run(self) -> None:
        """Main application loop"""
        self.console.print("[bold blue]Welcome to TerminalMind 2.0![/bold blue]")
        self.console.print("Available commands:")
        self.console.print("- Type your task in natural language")
        self.console.print("- 'history' to show command history")
        self.console.print("- 'clear' to clear the screen")
        self.console.print("- 'exit' to quit")
        
        if self.interactive_mode:
            self.console.print("[yellow]Running in interactive mode (y/n/edit)[/yellow]")
        
        while True:
            try:
                user_input = Prompt.ask("\nEnter a command or task").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == 'exit':
                    self.console.print("[yellow]Goodbye![/yellow]")
                    break
                    
                if user_input.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                    
                if user_input.lower() == 'history':
                    self.show_history()
                    continue
                
                self.process_task(user_input)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use 'exit' to quit safely[/yellow]")
            except Exception as e:
                self.console.print(f"[red]An unexpected error occurred: {e}[/red]")

def main():
    parser = argparse.ArgumentParser(description="TerminalMind - AI-powered terminal assistant")
    parser.add_argument(
        "-c", "--care",
        action="store_true",
        help="Enable interactive mode (y/n/edit) for command review"
    )
    args = parser.parse_args()
    
    terminal_mind = TerminalMind(interactive_mode=args.care)
    terminal_mind.run()

if __name__ == "__main__":
    main()