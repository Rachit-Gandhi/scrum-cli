#!/usr/bin/env python3
import asyncio
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
import json
import requests
import os
import glob

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
from rich.prompt import Prompt
from rich.markdown import Markdown

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScrumChatBot:
    """Interactive Scrum Meeting Chatbot"""
    
    def __init__(self, proxy_url: str):
        self.proxy_url = proxy_url.rstrip('/')
        self.console = Console()
        self.mode = "normal"
        self.chat_history = []
        self.meeting_context = None
        self.meeting_history = []
        self.is_live = False
        self.live_transcriber = None
        self.max_chat_history = 100  # Show more chat history
        self.scroll_offset = 0
        self.available_commands = [
            "/help", "/ridiculous", "/professional", "/export", "/load", "/quit",
            "/demo", "/live", "/actionitems", "/summary", "/stats", "/whosaid"
        ]
        
        # UI components
        self.layout = Layout()
        self.setup_layout()
        
        # Load existing chat history if available
        self.load_latest_chat_history()
        
    def setup_layout(self):
        """Setup the chat UI layout"""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="input_area", size=5),
            Layout(name="footer", size=3)
        )
        
        self.layout["main"].split_row(
            Layout(name="chat", ratio=3),
            Layout(name="sidebar", ratio=1)
        )
    
    def get_header(self) -> Panel:
        """Create header panel"""
        title = Text("SCRUMS-CLI v1.0", style="bold blue")
        
        if self.meeting_context:
            meeting_info = f"Meeting: {self.meeting_context}"
        else:
            meeting_info = "No Active Meeting"
        
        status = "[LIVE]" if self.is_live else "[IDLE]"
        
        header_text = Text()
        header_text.append(title)
        header_text.append(f" | {meeting_info}", style="dim")
        header_text.append(f" | {status}", style="green" if self.is_live else "dim")
        
        return Panel(
            Align.center(header_text),
            style="bright_blue",
            height=3
        )
    
    
    def get_footer(self) -> Panel:
        """Create footer panel with commands"""
        mode_indicator = "ROAST MODE" if self.mode == "ridiculous" else "PROFESSIONAL"
        
        commands = [
            "/help - Show commands",
            "/ridiculous - Roast mode", 
            "/summary - Meeting summary",
            "/export - Save chat",
            "/load - Load chat history",
            "/quit - Exit"
        ]
        
        footer_text = Text(f"{mode_indicator} | {' | '.join(commands)}", style="dim")
        
        return Panel(
            Align.center(footer_text),
            style="bright_black",
            height=3
        )
    
    def get_sidebar(self) -> Panel:
        """Create sidebar with meeting info"""
        if not self.meeting_history:
            if self.is_live:
                content = Text("Transcribing...\nAsk questions anytime!", style="cyan", justify="center")
            else:
                content = Text("No meeting data\navailable", style="dim", justify="center")
        else:
            speakers = list(set(item.get('speaker', 'Unknown') for item in self.meeting_history[-10:]))
            
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Info", style="cyan")
            
            if self.is_live:
                table.add_row("LIVE STREAM")
                table.add_row("")
            
            table.add_row("Quick Stats")
            table.add_row(f"Speakers: {len(speakers)}")
            table.add_row(f"Segments: {len(self.meeting_history)}")
            
            if speakers:
                table.add_row("")
                table.add_row("Active Speakers")
                for speaker in speakers[-5:]:
                    table.add_row(f"- {speaker}")
            
            if self.is_live:
                table.add_row("")
                table.add_row("Try asking:")
                table.add_row("- What's been said?")
                table.add_row("- Who's talking most?")
                table.add_row("- Summarize so far")
            
            content = table
        
        return Panel(
            content,
            title="Meeting Info",
            style="bright_black"
        )
    
    def get_chat_panel(self) -> Panel:
        """Create main chat panel"""
        if not self.chat_history:
            if self.mode == "ridiculous":
                welcome_msg = """
Welcome to SCRUMS-CLI ROAST MODE! 🎭🔥

I'm your sarcastic meeting assistant ready to roast corporate culture! I can:
• Analyze meetings with hilarious commentary
• Count buzzwords and call out meeting disasters  
• Generate witty meeting report cards
• Provide helpful info with maximum sass

Commands: /ridiculous (for roast analysis), /summary, /actionitems
Let's make fun of some meetings! 😈
                """
            else:
                welcome_msg = """
Welcome to SCRUMS-CLI! 🚀

I'm your AI meeting assistant. I can:
• Answer questions about your meetings
• Extract action items and decisions  
• Provide professional meeting summaries
• Roast meeting culture (try /ridiculous)

Commands: /summary, /ridiculous, /actionitems, /decisions, /stats
            """
            content = Align.center(Text(welcome_msg.strip(), style="dim"))
        else:
            # Show chat history with better scrolling
            chat_text = Text()
            
            # Show more messages (last 50 instead of 20)
            display_messages = self.chat_history[-self.max_chat_history:]
            
            for i, message in enumerate(display_messages):
                timestamp = message.get('timestamp', time.time())
                time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M")
                
                # Add buffer space before each message
                if i > 0:
                    chat_text.append("\n", style="white")
                
                if message['type'] == 'user':
                    chat_text.append(f"[{time_str}] ", style="dim")
                    chat_text.append("YOU: ", style="bold cyan")
                    chat_text.append(f"{message['content']}\n", style="white")
                elif message['type'] == 'bot':
                    chat_text.append(f"[{time_str}] ", style="dim")
                    
                    if self.mode == "ridiculous":
                        chat_text.append("SCRUM-BOT: 🎭 ", style="bold magenta")
                    else:
                        chat_text.append("SCRUM-BOT: ", style="bold green")
                    
                    # Add buffer line before bot response
                    chat_text.append("\n", style="white")
                    
                    # Parse and style bot response
                    content = message['content']
                    if content.startswith('```') or '•' in content or '#' in content:
                        # Render as markdown for better formatting
                        chat_text.append(f"{content}\n", style="white")
                    else:
                        chat_text.append(f"{content}\n", style="white")
            
            # Chat text is ready for display
            content = chat_text
        
        return Panel(
            content,
            title=f"Chat ({len(self.chat_history)} messages)",
            style="bright_white",
            height=None
        )
    
    def get_input_area(self) -> Panel:
        """Create input area panel"""
        input_text = Text()
        input_text.append("Commands: ", style="dim")
        input_text.append("/help /ridiculous /summary /export /load", style="cyan")
        input_text.append("\n\nType your message or command:\n", style="dim")
        
        # Create a simple input box representation
        input_text.append("┌" + "─" * 68 + "┐\n", style="bright_black")
        input_text.append("│ ", style="bright_black")
        input_text.append("> Ready for input...", style="dim italic")
        input_text.append(" " * 45 + " │\n", style="bright_black")
        input_text.append("└" + "─" * 68 + "┘", style="bright_black")
        
        return Panel(
            input_text,
            title="Input Area",
            style="bright_blue",
            height=5
        )
    
    def update_display(self):
        """Update the display layout"""
        self.layout["header"].update(self.get_header())
        self.layout["chat"].update(self.get_chat_panel())
        self.layout["sidebar"].update(self.get_sidebar())
        self.layout["input_area"].update(self.get_input_area())
        self.layout["footer"].update(self.get_footer())
    
    async def send_message(self, message: str) -> str:
        """Send message to the bot via proxy"""
        try:
            # Get live transcription data if available
            current_meeting_data = self.meeting_history[-50:]
            if self.live_transcriber and self.is_live:
                # Get fresh data from live transcriber
                live_data = self.live_transcriber.get_current_transcript()
                if live_data:
                    current_meeting_data = live_data[-50:]  # Last 50 segments
            
            response = requests.post(
                f"{self.proxy_url}/chat",
                json={
                    "message": message,
                    "context": self.meeting_context,
                    "mode": self.mode,
                    "meeting_history": current_meeting_data
                },
                timeout=60  # Increased timeout for longer responses
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Validate response data
            if 'response' not in data:
                logger.error(f"Invalid response format: {data}")
                return "❌ Invalid response format from server"
            
            response_text = data['response']
            
            # Check if response seems complete
            if not response_text or len(response_text.strip()) < 10:
                logger.warning("Received very short or empty response")
                return "❌ Received incomplete response. Please try again."
            
            return response_text
            
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return f"❌ Failed to get response from bot: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"❌ Unexpected error: {str(e)}"
    
    def handle_command(self, command: str) -> Optional[str]:
        """Handle special commands"""
        command = command.lower().strip()
        
        if command == "/help":
            return """
🔧 **Available Commands:**

**Chat Commands:**
- `/ridiculous` - Switch to roast mode 🎭
- `/professional` - Switch to professional mode 💼
- `/export` - Export current chat history
- `/load` - List available chat history files
- `/load <filename>` - Load specific chat history
- `/quit` - Exit the chat

**Meeting Commands:**
- `/demo` - Run all commands demo
- `/live` - Show recent transcription (during streaming)
- `/actionitems` - Show action items from current meeting
- `/summary` - Get professional meeting summary
- `/ridiculous` - Get hilarious roast analysis of meeting
- `/stats` - Show speaker statistics
- `/whosaid <text>` - Find who said something

Just type normally to chat with me!
            """
        
        elif command == "/ridiculous":
            self.mode = "ridiculous"
            # If we have meeting data and not in live mode, generate a roast summary
            if self.meeting_history and not self.is_live:
                return None  # Let the AI proxy handle the roast analysis
            elif not self.meeting_history:
                return "**ROAST MODE ACTIVATED!** 🎭 No meeting data to roast yet - start a transcription or ask me anything!"
            else:
                return "**ROAST MODE ACTIVATED!** Let's have some fun with meeting culture!"
        
        elif command == "/summary":
            # Generate a professional summary of the meeting data
            if self.meeting_history:
                # Keep current mode, the proxy will handle professional summary
                return None  # Let the AI proxy handle the summary
            else:
                return "No meeting data available for summary."
        
        elif command == "/professional":
            self.mode = "normal"
            return "**Professional mode activated.** Back to serious business."
        
        elif command == "/export":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scrum_chat_{timestamp}.json"
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.chat_history, f, indent=2, ensure_ascii=False)
                return f"Chat history exported to: `{filename}`"
            except Exception as e:
                return f"Export failed: {str(e)}"
        
        elif command == "/load":
            available_files = self.get_available_chat_files()
            if not available_files:
                return "No chat history files found."
            
            response = "**Available chat history files:**\n\n"
            for i, filename in enumerate(available_files[:10]):
                try:
                    # Get file timestamp from filename or modification time
                    file_time = datetime.fromtimestamp(os.path.getctime(filename))
                    response += f"{i+1}. `{filename}` ({file_time.strftime('%Y-%m-%d %H:%M')})\n"
                except:
                    response += f"{i+1}. `{filename}`\n"
            
            response += "\nUse `/load <filename>` to load a specific file."
            return response
        
        elif command.startswith("/load "):
            filename = command[6:].strip()
            if not filename:
                return "Usage: `/load <filename>`"
            
            if self.load_specific_chat_file(filename):
                return f"Loaded chat history from `{filename}` ({len(self.chat_history)} messages)"
            else:
                return f"Failed to load `{filename}`"
        
        elif command == "/demo":
            if not self.meeting_history:
                return "No meeting data available for demo"
            
            demo_response = "**Demo Mode - Running All Commands:**\n\n"
            
            # Show stats
            speakers = list(set(item.get('speaker', 'Unknown') for item in self.meeting_history))
            demo_response += f"**Stats**: {len(speakers)} speakers, {len(self.meeting_history)} segments\n\n"
            
            # Show recent transcript
            recent = self.meeting_history[-3:] if len(self.meeting_history) >= 3 else self.meeting_history
            demo_response += "**Recent Transcript:**\n"
            for item in recent:
                speaker = item.get('speaker', 'Unknown')
                text = item.get('text', '')[:100] + "..." if len(item.get('text', '')) > 100 else item.get('text', '')
                demo_response += f"- **{speaker}:** {text}\n"
            
            demo_response += "\n**Available Analysis:**\n"
            demo_response += "• `/summary` - Professional meeting analysis\n"
            demo_response += "• `/ridiculous` - Hilarious roast analysis 🎭\n"
            demo_response += "• Or ask natural questions!"
            
            return demo_response
        
        elif command == "/quit":
            return "QUIT_COMMAND"
        
        elif command == "/live":
            if self.is_live:
                recent = self.meeting_history[-5:] if self.meeting_history else []
                if recent:
                    response = "**Most recent transcription:**\n\n"
                    for item in recent:
                        speaker = item.get('speaker', 'Unknown')
                        text = item.get('text', '')
                        response += f"**{speaker}:** {text}\n\n"
                    return response
                else:
                    return "**Live stream active** - transcription starting soon!"
            else:
                return "No live stream active"
        
        elif command.startswith("/whosaid "):
            search_text = command[9:].strip()
            if not search_text:
                return "Usage: `/whosaid <text to search for>`"
            
            matches = []
            for item in self.meeting_history:
                if search_text.lower() in item.get('text', '').lower():
                    speaker = item.get('speaker', 'Unknown')
                    text = item.get('text', '')[:100] + "..." if len(item.get('text', '')) > 100 else item.get('text', '')
                    matches.append(f"**{speaker}:** {text}")
            
            if matches:
                return f"**Found '{search_text}' mentioned by:**\n\n" + "\n\n".join(matches[:5])
            else:
                return f"No mentions of '{search_text}' found in meeting history."
        
        return None
    
    async def run_chat(self):
        """Main chat loop"""
        self.console.print(Panel(
            Align.center(Text("Starting SCRUMS-CLI Chat Bot...", style="bold green")),
            style="green"
        ))
        
        with Live(self.layout, console=self.console, screen=True, auto_refresh=False) as live:
            while True:
                self.update_display()
                live.refresh()
                
                try:
                    # Enhanced input with autocomplete hints
                    user_input = self.get_user_input()
                    
                    if not user_input:
                        continue
                    
                    await self.process_user_input(user_input)
                    
                except KeyboardInterrupt:
                    break
                except EOFError:
                    break
                except Exception as e:
                    logger.error(f"Chat error: {e}")
                    self.chat_history.append({
                        'type': 'bot',
                        'content': f"❌ Error: {str(e)}",
                        'timestamp': time.time()
                    })
        
        self.console.print("\n[green]Thanks for using SCRUMS-CLI! 👋[/green]")
    
    async def process_user_input(self, user_input: str):
        """Process user input and generate response"""
        # Add user message to history
        self.chat_history.append({
            'type': 'user',
            'content': user_input,
            'timestamp': time.time()
        })
        
        # Handle commands
        if user_input.startswith('/'):
            command_response = self.handle_command(user_input)
            
            if command_response == "QUIT_COMMAND":
                raise KeyboardInterrupt
            
            if command_response:
                self.chat_history.append({
                    'type': 'bot',
                    'content': command_response,
                    'timestamp': time.time()
                })
                return
        
        # Send to bot
        bot_response = await self.send_message(user_input)
        
        # Add bot response to history
        self.chat_history.append({
            'type': 'bot',
            'content': bot_response,
            'timestamp': time.time()
        })
    
    def set_meeting_context(self, context: str):
        """Set the current meeting context"""
        self.meeting_context = context
    
    def add_meeting_data(self, data: Dict):
        """Add meeting transcript data"""
        self.meeting_history.append(data)
    
    def set_live_status(self, is_live: bool):
        """Set whether we're in a live meeting"""
        self.is_live = is_live
    
    def get_user_input(self) -> str:
        """Enhanced input with basic autocomplete"""
        try:
            # Clear previous input area and show prompt
            self.console.print("\n" + "─" * 70, style="dim")
            user_input = input("[bold cyan]>[/bold cyan] ").strip()
            
            # Basic autocomplete suggestion
            if user_input.startswith('/') and len(user_input) > 1:
                matches = [cmd for cmd in self.available_commands if cmd.startswith(user_input)]
                if len(matches) == 1 and matches[0] != user_input:
                    suggestion = matches[0]
                    self.console.print(f"[dim]Did you mean: {suggestion}? (Press Enter to use, or continue typing)[/dim]")
                elif len(matches) > 1:
                    self.console.print(f"[dim]Available: {', '.join(matches[:5])}[/dim]")
            
            return user_input
            
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception as e:
            self.console.print(f"[red]Input error: {e}[/red]")
            return ""
    
    def load_latest_chat_history(self):
        """Load the most recent chat history from JSON files"""
        try:
            # Find all scrum_chat_*.json files
            chat_files = glob.glob("scrum_chat_*.json")
            
            if not chat_files:
                return
            
            # Get the most recent file
            latest_file = max(chat_files, key=os.path.getctime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                loaded_history = json.load(f)
            
            # Validate and load the history
            if isinstance(loaded_history, list):
                self.chat_history = loaded_history
                self.console.print(f"[dim]Loaded {len(self.chat_history)} messages from {latest_file}[/dim]")
            
        except Exception as e:
            logger.warning(f"Could not load chat history: {e}")
    
    def get_available_chat_files(self):
        """Get list of available chat history files"""
        try:
            chat_files = glob.glob("scrum_chat_*.json")
            return sorted(chat_files, key=os.path.getctime, reverse=True)
        except Exception:
            return []
    
    def load_specific_chat_file(self, filename: str):
        """Load a specific chat history file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                loaded_history = json.load(f)
            
            if isinstance(loaded_history, list):
                self.chat_history = loaded_history
                return True
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
        return False

# Standalone chat function
async def start_chat_interface(proxy_url: str, meeting_context: Optional[str] = None):
    """Start the chat interface"""
    bot = ScrumChatBot(proxy_url)
    
    if meeting_context:
        bot.set_meeting_context(meeting_context)
    
    await bot.run_chat()

# CLI entry point
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        proxy_url = sys.argv[1]
    else:
        proxy_url = "http://localhost:8000"
    
    asyncio.run(start_chat_interface(proxy_url))