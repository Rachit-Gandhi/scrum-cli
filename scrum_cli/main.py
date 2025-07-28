import click
import time
import sys
import os
import json
import logging
import asyncio
import threading
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import our new modules
try:
    from .proxy_server import start_proxy_server
    from .ngrok_manager import start_ngrok_tunnel
    from .chatbot_ui import start_chat_interface, ScrumChatBot
    from .memory_store import create_memory_store
    from .live_transcription import create_live_transcriber, ChatBotIntegration
    from .roast_engine import MeetingRoastEngine
except ImportError:
    # Fallback for direct execution
    from proxy_server import start_proxy_server
    from ngrok_manager import start_ngrok_tunnel
    from chatbot_ui import start_chat_interface, ScrumChatBot
    from memory_store import create_memory_store
    from live_transcription import create_live_transcriber, ChatBotIntegration
    from roast_engine import MeetingRoastEngine

# Add transcription-tool to path
sys.path.append(str(Path(__file__).parent.parent / "transcription-tool"))

# Import transcription tool
try:
    from transcriber import AudioTranscriber, save_transcription_json
    from speaker_diarization import SpeakerDiarizer, analyze_speaker_statistics
except ImportError as e:
    console = Console()
    console.print(f"[red]Error importing transcription modules: {e}[/red]")
    console.print("[yellow]Make sure the transcription-tool module is properly installed[/yellow]")

console = Console()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """SCRUMS-CLI: Real-time meeting transcription tool"""
    pass

@cli.command()
def status():
    """Check CLI status"""
    console.print("[green]SCRUMS-CLI is running[/green]")
    logger.info("CLI status check completed")

@cli.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--title', '-t', help='Meeting title/name')
@click.option('--speed', '-s', default=1.0, help='Transcription speed (1x, 2x, etc)')
@click.option('--proxy', '-p', help='Proxy server URL')
@click.option('--chunks', default=30, help='Audio chunk size in seconds')
def stream(audio_file, title, speed, proxy, chunks):
    """Stream a recorded meeting like a live session with real-time roasting"""
    
    console.print(Panel(
        "[bold green]SCRUMS-CLI Meeting Stream[/bold green]\n"
        f"Streaming: {audio_file}",
        style="green"
    ))
    
    # Check if file exists and is supported
    audio_path = Path(audio_file)
    if not audio_path.exists():
        console.print(f"[red]Error: Audio file not found: {audio_file}[/red]")
        return
    
    supported_formats = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
    if audio_path.suffix.lower() not in supported_formats:
        console.print(f"[yellow]Warning: File format {audio_path.suffix} might not be supported[/yellow]")
        console.print(f"Supported formats: {', '.join(supported_formats)}")
    
    final_proxy_url = None
    hf_token = None
    
    if proxy:
        console.print(f"[green]Using proxy: {proxy}[/green]")
        final_proxy_url = proxy.rstrip('/')
        
        try:
            import requests
            response = requests.get(f"{final_proxy_url}/health", timeout=10)
            if response.status_code == 200:
                console.print("[green]✅ Proxy connected[/green]")
                
                try:
                    hf_response = requests.get(f"{final_proxy_url}/hf-token", timeout=5)
                    if hf_response.status_code == 200:
                        hf_token = hf_response.json().get('hf_token')
                        if hf_token:
                            console.print("[green]Speaker diarization enabled[/green]")
                except:
                    console.print("[yellow]Speaker diarization disabled[/yellow]")
            else:
                console.print("[red]Proxy connection failed[/red]")
                return
        except Exception as e:
            console.print(f"[red]Cannot connect to proxy: {e}[/red]")
            return
    else:
        hf_token = os.getenv('HUGGING_FACE_TOKEN')
        if not hf_token:
            console.print("[yellow]No HuggingFace token - speaker diarization disabled[/yellow]")
    
    try:
        # Initialize components
        console.print("[blue]Initializing streaming components...[/blue]")
        
        live_transcriber = create_live_transcriber(hf_token, final_proxy_url)
        roast_engine = MeetingRoastEngine()
        
        if final_proxy_url:
            chatbot = ScrumChatBot(final_proxy_url)
        else:
            chatbot = None
        
        # Start streaming session
        meeting_title = title or f"Meeting - {audio_path.stem}"
        live_transcriber.start_meeting(meeting_title)
        
        if chatbot:
            chatbot.set_meeting_context(meeting_title)
            chatbot.set_live_status(True)
        
        console.print(f"\n[bold green]{meeting_title}[/bold green]")
        console.print(f"[dim]{audio_file} | {speed}x speed[/dim]")
        console.print("=" * 60)
        
        # Start chat and streaming simultaneously
        if chatbot and final_proxy_url:
            console.print(f"\n[bold green]Starting live chat mode - ask me anything![/bold green]")
            console.print("[dim]Transcription will continue in background[/dim]\n")
            
            # Pass live transcriber to chatbot for direct access
            chatbot.live_transcriber = live_transcriber
            
            # Start streaming in background thread
            import threading
            stream_thread = threading.Thread(
                target=_stream_audio_file,
                args=(audio_file, live_transcriber, roast_engine, chatbot, chunks, speed),
                daemon=True
            )
            stream_thread.start()
            
            # Start chat interface (blocking)
            asyncio.run(chatbot.run_chat())
            
            # Wait for streaming to complete
            stream_thread.join()
            
            # Show final transcript and assign speaker names
            _show_final_transcript_and_assign_names(live_transcriber)
            
            meeting_id = live_transcriber.stop_meeting()
        else:
            # Stream only without chat
            _stream_audio_file(
                audio_file, 
                live_transcriber, 
                roast_engine, 
                chatbot,
                chunks,
                speed
            )
            
            # Show final transcript and assign speaker names
            _show_final_transcript_and_assign_names(live_transcriber)
            
            meeting_id = live_transcriber.stop_meeting()
            _show_final_roast_report(roast_engine, meeting_id)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Meeting stream stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Stream error: {e}[/red]")
        logger.error(f"Stream error: {e}", exc_info=True)

@cli.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output JSON file path')
@click.option('--model', default='medium', type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']), help='Whisper model size')
@click.option('--language', default='en', help='Audio language (e.g., en, es, fr)')
@click.option('--hf-token', help='HuggingFace token for speaker diarization')
@click.option('--speakers', help='Path to JSON file with known speakers')
@click.option('--no-diarization', is_flag=True, help='Skip speaker diarization')
def transcribe(audio_file, output, model, language, hf_token, speakers, no_diarization):
    """Transcribe an audio file (MP3, WAV, etc.) with speaker identification"""
    
    console.print(f"[blue]Transcribing audio file: {audio_file}[/blue]")
    
    # Check if file exists and is supported
    audio_path = Path(audio_file)
    if not audio_path.exists():
        console.print(f"[red]Error: Audio file not found: {audio_file}[/red]")
        return
    
    supported_formats = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
    if audio_path.suffix.lower() not in supported_formats:
        console.print(f"[yellow]Warning: File format {audio_path.suffix} might not be supported[/yellow]")
        console.print(f"Supported formats: {', '.join(supported_formats)}")
    
    # Initialize transcriber
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            setup_task = progress.add_task("Initializing models...", total=None)
            
            # Determine if we should use diarization
            use_diarization = not no_diarization and hf_token is not None
            
            if not use_diarization and not no_diarization:
                console.print("[yellow]No HuggingFace token provided - speaker diarization disabled[/yellow]")
                console.print("[yellow]Use --hf-token to enable speaker diarization[/yellow]")
            
            transcriber = AudioTranscriber(
                hugging_face_token=hf_token if use_diarization else None,
                model_size=model,
                language=language
            )
            
            progress.update(setup_task, description="Loading known speakers...")
            
            # Load known speakers if provided
            speaker_samples = {}
            if speakers and Path(speakers).exists():
                try:
                    with open(speakers, 'r') as f:
                        speaker_data = json.load(f)
                    
                    # Validate speaker files exist
                    for name, path in speaker_data.items():
                        if Path(path).exists():
                            speaker_samples[name] = path
                        else:
                            console.print(f"[yellow]Warning: Speaker sample not found: {path}[/yellow]")
                    
                    console.print(f"[green]Loaded {len(speaker_samples)} speaker samples[/green]")
                    
                except Exception as e:
                    console.print(f"[yellow]Warning: Error loading speaker samples: {e}[/yellow]")
            
            progress.update(setup_task, description="Transcribing audio...")
            
            # Perform transcription
            if speaker_samples and use_diarization:
                result = transcriber.transcribe_with_known_speakers(audio_file, speaker_samples)
            else:
                result = transcriber.transcribe_audio(audio_file)
            
            progress.remove_task(setup_task)
        
        # Display results
        _display_transcription_results(result)
        
        # Speaker name assignment for transcription results
        if use_diarization and result.get('transcription', {}).get('segments'):
            _assign_speaker_names_to_result(result)
        
        # Save output
        if output:
            output_path = output
        else:
            # Generate default output filename
            output_path = audio_path.with_suffix('.json')
        
        save_transcription_json(result, output_path)
        console.print(f"\n[green]Transcription saved to: {output_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during transcription: {e}[/red]")
        logger.error(f"Transcription error: {e}", exc_info=True)

def _display_transcription_results(result: dict):
    """Display transcription results in a formatted way"""
    
    transcription = result['transcription']
    metadata = result['metadata']
    
    # Create summary table
    table = Table(title="Transcription Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Duration", f"{transcription['total_duration']:.1f} seconds")
    table.add_row("Language", transcription['language'])
    table.add_row("Speaker Count", str(transcription['speaker_count']))
    table.add_row("Model", metadata['model'])
    table.add_row("Diarization", metadata['diarization'])
    
    console.print(table)
    
    # Display speakers
    if transcription['speaker_count'] > 1:
        console.print(f"\n[bold]Speakers found:[/bold] {', '.join(transcription['speakers'])}")
    
    # Display transcript segments
    console.print("\n[bold]Transcript:[/bold]")
    console.print("=" * 60)
    
    for i, segment in enumerate(transcription['segments']):
        timestamp = f"[{segment['start']:.1f}s - {segment['end']:.1f}s]"
        speaker = segment['speaker']
        text = segment['text']
        
        # Color code speakers
        colors = ['blue', 'green', 'yellow', 'magenta', 'cyan', 'red']
        speaker_color = colors[hash(speaker) % len(colors)]
        
        console.print(f"[dim]{timestamp}[/dim] [bold {speaker_color}]{speaker}:[/bold {speaker_color}] {text}")
        
        # Add spacing every few segments for readability
        if (i + 1) % 5 == 0:
            console.print()

@cli.command()
@click.argument('speakers_file', type=click.Path())
def create_speakers_config(speakers_file):
    """Create a speaker configuration file template"""
    
    template = {
        "John_Doe": "/path/to/john_sample.wav",
        "Jane_Smith": "/path/to/jane_sample.wav",
        "Bob_Johnson": "/path/to/bob_sample.wav"
    }
    
    try:
        with open(speakers_file, 'w') as f:
            json.dump(template, f, indent=2)
        
        console.print(f"[green]Speaker configuration template created: {speakers_file}[/green]")
        console.print("[yellow]Edit the file to add paths to your speaker audio samples[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error creating speakers config: {e}[/red]")


def _stream_audio_file(audio_file, live_transcriber, roast_engine, chatbot, chunk_size, stream_speed):
    """Stream audio file in chunks like a live meeting"""
    import tempfile
    from pydub import AudioSegment
    
    try:
        # Load the audio file
        console.print("[blue]Loading audio file...[/blue]")
        audio = AudioSegment.from_file(audio_file)
        total_duration = len(audio) / 1000.0
        
        # Demo mode: limit to 15 minutes for long files
        demo_limit = 15 * 60  # 15 minutes in seconds
        if total_duration > demo_limit:
            console.print(f"[yellow]Demo mode: Using first 10 minutes of {total_duration/60:.1f} minute audio[/yellow]")
            audio = audio[:demo_limit * 1000]  # Truncate to 15 minutes
            total_duration = demo_limit
        
        console.print(f"[green]Audio loaded: {total_duration:.1f} seconds[/green]")
        
        # Set up streaming callback
        transcribed_segments = []
        def on_transcription_update(data):
            if data['type'] == 'new_segment':
                segment = data['segment']
                transcribed_segments.append(segment)
                
                if chatbot:
                    chatbot.add_meeting_data(segment)
                
                roast_analysis = roast_engine.analyze_segment(segment)
        
        # Register callback
        live_transcriber.add_callback(on_transcription_update)
        
        # Stream audio in chunks
        chunk_size_ms = int(chunk_size * 1000)
        total_chunks = len(audio) // chunk_size_ms + (1 if len(audio) % chunk_size_ms > 0 else 0)
        
        # Create progress bar
        from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
        
        with Progress(
            TextColumn("[bold blue]Transcribing..."),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            transcribe_task = progress.add_task("Processing audio", total=total_chunks)
            
            for i in range(0, len(audio), chunk_size_ms):
                chunk_number = (i // chunk_size_ms) + 1
                chunk = audio[i:i + chunk_size_ms]
                
                if len(chunk) < 1000:
                    continue
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    chunk.export(temp_file.name, format="wav")
                    
                    chunk_info = {
                        'time_offset': i / 1000.0,
                        'chunk_number': chunk_number,
                        'total_chunks': total_chunks
                    }
                    
                    try:
                        live_transcriber.process_audio_chunk(temp_file.name, chunk_info)
                    finally:
                        try:
                            os.unlink(temp_file.name)
                        except PermissionError:
                            time.sleep(0.1)
                            try:
                                os.unlink(temp_file.name)
                            except PermissionError:
                                pass
                
                progress.update(transcribe_task, advance=1)
                
                chunk_duration = len(chunk) / 1000.0
                wait_time = chunk_duration / stream_speed
                time.sleep(min(wait_time, 5.0))
        
        console.print(f"\n[green]Transcription complete! Ready for chat.[/green]")
        console.print(f"[cyan]Found {len(transcribed_segments)} segments from {len(set(s['speaker'] for s in transcribed_segments))} speakers[/cyan]")
        
    except Exception as e:
        console.print(f"[red]Streaming error: {e}[/red]")
        logger.error(f"Audio streaming error: {e}", exc_info=True)

def _assign_speaker_names_to_result(result):
    """Assign speaker names to transcription result"""
    segments = result['transcription']['segments']
    speakers = set(segment.get('speaker', 'Unknown') for segment in segments if segment.get('speaker'))
    
    if not speakers:
        console.print("[yellow]No speakers detected in transcription[/yellow]")
        return
    
    # Load existing speaker names
    existing_names = _load_speaker_names()
    
    console.print(f"\n[bold cyan]🎤 Speaker Name Assignment[/bold cyan]")
    console.print("Assign real names to speakers (press Enter to skip):")
    
    speaker_names = {}
    
    for speaker in sorted(speakers):
        if speaker in existing_names:
            console.print(f"[green]{speaker} -> {existing_names[speaker]} (using saved name)[/green]")
            speaker_names[speaker] = existing_names[speaker]
        else:
            console.print(f"\n[cyan]{speaker}[/cyan] appears {sum(1 for s in segments if s.get('speaker') == speaker)} times")  
            name = input(f"Enter real name for {speaker} (or press Enter to skip): ").strip()
            if name:
                speaker_names[speaker] = name
                console.print(f"[green]✅ {speaker} -> {name}[/green]")
            else:
                speaker_names[speaker] = speaker
    
    # Save names and update segments
    try:
        _save_speaker_names(speaker_names)
        console.print(f"\n[green]Speaker names saved for future meetings[/green]")
        
        # Update segments with assigned names
        for segment in segments:
            if segment.get('speaker') and segment['speaker'] in speaker_names:
                segment['assigned_name'] = speaker_names[segment['speaker']]
                
    except Exception as e:
        console.print(f"\n[yellow]Could not save speaker names: {e}[/yellow]")

def _load_speaker_names():
    """Load existing speaker names from file"""
    speaker_file = Path("meeting_data") / "speaker_names.json"
    if speaker_file.exists():
        try:
            with open(speaker_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading speaker names: {e}")
    return {}

def _save_speaker_names(speaker_names):
    """Save speaker names to file"""
    existing_names = _load_speaker_names()
    existing_names.update(speaker_names)
    
    speaker_file = Path("meeting_data") / "speaker_names.json"
    speaker_file.parent.mkdir(exist_ok=True)
    
    try:
        with open(speaker_file, 'w') as f:
            json.dump(existing_names, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving speaker names: {e}")

def _show_final_transcript_and_assign_names(live_transcriber):
    """Show final transcript and allow speaker name assignment"""
    transcript = live_transcriber.get_current_transcript()
    if not transcript:
        return
    
    console.print("\n" + "=" * 60)
    console.print("[bold blue]FINAL TRANSCRIPT[/bold blue]")
    console.print("=" * 60)
    
    # Show transcript
    speakers = set()
    for segment in transcript:
        speaker = segment.get('speaker', 'Unknown')
        text = segment.get('text', '').strip()
        start_time = segment.get('start', 0)
        
        speakers.add(speaker)
        
        # Color code speakers
        colors = ['blue', 'green', 'yellow', 'magenta', 'cyan', 'red']
        speaker_color = colors[hash(speaker) % len(colors)]
        
        time_str = f"[{start_time:.1f}s]"
        console.print(f"[dim]{time_str}[/dim] [bold {speaker_color}]{speaker}:[/bold {speaker_color}] {text}")
    
    # Speaker name assignment
    console.print(f"\n[bold cyan]Speaker Name Assignment[/bold cyan]")
    console.print("Assign real names to speakers (press Enter to skip):")
    
    speaker_names = {}
    
    # Load existing names
    try:
        existing_names = live_transcriber._load_speaker_names()
    except:
        existing_names = {}
    
    for speaker in sorted(speakers):
        if speaker in existing_names:
            console.print(f"[green]{speaker} -> {existing_names[speaker]} (saved)[/green]")
            speaker_names[speaker] = existing_names[speaker]
        else:
            console.print(f"\n[cyan]{speaker}[/cyan] appears {sum(1 for s in transcript if s.get('speaker') == speaker)} times")
            name = input(f"Enter real name for {speaker} (or press Enter to skip): ").strip()
            if name:
                speaker_names[speaker] = name
                console.print(f"[green]{speaker} -> {name}[/green]")
            else:
                speaker_names[speaker] = speaker
    
    # Save names
    try:
        live_transcriber._save_speaker_names(speaker_names)
        console.print(f"\n[green]Speaker names saved for future meetings[/green]")
    except Exception as e:
        console.print(f"\n[yellow]Could not save speaker names: {e}[/yellow]")

def _show_final_roast_report(roast_engine, meeting_id):
    """Show the final roast report"""
    console.print("\n" + "=" * 60)
    console.print("[bold blue]FINAL ROAST REPORT[/bold blue]")
    console.print("=" * 60)
    
    # Generate roast observations
    observations = roast_engine.generate_roast_observations()
    if observations:
        console.print("\n[bold magenta]Roast Observations:[/bold magenta]")
        for obs in observations:
            console.print(f"• {obs}")
    
    # Generate personality profiles
    profiles = roast_engine.generate_personality_profiles()
    if profiles:
        console.print("\n[bold cyan]Meeting Personality Profiles:[/bold cyan]")
        for speaker, profile in profiles.items():
            console.print(f"• [bold]{speaker}[/bold]: {profile}")
    
    # Generate report card
    report_card = roast_engine.generate_meeting_report_card()
    console.print("\n[bold yellow]📊 Meeting Report Card:[/bold yellow]")
    
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green")
    
    table.add_row("Efficiency Score", report_card['efficiency_score'])
    table.add_row("Grade", report_card['grade'])
    table.add_row("Buzzword Density", report_card['buzzword_density'])
    table.add_row("Duration", report_card['duration'])
    table.add_row("Speakers", str(report_card['speakers']))
    
    console.print(table)
    
    if report_card['awards']:
        console.print("\n[bold yellow]🏆 Awards:[/bold yellow]")
        for award in report_card['awards']:
            console.print(f"  {award}")
    
    console.print(f"\n[bold green]💡 AI Recommendation:[/bold green]")
    console.print(f"  {report_card['ai_recommendation']}")
    
    if meeting_id:
        console.print(f"\n[green]✅ Meeting saved with ID: {meeting_id}[/green]")
        console.print(f"[dim]Use 'scrum-cli show-meeting {meeting_id}' to view later[/dim]")

def _generate_report(transcriber, export_format: str):
    """Generate final transcription report"""
    
    console.print("\n[bold blue]Generating Report...[/bold blue]")
    logger.info("Generating final report")
    
    stats = transcriber.get_session_stats()
    
    table = Table(title="Transcription Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Duration", f"{stats['duration_minutes']} minutes")
    table.add_row("Total Chunks", str(stats['total_chunks']))
    table.add_row("Final Chunks", str(stats['final_chunks']))
    table.add_row("Words per Minute", f"{stats['words_per_minute']:.1f}")
    
    console.print(table)
    
    full_transcript = transcriber.get_full_transcript()
    if full_transcript:
        console.print(f"\n[bold]Transcript Preview:[/bold]")
        preview = full_transcript[:200] + "..." if len(full_transcript) > 200 else full_transcript
        console.print(f"[italic]{preview}[/italic]")
    
    exported_content = transcriber.export_transcript(export_format)
    
    timestamp = int(time.time())
    filename = f"transcript_{timestamp}.{export_format}"
    
    try:
        with open(filename, 'w') as f:
            f.write(exported_content)
        console.print(f"\n[green]Transcript saved: {filename}[/green]")
        logger.info(f"Transcript exported to {filename}")
    except Exception as e:
        console.print(f"\n[red]Export failed: {e}[/red]")
        logger.error(f"Export failed: {e}")


@cli.command()
@click.option('--proxy-url', help='Use external proxy server (e.g., your hosted ngrok URL)')
@click.option('--port', default=8000, help='Local proxy server port (if not using external)')
@click.option('--meeting-title', help='Set meeting title/context')
def chat(proxy_url, port, meeting_title):
    """Start the interactive SCRUM chatbot"""
    
    console.print(Panel(
        "[bold blue]🚀 Starting SCRUMS-CLI Chat Bot 🚀[/bold blue]",
        style="blue"
    ))
    
    # Determine proxy strategy
    if proxy_url:
        # Use external proxy (your hosted server)
        console.print(f"[green]Using external proxy server: {proxy_url}[/green]")
        final_proxy_url = proxy_url.rstrip('/')
        
        # Test connection to external proxy
        try:
            import requests
            response = requests.get(f"{final_proxy_url}/health", timeout=10)
            if response.status_code == 200:
                console.print("[green]✅ External proxy server is healthy[/green]")
            else:
                console.print("[red]❌ External proxy server health check failed[/red]")
                return
        except Exception as e:
            console.print(f"[red]❌ Cannot connect to external proxy: {e}[/red]")
            console.print("[yellow]Make sure the proxy URL is correct and accessible[/yellow]")
            return
            
    else:
        # Use local proxy with user's API keys
        console.print("[blue]Using local proxy with your API keys...[/blue]")
        
        # Check environment variables
        required_env = ['GEMINI_API_KEY']
        missing_env = [var for var in required_env if not os.getenv(var)]
        
        if missing_env:
            console.print(f"[red]Missing required environment variables: {', '.join(missing_env)}[/red]")
            console.print("[yellow]Options:[/yellow]")
            console.print("1. Create a .env file with:")
            console.print("   GEMINI_API_KEY=your_gemini_api_key")
            console.print("   HUGGING_FACE_TOKEN=your_hf_token  # Optional")
            console.print("")
            console.print("2. Or use the hosted proxy server:")
            console.print("   scrum-cli chat --proxy-url https://your-hosted-server.ngrok.io")
            return
        
        # Start local proxy server
        proxy_thread = threading.Thread(
            target=start_proxy_server,
            args=("127.0.0.1", port),
            daemon=True
        )
        proxy_thread.start()
        time.sleep(2)
        
        final_proxy_url = f"http://127.0.0.1:{port}"
        console.print(f"[green]Local proxy server running: {final_proxy_url}[/green]")
        
        # Test local proxy connection
        try:
            import requests
            response = requests.get(f"{final_proxy_url}/health", timeout=5)
            if response.status_code == 200:
                console.print("[green]✅ Local proxy server is healthy[/green]")
            else:
                console.print("[red]❌ Local proxy server health check failed[/red]")
                return
        except Exception as e:
            console.print(f"[red]❌ Cannot connect to local proxy server: {e}[/red]")
            return
    
    # Start chat interface
    try:
        asyncio.run(start_chat_interface(final_proxy_url, meeting_title))
    except KeyboardInterrupt:
        console.print("\n[yellow]Chat session ended[/yellow]")
    except Exception as e:
        console.print(f"[red]Chat error: {e}[/red]")
        logger.error(f"Chat error: {e}", exc_info=True)

@cli.command()
@click.option('--limit', default=20, help='Number of meetings to show')
def list_meetings(limit):
    """List stored meetings"""
    
    console.print("[blue]Loading meeting history...[/blue]")
    
    try:
        store = create_memory_store()
        meetings = store.list_meetings(limit)
        
        if not meetings:
            console.print("[yellow]No meetings found[/yellow]")
            return
        
        # Create table
        table = Table(title=f"Recent Meetings (Last {len(meetings)})")
        table.add_column("Date", style="cyan")
        table.add_column("Title", style="green")
        table.add_column("ID", style="dim")
        
        for meeting in meetings:
            date_str = meeting['timestamp'][:10] if isinstance(meeting['timestamp'], str) else str(meeting['timestamp'])[:10]
            table.add_row(
                date_str,
                meeting['title'],
                meeting['id'][:8] + "..."
            )
        
        console.print(table)
        
        # Show statistics
        stats = store.get_statistics()
        console.print(f"\n[dim]Total meetings: {stats['total_meetings']} | "
                     f"Vector documents: {stats['vector_documents']} | "
                     f"ChromaDB: {'✅' if stats['chromadb_available'] else '❌'}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error loading meetings: {e}[/red]")
        logger.error(f"Error loading meetings: {e}", exc_info=True)

@cli.command()
@click.argument('meeting_id')
def show_meeting(meeting_id):
    """Show details of a specific meeting"""
    
    try:
        store = create_memory_store()
        meeting = store.get_meeting(meeting_id)
        
        if not meeting:
            console.print(f"[red]Meeting not found: {meeting_id}[/red]")
            return
        
        # Show meeting header
        title = meeting.get('title', 'Unknown Meeting')
        timestamp = meeting.get('timestamp', 'Unknown Time')
        
        console.print(Panel(
            f"[bold]{title}[/bold]\n"
            f"ID: {meeting_id}\n"
            f"Date: {timestamp}",
            title="Meeting Details",
            style="blue"
        ))
        
        # Show transcript if available
        transcript_data = meeting.get('transcript', {})
        if transcript_data:
            transcription = transcript_data.get('transcription', {})
            segments = transcription.get('segments', [])
            
            console.print(f"\n[bold]Transcript ({len(segments)} segments):[/bold]")
            console.print("=" * 60)
            
            for segment in segments[:20]:  # Show first 20 segments
                timestamp_seg = f"[{segment.get('start', 0):.1f}s]"
                speaker = segment.get('speaker', 'Unknown')
                text = segment.get('text', '').strip()
                
                console.print(f"[dim]{timestamp_seg}[/dim] [bold cyan]{speaker}:[/bold cyan] {text}")
            
            if len(segments) > 20:
                console.print(f"\n[dim]... and {len(segments) - 20} more segments[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error loading meeting: {e}[/red]")
        logger.error(f"Error loading meeting: {e}", exc_info=True)

@cli.command()
@click.argument('query')
@click.option('--limit', default=10, help='Number of results to show')
def search_meetings(query, limit):
    """Search through meeting transcripts"""
    
    console.print(f"[blue]Searching for: '{query}'[/blue]")
    
    try:
        store = create_memory_store()
        results = store.search_meetings(query, limit)
        
        if not results:
            console.print("[yellow]No results found[/yellow]")
            return
        
        console.print(f"\n[bold]Found {len(results)} results:[/bold]")
        console.print("=" * 60)
        
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            text = result['text']
            score = result['relevance_score']
            
            meeting_title = metadata.get('meeting_title', 'Unknown Meeting')
            speaker = metadata.get('speaker', 'Unknown')
            
            console.print(f"\n[bold cyan]{i}. {meeting_title}[/bold cyan] "
                         f"[dim](Score: {score:.2f})[/dim]")
            console.print(f"[bold green]{speaker}:[/bold green] {text}")
        
    except Exception as e:
        console.print(f"[red]Search error: {e}[/red]")
        logger.error(f"Search error: {e}", exc_info=True)

@cli.command()
@click.option('--meeting-title', help='Meeting title/name')
@click.option('--proxy-url', help='Use external proxy server (e.g., your hosted ngrok URL)')
@click.option('--port', default=8000, help='Local proxy server port (if not using external)')
@click.option('--demo-mode', is_flag=True, help='Run in demo mode with simulated data')
def live_meeting(meeting_title, proxy_url, port, demo_mode):
    """Start live meeting with real-time transcription and chatbot"""
    
    console.print(Panel(
        "[bold green]🎭 SCRUMS-CLI Live Meeting Mode 🎭[/bold green]\n"
        "Real-time transcription + AI chatbot + Meeting roasts!",
        style="green"
    ))
    
    # Determine proxy strategy
    final_proxy_url = None
    hf_token = None
    
    if demo_mode:
        console.print("[yellow]🎭 Demo Mode: No API keys required![/yellow]")
    elif proxy_url:
        # Use external proxy (your hosted server)
        console.print(f"[green]Using external proxy server: {proxy_url}[/green]")
        final_proxy_url = proxy_url.rstrip('/')
        
        # Test connection to external proxy
        try:
            import requests
            response = requests.get(f"{final_proxy_url}/health", timeout=10)
            if response.status_code == 200:
                console.print("[green]✅ External proxy server is healthy[/green]")
                
                # Try to get HF token from external proxy
                try:
                    hf_response = requests.get(f"{final_proxy_url}/hf-token", timeout=5)
                    if hf_response.status_code == 200:
                        hf_token = hf_response.json().get('hf_token')
                        if hf_token:
                            console.print("[green]✅ HuggingFace token available from proxy[/green]")
                except:
                    console.print("[yellow]⚠️ HuggingFace token not available from proxy[/yellow]")
            else:
                console.print("[red]❌ External proxy server health check failed[/red]")
                return
        except Exception as e:
            console.print(f"[red]❌ Cannot connect to external proxy: {e}[/red]")
            console.print("[yellow]Make sure the proxy URL is correct and accessible[/yellow]")
            return
            
    else:
        # Use local proxy with user's API keys
        console.print("[blue]Using local proxy with your API keys...[/blue]")
        
        # Check environment variables
        required_env = ['GEMINI_API_KEY']
        missing_env = [var for var in required_env if not os.getenv(var)]
        
        if missing_env:
            console.print(f"[red]Missing required environment variables: {', '.join(missing_env)}[/red]")
            console.print("[yellow]Options:[/yellow]")
            console.print("1. Create a .env file with:")
            console.print("   GEMINI_API_KEY=your_gemini_api_key")
            console.print("   HUGGING_FACE_TOKEN=your_hf_token  # Optional")
            console.print("")
            console.print("2. Use the hosted proxy server:")
            console.print("   scrum-cli live-meeting --proxy-url https://your-hosted-server.ngrok.io")
            console.print("")
            console.print("3. Or just try the demo:")
            console.print("   scrum-cli live-meeting --demo-mode")
            return
        
        # Get local tokens
        hf_token = os.getenv('HUGGING_FACE_TOKEN')
        if not hf_token:
            console.print("[yellow]No HuggingFace token - speaker diarization will be disabled[/yellow]")
        
        # Start local proxy server
        proxy_thread = threading.Thread(
            target=start_proxy_server,
            args=("127.0.0.1", port),
            daemon=True
        )
        proxy_thread.start()
        time.sleep(2)
        
        final_proxy_url = f"http://127.0.0.1:{port}"
        console.print(f"[green]Local proxy server running: {final_proxy_url}[/green]")
        
        # Test local proxy connection
        try:
            import requests
            response = requests.get(f"{final_proxy_url}/health", timeout=5)
            if response.status_code == 200:
                console.print("[green]✅ Local proxy server is healthy[/green]")
            else:
                console.print("[red]❌ Local proxy server health check failed[/red]")
                return
        except Exception as e:
            console.print(f"[red]❌ Cannot connect to local proxy server: {e}[/red]")
            return
    
    try:
        
        # Initialize components
        console.print("[blue]Initializing AI components...[/blue]")
        
        # Create live transcriber
        live_transcriber = create_live_transcriber(hf_token, final_proxy_url)
        
        # Create roast engine
        roast_engine = MeetingRoastEngine()
        
        # Create chatbot
        if final_proxy_url:
            chatbot = ScrumChatBot(final_proxy_url)
        else:
            chatbot = None
        
        # Integrate components
        def on_transcription_update(data):
            """Handle transcription updates"""
            if data['type'] == 'new_segment':
                segment = data['segment']
                
                # Update chatbot
                if chatbot:
                    chatbot.add_meeting_data(segment)
                
                # Update roast engine
                roast_analysis = roast_engine.analyze_segment(segment)
                
                # Display real-time info
                speaker = segment['speaker']
                text = segment['text']
                
                # Color code speakers
                colors = ['blue', 'green', 'yellow', 'magenta', 'cyan', 'red']
                speaker_color = colors[hash(speaker) % len(colors)]
                
                console.print(f"[{speaker_color}]{speaker}:[/{speaker_color}] {text}")
                
                # Show roast updates occasionally
                if roast_analysis['buzzwords_found']:
                    buzzwords = ', '.join(roast_analysis['buzzwords_found'].keys())
                    console.print(f"[dim]🎭 Buzzword alert: {buzzwords}[/dim]")
        
        # Register callback
        live_transcriber.add_callback(on_transcription_update)
        
        # Start meeting
        title = meeting_title or f"Live Meeting {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        live_transcriber.start_meeting(title)
        
        if chatbot:
            chatbot.set_meeting_context(title)
            chatbot.set_live_status(True)
        
        console.print(f"\n[bold green]🎬 Meeting Started: {title}[/bold green]")
        console.print("=" * 60)
        
        if demo_mode:
            console.print("[yellow]Demo Mode: Simulating meeting data...[/yellow]")
            
            # Run demo meeting
            demo_segments = [
                ("Alice", "Good morning everyone! Let's leverage our synergy to make this meeting actionable"),
                ("Bob", "Absolutely! We need to circle back on the API redesign and take it offline"),
                ("Charlie", "I think we should pivot our approach and do a deep dive into the low hanging fruit"),
                ("Alice", "Great synergy, Bob! Let's touch base after we leverage these insights"),
                ("Dave", "Can we take this offline? I don't have the bandwidth for another deep dive"),
                ("Bob", "Let's circle back on that. We need more synergy in our actionable pivots"),
            ]
            
            for i, (speaker, text) in enumerate(demo_segments):
                time.sleep(2)  # Simulate speaking time
                live_transcriber.add_manual_segment(speaker, text)
                
                if i == 3:  # Halfway through, show roast update
                    roast_update = roast_engine.get_live_roast_update()
                    console.print(f"\n[magenta]🎭 Roast Update: {roast_update}[/magenta]\n")
            
            time.sleep(2)
            console.print("\n[yellow]Demo meeting completed! Press Enter to see final roast...[/yellow]")
            input()
            
        else:
            # Real meeting mode
            console.print("[yellow]Listening for audio... (In real implementation, this would capture live audio)[/yellow]")
            console.print("[dim]For now, you can simulate by using the 'transcribe' command with audio files[/dim]")
            console.print("\nPress Enter to end meeting...")
            input()
        
        # End meeting and generate final roast
        meeting_id = live_transcriber.stop_meeting()
        
        console.print("\n" + "=" * 60)
        console.print("[bold blue]🎭 FINAL ROAST REPORT 🎭[/bold blue]")
        console.print("=" * 60)
        
        # Generate roast observations
        observations = roast_engine.generate_roast_observations()
        if observations:
            console.print("\n[bold magenta]Roast Observations:[/bold magenta]")
            for obs in observations:
                console.print(f"• {obs}")
        
        # Generate personality profiles
        profiles = roast_engine.generate_personality_profiles()
        if profiles:
            console.print("\n[bold cyan]Meeting Personality Profiles:[/bold cyan]")
            for speaker, profile in profiles.items():
                console.print(f"• [bold]{speaker}[/bold]: {profile}")
        
        # Generate report card
        report_card = roast_engine.generate_meeting_report_card()
        console.print("\n[bold yellow]📊 Meeting Report Card:[/bold yellow]")
        
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="green")
        
        table.add_row("Efficiency Score", report_card['efficiency_score'])
        table.add_row("Grade", report_card['grade'])
        table.add_row("Buzzword Density", report_card['buzzword_density'])
        table.add_row("Duration", report_card['duration'])
        table.add_row("Speakers", str(report_card['speakers']))
        
        console.print(table)
        
        if report_card['awards']:
            console.print("\n[bold yellow]🏆 Awards:[/bold yellow]")
            for award in report_card['awards']:
                console.print(f"  {award}")
        
        console.print(f"\n[bold green]💡 AI Recommendation:[/bold green]")
        console.print(f"  {report_card['ai_recommendation']}")
        
        if meeting_id:
            console.print(f"\n[green]✅ Meeting saved with ID: {meeting_id}[/green]")
            console.print(f"[dim]Use 'scrum-cli show-meeting {meeting_id}' to view later[/dim]")
        
        # Offer to start chatbot
        console.print("\n[cyan]Start interactive chat about this meeting? (y/N): [/cyan]", end="")
        if chatbot and input().lower().startswith('y'):
            chatbot.set_live_status(False)
            asyncio.run(chatbot.run_chat())
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Meeting interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Live meeting error: {e}[/red]")
        logger.error(f"Live meeting error: {e}", exc_info=True)
    finally:
        # Cleanup is handled automatically since we're not managing ngrok here
        pass

@cli.command()
@click.argument('text', required=False)
def test_transcript(text):
    """Test transcript processing"""
    
    if text:
        words = text.strip().split()
        if words:
            last_word = words[-1]
            console.print(f"[green]Last word: [bold]{last_word}[/bold][/green]")
            logger.info(f"Test transcript processed: {text}")
        else:
            console.print("[red]No words found[/red]")
    else:
        console.print("[yellow]Usage: scrum-cli test-transcript \"your text here\"[/yellow]")

@cli.command()
@click.option('--port', default=8000, help='Server port')
@click.option('--host', default='127.0.0.1', help='Server host')
@click.option('--no-ngrok', is_flag=True, help='Skip ngrok tunnel')
@click.option('--ngrok-domain', help='Custom ngrok domain')
def host_proxy(port, host, no_ngrok, ngrok_domain):
    """Host the centralized proxy server for other users to connect to"""
    
    console.print(Panel(
        "[bold blue]🌐 SCRUMS-CLI Proxy Server Host 🌐[/bold blue]\n"
        "Host a centralized proxy server with your API keys",
        style="blue"
    ))
    
    # Check environment variables
    required_env = ['GEMINI_API_KEY']
    missing_env = [var for var in required_env if not os.getenv(var)]
    
    if missing_env:
        console.print(f"[red]Missing required environment variables: {', '.join(missing_env)}[/red]")
        console.print("[yellow]Create a .env file with:[/yellow]")
        console.print("GEMINI_API_KEY=your_gemini_api_key")
        console.print("HUGGING_FACE_TOKEN=your_hf_token  # Optional for speaker diarization")
        return
    
    # Show configuration
    console.print(f"[green]Server: {host}:{port}[/green]")
    
    hf_token = os.getenv('HUGGING_FACE_TOKEN')
    if hf_token:
        console.print("[green]✅ HuggingFace token configured - speaker diarization enabled[/green]")
    else:
        console.print("[yellow]⚠️ No HuggingFace token - speaker diarization disabled[/yellow]")
    
    try:
        # Start ngrok tunnel if requested
        ngrok_manager = None
        if not no_ngrok:
            console.print("[yellow]🌐 Starting ngrok tunnel...[/yellow]")
            ngrok_manager = start_ngrok_tunnel(port, ngrok_domain)
            
            if ngrok_manager:
                tunnel_url = ngrok_manager.get_tunnel_url()
                console.print(f"[green]✅ Ngrok tunnel active: {tunnel_url}[/green]")
                console.print(Panel(
                    f"[bold green]🔗 Share this URL with users:[/bold green]\n\n"
                    f"[bold cyan]{tunnel_url}[/bold cyan]\n\n"
                    f"[dim]Users can connect with:[/dim]\n"
                    f"[yellow]scrum-cli chat --proxy-url {tunnel_url}[/yellow]\n"
                    f"[yellow]scrum-cli live-meeting --proxy-url {tunnel_url}[/yellow]",
                    title="🌐 Public Proxy URL",
                    style="green"
                ))
            else:
                console.print("[yellow]⚠️ Ngrok tunnel failed, running locally only[/yellow]")
                console.print(f"[dim]Local access only: http://{host}:{port}[/dim]")
        else:
            console.print(f"[yellow]Running locally only: http://{host}:{port}[/yellow]")
        
        # Start the proxy server (this blocks)
        console.print("[green]🔥 Starting proxy server... (Ctrl+C to stop)[/green]")
        start_proxy_server(host, port)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]🛑 Shutting down proxy server...[/yellow]")
    except Exception as e:
        console.print(f"[red]💥 Server error: {e}[/red]")
        logger.error(f"Server error: {e}", exc_info=True)
    finally:
        if ngrok_manager:
            console.print("[blue]🔒 Stopping ngrok tunnel...[/blue]")
            ngrok_manager.stop_tunnel()
        console.print("[green]👋 Proxy server stopped[/green]")

if __name__ == "__main__":
    cli()