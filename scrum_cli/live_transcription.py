#!/usr/bin/env python3
import asyncio
import threading
import queue
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
import json
from pathlib import Path
import sys

from rich.console import Console

# Add transcription-tool to path
sys.path.append(str(Path(__file__).parent.parent / "transcription-tool"))

try:
    from transcriber import AudioTranscriber
    from speaker_diarization import SpeakerDiarizer, analyze_speaker_statistics
except ImportError as e:
    logging.error(f"Failed to import transcription modules: {e}")

from .memory_store import create_memory_store

logger = logging.getLogger(__name__)

class LiveMeetingTranscriber:
    
    def __init__(
        self,
        hf_token: Optional[str] = None,
        model_size: str = "medium",
        proxy_url: Optional[str] = None
    ):
        self.hf_token = hf_token
        self.model_size = model_size
        self.proxy_url = proxy_url
        
        # Transcription components
        self.transcriber = None
        self.diarizer = None
        
        # Live data
        self.meeting_data = []
        self.current_speakers = set()
        self.start_time = None
        self.meeting_title = None
        
        # Threading
        self.is_running = False
        self.transcription_queue = queue.Queue()
        self.callbacks = []
        
        # Storage
        self.memory_store = create_memory_store()
        
        # Initialize transcription models
        self._init_models()
    
    def _init_models(self):
        try:
            logger.info("Initializing transcription models...")
            
            # Initialize main transcriber
            self.transcriber = AudioTranscriber(
                hugging_face_token=self.hf_token,
                model_size=self.model_size,
                language="en"
            )
            
            # Initialize speaker diarizer if token available
            if self.hf_token:
                self.diarizer = SpeakerDiarizer(
                    hugging_face_token=self.hf_token,
                    embedding_match_threshold=0.5,
                    min_segment_duration=2.0
                )
                logger.info("Speaker diarizer initialized")
            else:
                logger.warning("No HF token - speaker diarization disabled")
            
            logger.info("Transcription models ready")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def add_callback(self, callback: Callable[[Dict], None]):
        self.callbacks.append(callback)
    
    def start_meeting(self, title: str = None) -> str:
        self.meeting_title = title or f"Meeting {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        self.start_time = time.time()
        self.meeting_data = []
        self.current_speakers = set()
        self.is_running = True
        
        logger.info(f"Started meeting: {self.meeting_title}")
        
        # Notify callbacks
        self._notify_callbacks({
            'type': 'meeting_start',
            'title': self.meeting_title,
            'timestamp': time.time()
        })
        
        return self.meeting_title
    
    def stop_meeting(self) -> Optional[str]:
        if not self.is_running:
            return None
        
        self.is_running = False
        
        # Generate final transcript
        if self.meeting_data:
            # Assign speaker names before saving
            speaker_names = self._assign_speaker_names()
            
            transcript_data = self._generate_transcript()
            
            # Save to memory store
            meeting_id = self.memory_store.store_meeting(
                self.meeting_title,
                transcript_data,
                {
                    'duration': time.time() - self.start_time,
                    'speakers': list(self.current_speakers),
                    'speaker_names': speaker_names,
                    'total_segments': len(self.meeting_data)
                }
            )
            
            logger.info(f"Meeting saved with ID: {meeting_id}")
            
            # Notify callbacks
            self._notify_callbacks({
                'type': 'meeting_end',
                'meeting_id': meeting_id,
                'transcript': transcript_data,
                'speaker_names': speaker_names,
                'timestamp': time.time()
            })
            
            return meeting_id
        
        return None
    
    def _assign_speaker_names(self) -> Dict[str, str]:
        speaker_names = {}
        
        # Load existing speaker names
        existing_names = self._load_speaker_names()
        
        console = Console()
        console.print("\n[bold cyan]🎤 Speaker Name Assignment[/bold cyan]")
        console.print("Assign real names to speakers (press Enter to skip):")
        
        for speaker in self.current_speakers:
            if speaker in existing_names:
                console.print(f"[green]{speaker} -> {existing_names[speaker]} (using saved name)[/green]")
                speaker_names[speaker] = existing_names[speaker]
            else:
                name = input(f"{speaker} -> ").strip()
                if name:
                    speaker_names[speaker] = name
                    console.print(f"[green]✅ {speaker} -> {name}[/green]")
                else:
                    speaker_names[speaker] = speaker
        
        # Save new names
        self._save_speaker_names(speaker_names)
        
        return speaker_names
    
    def _load_speaker_names(self) -> Dict[str, str]:
        speaker_file = self.memory_store.data_dir / "speaker_names.json"
        if speaker_file.exists():
            try:
                with open(speaker_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading speaker names: {e}")
        return {}
    
    def _save_speaker_names(self, speaker_names: Dict[str, str]):
        existing_names = self._load_speaker_names()
        existing_names.update(speaker_names)
        
        speaker_file = self.memory_store.data_dir / "speaker_names.json"
        try:
            with open(speaker_file, 'w') as f:
                json.dump(existing_names, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving speaker names: {e}")
    
    def process_audio_chunk(self, audio_file: str, chunk_info: Dict = None) -> Dict:
        try:
            # Transcribe the chunk
            if chunk_info and 'known_speakers' in chunk_info:
                result = self.transcriber.transcribe_with_known_speakers(
                    audio_file, chunk_info['known_speakers']
                )
            else:
                result = self.transcriber.transcribe_audio(audio_file)
            
            # Process segments
            segments = result.get('transcription', {}).get('segments', [])
            
            for segment in segments:
                # Add timing offset if this is a chunk
                if chunk_info and 'time_offset' in chunk_info:
                    segment['start'] += chunk_info['time_offset']
                    segment['end'] += chunk_info['time_offset']
                
                # Add to meeting data
                segment_data = {
                    'timestamp': time.time(),
                    'speaker': segment.get('speaker', 'Unknown'),
                    'text': segment.get('text', '').strip(),
                    'start': segment.get('start', 0),
                    'end': segment.get('end', 0),
                    'confidence': segment.get('confidence', 0.0)
                }
                
                self.meeting_data.append(segment_data)
                self.current_speakers.add(segment_data['speaker'])
                
                # Notify callbacks about new segment
                self._notify_callbacks({
                    'type': 'new_segment',
                    'segment': segment_data,
                    'meeting_title': self.meeting_title
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return {'error': str(e)}
    
    def add_manual_segment(self, speaker: str, text: str) -> Dict:
        segment_data = {
            'timestamp': time.time(),
            'speaker': speaker,
            'text': text.strip(),
            'start': time.time() - self.start_time if self.start_time else 0,
            'end': time.time() - self.start_time + 5 if self.start_time else 5,
            'confidence': 1.0,
            'manual': True
        }
        
        self.meeting_data.append(segment_data)
        self.current_speakers.add(speaker)
        
        # Notify callbacks
        self._notify_callbacks({
            'type': 'new_segment',
            'segment': segment_data,
            'meeting_title': self.meeting_title
        })
        
        return segment_data
    
    def get_current_transcript(self) -> List[Dict]:
        return self.meeting_data.copy()
    
    def get_recent_segments(self, count: int = 10) -> List[Dict]:
        return self.meeting_data[-count:] if self.meeting_data else []
    
    def get_speaker_stats(self) -> Dict:
        if not self.meeting_data:
            return {}
        
        speaker_stats = {}
        total_time = 0
        
        for segment in self.meeting_data:
            speaker = segment['speaker']
            duration = segment['end'] - segment['start']
            
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    'total_time': 0,
                    'segments': 0,
                    'words': 0
                }
            
            speaker_stats[speaker]['total_time'] += duration
            speaker_stats[speaker]['segments'] += 1
            speaker_stats[speaker]['words'] += len(segment['text'].split())
            total_time += duration
        
        # Calculate percentages
        for speaker in speaker_stats:
            if total_time > 0:
                speaker_stats[speaker]['percentage'] = (
                    speaker_stats[speaker]['total_time'] / total_time * 100
                )
            else:
                speaker_stats[speaker]['percentage'] = 0
        
        return {
            'speakers': speaker_stats,
            'total_time': total_time,
            'total_segments': len(self.meeting_data)
        }
    
    def save_partial_meeting(self) -> str:
        if not self.meeting_data:
            return None
        
        transcript_data = self._generate_transcript()
        meeting_id = self.memory_store.store_meeting(
            f"[PARTIAL] {self.meeting_title}",
            transcript_data,
            {
                'duration': time.time() - self.start_time if self.start_time else 0,
                'speakers': list(self.current_speakers),
                'total_segments': len(self.meeting_data),
                'partial': True
            }
        )
        
        logger.info(f"Partial meeting saved with ID: {meeting_id}")
        return meeting_id
    
    def _generate_transcript(self) -> Dict:
        if not self.meeting_data:
            return {}
        
        # Convert to standard transcription format
        segments = []
        for segment in self.meeting_data:
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'speaker': segment['speaker'],
                'text': segment['text'],
                'confidence': segment.get('confidence', 0.0)
            })
        
        return {
            'transcription': {
                'segments': segments,
                'speakers': list(self.current_speakers),
                'speaker_count': len(self.current_speakers),
                'total_duration': segments[-1]['end'] if segments else 0,
                'language': 'en'
            },
            'metadata': {
                'model': f'whisper-{self.model_size}',
                'diarization': 'pyannote' if self.diarizer else 'none',
                'meeting_title': self.meeting_title,
                'start_time': self.start_time,
                'end_time': time.time()
            }
        }
    
    def _notify_callbacks(self, data: Dict):
        for callback in self.callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

class ChatBotIntegration:
    
    def __init__(self, transcriber: LiveMeetingTranscriber, chatbot_ui=None):
        self.transcriber = transcriber
        self.chatbot_ui = chatbot_ui
        self.question_queue = queue.Queue()
        
        # Register callback for transcription updates
        self.transcriber.add_callback(self._on_transcription_update)
    
    def _on_transcription_update(self, data: Dict):
        if data['type'] == 'new_segment':
            segment = data['segment']
            
            # Update chatbot with new segment
            if self.chatbot_ui:
                self.chatbot_ui.add_meeting_data(segment)
                self.chatbot_ui.set_live_status(True)
        
        elif data['type'] == 'meeting_start':
            if self.chatbot_ui:
                self.chatbot_ui.set_meeting_context(data['title'])
                self.chatbot_ui.set_live_status(True)
        
        elif data['type'] == 'meeting_end':
            if self.chatbot_ui:
                self.chatbot_ui.set_live_status(False)
    
    def ask_question_about_meeting(self, question: str) -> str:
        recent_segments = self.transcriber.get_recent_segments(20)
        
        # Format context for the question
        context = []
        for segment in recent_segments:
            context.append(f"{segment['speaker']}: {segment['text']}")
        
        context_text = "\n".join(context)
        
        # This would be sent to the chatbot via the proxy
        return f"Question about meeting: {question}\nRecent context:\n{context_text}"

def create_live_transcriber(hf_token: str = None, proxy_url: str = None) -> LiveMeetingTranscriber:
    return LiveMeetingTranscriber(
        hf_token=hf_token,
        model_size="medium",
        proxy_url=proxy_url
    )

def demo_live_transcription():
    import os
    
    hf_token = os.getenv('HUGGING_FACE_TOKEN')
    
    transcriber = create_live_transcriber(hf_token)
    
    def demo_callback(data):
        if data['type'] == 'new_segment':
            segment = data['segment']
            print(f"[{segment['speaker']}]: {segment['text']}")
    
    transcriber.add_callback(demo_callback)
    
    meeting_title = transcriber.start_meeting("Demo Meeting")
    print(f"Started: {meeting_title}")
    
    transcriber.add_manual_segment("Alice", "Good morning everyone, let's start our daily standup")
    time.sleep(1)
    transcriber.add_manual_segment("Bob", "I finished the authentication feature yesterday")
    time.sleep(1)
    transcriber.add_manual_segment("Charlie", "I'm working on the API documentation today")
    
    stats = transcriber.get_speaker_stats()
    print(f"\nSpeaker Stats: {json.dumps(stats, indent=2)}")
    
    meeting_id = transcriber.stop_meeting()
    print(f"Meeting saved: {meeting_id}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_live_transcription()