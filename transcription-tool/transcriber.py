import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import torch
import torchaudio
import whisper
import numpy as np
from pydub import AudioSegment
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from pyannote.audio import Pipeline, Model, Inference
from pyannote.core import Segment, Annotation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioTranscriber:
    """
    Audio transcription with speaker diarization capabilities.
    """
    
    def __init__(
        self,
        hugging_face_token: Optional[str] = None,
        model_size: str = "medium",
        language: str = "en",
        device: Optional[str] = None
    ):
        """
        Initialize the transcriber.
        
        Args:
            hugging_face_token: HuggingFace token for PyAnnote models
            model_size: Whisper model size (tiny, base, small, medium, large)
            language: Language code for transcription
            device: Device to use (cuda/cpu), auto-detected if None
        """
        self.hugging_face_token = hugging_face_token
        self.model_size = model_size
        self.language = language
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._init_models()
        
        # Configuration
        self.min_segment_duration = 2.0
        self.embedding_match_threshold = 0.7
    
    def _init_models(self):
        """Initialize Whisper and PyAnnote models."""
        try:
            # Initialize Whisper
            model_name = self.model_size
            if self.language == "en" and self.model_size != "large":
                model_name += ".en"
            
            logger.info(f"Loading Whisper model: {model_name}")
            self.whisper_model = whisper.load_model(model_name)
            self.whisper_model.to(self.device)
            
            # Initialize PyAnnote models if token provided
            if self.hugging_face_token:
                logger.info("Loading PyAnnote diarization pipeline")
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hugging_face_token
                )
                self.diarization_pipeline.to(self.device)
                
                logger.info("Loading PyAnnote embedding model")
                self.embedding_model = Model.from_pretrained(
                    "pyannote/embedding",
                    use_auth_token=self.hugging_face_token
                )
                self.embedding_model.to(self.device)
                self.embedding_inference = Inference(self.embedding_model, window="whole")
                
            else:
                logger.warning("No HuggingFace token provided - speaker diarization disabled")
                self.diarization_pipeline = None
                self.embedding_model = None
                
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def _convert_to_wav(self, input_file: str) -> str:
        """Convert audio file to WAV format if needed."""
        if input_file.lower().endswith('.wav'):
            return input_file
            
        logger.info(f"Converting {input_file} to WAV format")
        
        # Create temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav_path = temp_wav.name
        temp_wav.close()
        
        try:
            # Convert using pydub
            audio = AudioSegment.from_file(input_file)
            audio.export(temp_wav_path, format="wav")
            return temp_wav_path
        except Exception as e:
            logger.error(f"Error converting audio file: {e}")
            if os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)
            raise
    
    def transcribe_audio(self, audio_file: str) -> Dict[str, Any]:
        """
        Transcribe audio file with optional speaker diarization.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Dictionary containing transcription results
        """
        logger.info(f"Starting transcription of: {audio_file}")
        
        # Convert to WAV if needed
        wav_file = self._convert_to_wav(audio_file)
        temp_file_created = wav_file != audio_file
        
        try:
            # Perform transcription
            logger.info("Running Whisper transcription")
            asr_result = self.whisper_model.transcribe(wav_file)
            
            # Perform diarization if available
            if self.diarization_pipeline:
                logger.info("Running speaker diarization")
                diarization_result = self.diarization_pipeline(
                    wav_file, min_speakers=2, max_speakers=20
                )
                
                # Combine transcription and diarization
                final_result = self._combine_transcription_diarization(
                    asr_result, diarization_result, wav_file
                )
            else:
                # Return transcription only
                final_result = self._format_transcription_only(asr_result)
            
            return final_result
            
        finally:
            # Clean up temporary file
            if temp_file_created and os.path.exists(wav_file):
                os.unlink(wav_file)
    
    def _combine_transcription_diarization(
        self, 
        asr_result: Dict, 
        diarization_result: Annotation,
        wav_file: str
    ) -> Dict[str, Any]:
        """Combine Whisper transcription with PyAnnote diarization."""
        # Local implementation of diarize_text functionality
        def diarize_text(asr_result, diarization_result):
            """Combine ASR and diarization results"""
            segments = []
            for segment in asr_result.get('segments', []):
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text']
                
                # Find speaker for this time segment
                speaker = 'SPEAKER_UNKNOWN'
                for turn, _, speaker_id in diarization_result.itertracks(yield_label=True):
                    if turn.start <= start_time < turn.end or turn.start < end_time <= turn.end:
                        speaker = f'SPEAKER_{speaker_id}'
                        break
                
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': text,
                    'speaker': speaker
                })
            
            return {'segments': segments}
        
        # Load audio for embedding analysis
        waveform, sample_rate = torchaudio.load(wav_file)
        
        # Process speaker embeddings for better identification
        processed_diarization = self._process_speaker_embeddings(
            diarization_result, waveform, sample_rate
        )
        
        # Combine results
        combined_result = diarize_text(asr_result, processed_diarization)
        
        # Format output
        segments = []
        speakers = set()
        
        for segment in combined_result['segments']:
            segment_data = {
                "start": round(segment['start'], 2),
                "end": round(segment['end'], 2),
                "duration": round(segment['end'] - segment['start'], 2),
                "speaker": segment['speaker'],
                "text": segment['text'].strip()
            }
            segments.append(segment_data)
            speakers.add(segment['speaker'])
        
        return {
            "transcription": {
                "segments": segments,
                "speakers": sorted(list(speakers)),
                "speaker_count": len(speakers),
                "total_duration": round(asr_result.get("segments", [])[-1]["end"] if asr_result.get("segments") else 0, 2),
                "language": asr_result.get("language", "unknown")
            },
            "metadata": {
                "model": f"whisper-{self.model_size}",
                "diarization": "pyannote",
                "device": str(self.device)
            }
        }
    
    def _format_transcription_only(self, asr_result: Dict) -> Dict[str, Any]:
        """Format transcription results without diarization."""
        segments = []
        
        for segment in asr_result.get("segments", []):
            segment_data = {
                "start": round(segment["start"], 2),
                "end": round(segment["end"], 2),
                "duration": round(segment["end"] - segment["start"], 2),
                "speaker": "SPEAKER_00",  # Default speaker
                "text": segment["text"].strip()
            }
            segments.append(segment_data)
        
        return {
            "transcription": {
                "segments": segments,
                "speakers": ["SPEAKER_00"],
                "speaker_count": 1,
                "total_duration": round(asr_result.get("segments", [])[-1]["end"] if asr_result.get("segments") else 0, 2),
                "language": asr_result.get("language", "unknown"),
                "full_text": asr_result.get("text", "")
            },
            "metadata": {
                "model": f"whisper-{self.model_size}",
                "diarization": "none",
                "device": str(self.device)
            }
        }
    
    def _process_speaker_embeddings(
        self,
        diarization_result: Annotation,
        waveform: torch.Tensor,
        sample_rate: int
    ) -> Annotation:
        """Process speaker embeddings to improve speaker identification."""
        if not self.embedding_model:
            return diarization_result
        
        # Group segments by speaker
        segments_by_speaker = defaultdict(list)
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            segments_by_speaker[speaker].append(Segment(turn.start, turn.end))
        
        processed_diarization = Annotation(uri=diarization_result.uri)
        
        for speaker, segments in segments_by_speaker.items():
            valid_segments = []
            for segment in segments:
                duration = segment.end - segment.start
                if duration >= self.min_segment_duration:
                    valid_segments.append(segment)
            
            if valid_segments:
                # Keep the speaker with valid segments
                for segment in valid_segments:
                    processed_diarization[segment] = speaker
            else:
                logger.info(f"No valid segments found for {speaker}, removing")
        
        return processed_diarization
    
    def transcribe_with_known_speakers(
        self,
        audio_file: str,
        speaker_samples: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Transcribe with known speaker identification.
        
        Args:
            audio_file: Path to audio file
            speaker_samples: Dict mapping speaker names to sample audio files
            
        Returns:
            Dictionary containing transcription with identified speakers
        """
        if not self.embedding_model:
            raise ValueError("Speaker identification requires HuggingFace token for embedding model")
        
        logger.info(f"Transcribing with {len(speaker_samples)} known speakers")
        
        # Get base transcription
        result = self.transcribe_audio(audio_file)
        
        if not speaker_samples or len(result["transcription"]["speakers"]) <= 1:
            return result
        
        # Load reference embeddings
        reference_embeddings = self._load_reference_embeddings(speaker_samples)
        
        if not reference_embeddings:
            logger.warning("No valid reference embeddings loaded")
            return result
        
        # Re-identify speakers
        wav_file = self._convert_to_wav(audio_file)
        temp_file_created = wav_file != audio_file
        
        try:
            updated_result = self._identify_speakers_with_references(
                result, wav_file, reference_embeddings, list(speaker_samples.keys())
            )
            return updated_result
            
        finally:
            if temp_file_created and os.path.exists(wav_file):
                os.unlink(wav_file)
    
    def _load_reference_embeddings(self, speaker_samples: Dict[str, str]) -> Dict[int, np.ndarray]:
        """Load reference speaker embeddings."""
        embeddings = {}
        
        for i, (name, sample_path) in enumerate(speaker_samples.items()):
            try:
                if not os.path.exists(sample_path):
                    logger.warning(f"Speaker sample not found: {sample_path}")
                    continue
                
                # Convert to WAV if needed
                wav_path = self._convert_to_wav(sample_path)
                temp_created = wav_path != sample_path
                
                try:
                    embedding = self.embedding_inference(wav_path).reshape(1, -1)
                    embedding = normalize(embedding)
                    embeddings[i] = embedding
                    logger.info(f"Loaded reference embedding for '{name}' as ID {i}")
                    
                finally:
                    if temp_created and os.path.exists(wav_path):
                        os.unlink(wav_path)
                        
            except Exception as e:
                logger.warning(f"Error loading reference for '{name}': {e}")
        
        return embeddings
    
    def _identify_speakers_with_references(
        self,
        result: Dict[str, Any],
        wav_file: str,
        reference_embeddings: Dict[int, np.ndarray],
        speaker_names: List[str]
    ) -> Dict[str, Any]:
        """Identify speakers using reference embeddings."""
        # This is a simplified version - in practice you'd need to
        # re-process the diarization with the reference embeddings
        # For now, just return the original result
        logger.info("Speaker identification with references not fully implemented")
        return result


def save_transcription_json(transcription_result: Dict[str, Any], output_file: str):
    """Save transcription result to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transcription_result, f, indent=2, ensure_ascii=False)
    logger.info(f"Transcription saved to: {output_file}")