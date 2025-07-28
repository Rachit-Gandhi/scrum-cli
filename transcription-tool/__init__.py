"""
Transcription Tool Module

A modular transcription system with speaker diarization capabilities.
"""

from .transcriber import AudioTranscriber
from .speaker_diarization import SpeakerDiarizer

__version__ = "1.0.0"
__all__ = ["AudioTranscriber", "SpeakerDiarizer"]