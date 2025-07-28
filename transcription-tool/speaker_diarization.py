import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torchaudio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from pyannote.audio import Model, Inference
from pyannote.core import Segment, Annotation

logger = logging.getLogger(__name__)


class SpeakerDiarizer:
    """
    Advanced speaker diarization with speaker identification capabilities.
    """
    
    def __init__(
        self,
        hugging_face_token: str,
        embedding_match_threshold: float = 0.5,
        min_segment_duration: float = 2.0,
        device: Optional[str] = None
    ):
        """
        Initialize speaker diarizer.
        
        Args:
            hugging_face_token: HuggingFace token for PyAnnote models
            embedding_match_threshold: Cosine similarity threshold for speaker matching
            min_segment_duration: Minimum duration for segments to be considered
            device: Device to use (cuda/cpu), auto-detected if None
        """
        self.hugging_face_token = hugging_face_token
        self.embedding_match_threshold = embedding_match_threshold
        self.min_segment_duration = min_segment_duration
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize models
        self._init_models()
    
    def _init_models(self):
        """Initialize PyAnnote models for speaker identification."""
        try:
            logger.info("Loading embedding model for speaker identification")
            self.embedding_model = Model.from_pretrained(
                "pyannote/embedding",
                use_auth_token=self.hugging_face_token
            )
            self.embedding_model.to(self.device)
            self.embedding_inference = Inference(self.embedding_model, window="whole")
            
        except Exception as e:
            logger.error(f"Error initializing speaker models: {e}")
            raise
    
    def crop_waveform(self, waveform: torch.Tensor, sample_rate: int, segment: Segment) -> torch.Tensor:
        """Crop waveform to the specified segment."""
        start_sample = int(segment.start * sample_rate)
        end_sample = int(segment.end * sample_rate)
        return waveform[:, start_sample:end_sample]
    
    def get_reference_embeddings(self, speaker_samples: Dict[str, str]) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
        """
        Load and compute reference embeddings from speaker samples.
        
        Args:
            speaker_samples: Dict mapping speaker names to audio file paths
            
        Returns:
            Tuple of (embeddings dict, speaker names dict)
        """
        embeddings = {}
        speaker_names = {}
        
        for i, (name, path) in enumerate(speaker_samples.items()):
            try:
                # Load audio
                waveform, sample_rate = torchaudio.load(path)
                
                # Compute embedding
                embedding = self.embedding_inference({
                    "waveform": waveform,
                    "sample_rate": sample_rate
                }).reshape(1, -1)
                
                # Normalize
                embedding = normalize(embedding)
                
                embeddings[i] = embedding
                speaker_names[i] = name
                logger.info(f"Loaded reference embedding for '{name}' as ID {i}")
                
            except Exception as e:
                logger.warning(f"Error loading reference for '{name}': {e}")
        
        return embeddings, speaker_names
    
    def match_speaker_by_embedding(
        self,
        embedding: np.ndarray,
        reference_embeddings: Dict[int, np.ndarray],
        speaker_names: Dict[int, str]
    ) -> Optional[str]:
        """
        Match a speaker embedding to reference embeddings.
        
        Args:
            embedding: Speaker embedding to match
            reference_embeddings: Dict of reference embeddings
            speaker_names: Dict mapping embedding IDs to speaker names
            
        Returns:
            Best matching speaker name or None if no match above threshold
        """
        best_match = None
        highest_similarity = -1
        
        for ref_id, ref_embedding in reference_embeddings.items():
            similarity = cosine_similarity(embedding, ref_embedding)[0][0]
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = speaker_names[ref_id]
        
        logger.debug(f"Highest similarity with '{best_match}': {highest_similarity:.3f}")
        
        if highest_similarity >= self.embedding_match_threshold:
            return best_match
        else:
            return None
    
    def identify_speakers_in_diarization(
        self,
        diarization: Annotation,
        waveform: torch.Tensor,
        sample_rate: int,
        reference_embeddings: Dict[int, np.ndarray],
        speaker_names: Dict[int, str]
    ) -> Annotation:
        """
        Identify speakers in diarization using reference embeddings.
        
        Args:
            diarization: Original diarization annotation
            waveform: Audio waveform
            sample_rate: Audio sample rate
            reference_embeddings: Reference speaker embeddings
            speaker_names: Speaker name mapping
            
        Returns:
            Updated diarization with identified speakers
        """
        # Group segments by speaker
        segments_by_speaker = defaultdict(list)
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments_by_speaker[speaker].append(Segment(turn.start, turn.end))
        
        # Create new diarization with identified speakers
        identified_diarization = Annotation(uri=diarization.uri)
        
        for original_speaker, segments in segments_by_speaker.items():
            logger.info(f"Processing speaker '{original_speaker}' with {len(segments)} segments")
            
            valid_embeddings = []
            valid_segments = []
            
            # Process each segment for this speaker
            for segment in segments:
                duration = segment.end - segment.start
                
                # Skip very short segments
                if duration < self.min_segment_duration:
                    logger.debug(f"Skipping short segment: {segment} (duration: {duration:.2f}s)")
                    continue
                
                # Compute embedding for this segment
                try:
                    cropped = self.crop_waveform(waveform, sample_rate, segment)
                    embedding = self.embedding_inference({
                        "waveform": cropped,
                        "sample_rate": sample_rate
                    }).reshape(1, -1)
                    
                    embedding = normalize(embedding)
                    valid_embeddings.append(embedding)
                    valid_segments.append(segment)
                    
                except Exception as e:
                    logger.warning(f"Error computing embedding for segment {segment}: {e}")
            
            # Identify speaker if we have valid embeddings
            if valid_embeddings and reference_embeddings:
                # Average the embeddings for this speaker
                mean_embedding = np.mean(np.vstack(valid_embeddings), axis=0, keepdims=True)
                
                # Match against reference speakers
                identified_name = self.match_speaker_by_embedding(
                    mean_embedding, reference_embeddings, speaker_names
                )
                
                if identified_name:
                    logger.info(f"Speaker '{original_speaker}' identified as: {identified_name}")
                    new_speaker_label = identified_name
                else:
                    logger.info(f"Speaker '{original_speaker}' could not be identified, keeping original label")
                    new_speaker_label = original_speaker
            else:
                # No valid embeddings or no references, keep original
                new_speaker_label = original_speaker
                if not valid_segments:
                    logger.warning(f"No valid segments found for speaker '{original_speaker}'")
            
            # Add segments to new diarization
            for segment in valid_segments:
                identified_diarization[segment] = new_speaker_label
        
        return identified_diarization
    
    def remove_speaker_from_diarization(self, diarization: Annotation, speaker_to_remove: str) -> Annotation:
        """Remove a speaker from diarization annotation."""
        new_diarization = Annotation(uri=diarization.uri)
        
        for segment, track, speaker in diarization.itertracks(yield_label=True):
            if speaker != speaker_to_remove:
                new_diarization[segment, track] = speaker
        
        return new_diarization
    
    def rename_speaker_in_diarization(
        self,
        diarization: Annotation,
        old_label: str,
        new_label: str
    ) -> Annotation:
        """Rename a speaker in diarization annotation."""
        updated_diarization = Annotation(uri=diarization.uri)
        
        for segment, track, speaker in diarization.itertracks(yield_label=True):
            if speaker == old_label:
                updated_diarization[segment, track] = new_label
            else:
                updated_diarization[segment, track] = speaker
        
        return updated_diarization


def analyze_speaker_statistics(diarization: Annotation) -> Dict[str, any]:
    """
    Analyze speaker statistics from diarization.
    
    Args:
        diarization: Diarization annotation
        
    Returns:
        Dict containing speaker statistics
    """
    speaker_stats = defaultdict(lambda: {"duration": 0.0, "segments": 0})
    total_duration = 0.0
    
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        duration = segment.end - segment.start
        speaker_stats[speaker]["duration"] += duration
        speaker_stats[speaker]["segments"] += 1
        total_duration += duration
    
    # Calculate percentages
    for speaker in speaker_stats:
        speaker_stats[speaker]["percentage"] = (
            speaker_stats[speaker]["duration"] / total_duration * 100
            if total_duration > 0 else 0
        )
    
    return {
        "speakers": dict(speaker_stats),
        "total_duration": total_duration,
        "speaker_count": len(speaker_stats)
    }