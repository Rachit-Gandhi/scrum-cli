#!/usr/bin/env python3
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import uuid

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

logger = logging.getLogger(__name__)

class MeetingMemoryStore:
    
    def __init__(self, data_dir: str = "meeting_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize ChromaDB if available
        if CHROMADB_AVAILABLE:
            self._init_chromadb()
        else:
            logger.warning("ChromaDB not available - using JSON fallback storage")
            self.client = None
            self.collection = None
        
        # Fallback JSON storage
        self.meetings_file = self.data_dir / "meetings.json"
        self.transcripts_dir = self.data_dir / "transcripts"
        self.transcripts_dir.mkdir(exist_ok=True)
        
        # Load existing meetings index
        self.meetings_index = self._load_meetings_index()
    
    def _init_chromadb(self):
        try:
            # Create persistent client
            db_path = str(self.data_dir / "chromadb")
            self.client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="meeting_transcripts",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB initialized with {self.collection.count()} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.client = None
            self.collection = None
    
    def _load_meetings_index(self) -> Dict:
        if self.meetings_file.exists():
            try:
                with open(self.meetings_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading meetings index: {e}")
        
        return {"meetings": []}
    
    def _save_meetings_index(self):
        try:
            with open(self.meetings_file, 'w') as f:
                json.dump(self.meetings_index, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving meetings index: {e}")
    
    def store_meeting(self, title: str, transcript_data: Dict[str, Any], metadata: Optional[Dict] = None) -> str:
        # Generate meeting ID
        meeting_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Prepare meeting record
        meeting_record = {
            "id": meeting_id,
            "title": title,
            "timestamp": timestamp,
            "metadata": metadata or {},
            "transcript_file": f"{meeting_id}.json"
        }
        
        # Add to meetings index
        self.meetings_index["meetings"].append(meeting_record)
        self._save_meetings_index()
        
        # Save full transcript data
        transcript_file = self.transcripts_dir / f"{meeting_id}.json"
        try:
            with open(transcript_file, 'w') as f:
                json.dump({
                    "meeting_id": meeting_id,
                    "title": title,
                    "timestamp": timestamp.isoformat(),
                    "metadata": metadata or {},
                    "transcript": transcript_data
                }, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving transcript file: {e}")
            return meeting_id
        
        # Store in vector database if available
        if self.collection:
            self._store_in_vector_db(meeting_id, title, transcript_data, metadata)
        
        logger.info(f"Meeting stored: {title} (ID: {meeting_id})")
        return meeting_id
    
    def _store_in_vector_db(self, meeting_id: str, title: str, transcript_data: Dict, metadata: Optional[Dict]):
        try:
            segments = transcript_data.get("transcription", {}).get("segments", [])
            
            # Prepare documents for vector storage
            documents = []
            metadatas = []
            ids = []
            
            for i, segment in enumerate(segments):
                # Create document text
                text = segment.get("text", "").strip()
                if not text:
                    continue
                
                speaker = segment.get("speaker", "Unknown")
                start_time = segment.get("start", 0)
                
                documents.append(text)
                
                # Metadata for this segment
                segment_metadata = {
                    "meeting_id": meeting_id,
                    "meeting_title": title,
                    "speaker": speaker,
                    "start_time": start_time,
                    "segment_index": i,
                    "timestamp": datetime.now().isoformat()
                }
                
                if metadata:
                    for key, value in metadata.items():
                        if isinstance(value, list):
                            segment_metadata[key] = str(value)
                        else:
                            segment_metadata[key] = value
                
                metadatas.append(segment_metadata)
                ids.append(f"{meeting_id}_segment_{i}")
            
            # Add to ChromaDB collection
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.info(f"Added {len(documents)} segments to vector store")
        
        except Exception as e:
            logger.error(f"Error storing in vector DB: {e}")
    
    def search_meetings(self, query: str, limit: int = 10, meeting_ids: Optional[List[str]] = None) -> List[Dict]:
        if not self.collection:
            # Fallback to simple text search
            return self._fallback_search(query, limit, meeting_ids)
        
        try:
            # Prepare where clause for filtering
            where_clause = {}
            if meeting_ids:
                where_clause["meeting_id"] = {"$in": meeting_ids}
            
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_clause if where_clause else None
            )
            
            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    formatted_results.append({
                        "text": doc,
                        "metadata": metadata,
                        "relevance_score": 1 - distance,  # Convert distance to similarity
                        "rank": i + 1
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return self._fallback_search(query, limit, meeting_ids)
    
    def _fallback_search(self, query: str, limit: int, meeting_ids: Optional[List[str]] = None) -> List[Dict]:
        results = []
        query_lower = query.lower()
        
        # Search through stored transcripts
        for meeting in self.meetings_index["meetings"]:
            if meeting_ids and meeting["id"] not in meeting_ids:
                continue
            
            transcript_file = self.transcripts_dir / meeting["transcript_file"]
            if not transcript_file.exists():
                continue
            
            try:
                with open(transcript_file, 'r') as f:
                    data = json.load(f)
                
                segments = data.get("transcript", {}).get("transcription", {}).get("segments", [])
                
                for i, segment in enumerate(segments):
                    text = segment.get("text", "").strip()
                    if query_lower in text.lower():
                        results.append({
                            "text": text,
                            "metadata": {
                                "meeting_id": meeting["id"],
                                "meeting_title": meeting["title"],
                                "speaker": segment.get("speaker", "Unknown"),
                                "start_time": segment.get("start", 0),
                                "segment_index": i
                            },
                            "relevance_score": 0.5,  # Simple relevance
                            "rank": len(results) + 1
                        })
                
                if len(results) >= limit:
                    break
            except Exception as e:
                logger.error(f"Error searching transcript {transcript_file}: {e}")
        
        return results[:limit]
    
    def get_meeting(self, meeting_id: str) -> Optional[Dict]:
        # Find meeting in index
        meeting_record = None
        for meeting in self.meetings_index["meetings"]:
            if meeting["id"] == meeting_id:
                meeting_record = meeting
                break
        
        if not meeting_record:
            return None
        
        # Load full transcript
        transcript_file = self.transcripts_dir / meeting_record["transcript_file"]
        if not transcript_file.exists():
            logger.error(f"Transcript file not found: {transcript_file}")
            return meeting_record
        
        try:
            with open(transcript_file, 'r') as f:
                full_data = json.load(f)
            return full_data
        except Exception as e:
            logger.error(f"Error loading meeting {meeting_id}: {e}")
            return meeting_record
    
    def list_meetings(self, limit: int = 50) -> List[Dict]:
        meetings = sorted(
            self.meetings_index["meetings"],
            key=lambda x: x["timestamp"],
            reverse=True
        )
        return meetings[:limit]
    
    def delete_meeting(self, meeting_id: str) -> bool:
        try:
            # Remove from index
            original_count = len(self.meetings_index["meetings"])
            self.meetings_index["meetings"] = [
                m for m in self.meetings_index["meetings"] 
                if m["id"] != meeting_id
            ]
            
            if len(self.meetings_index["meetings"]) == original_count:
                logger.warning(f"Meeting {meeting_id} not found in index")
                return False
            
            self._save_meetings_index()
            
            # Delete transcript file
            for meeting in self.meetings_index["meetings"]:
                if meeting["id"] == meeting_id:
                    transcript_file = self.transcripts_dir / meeting["transcript_file"]
                    if transcript_file.exists():
                        transcript_file.unlink()
                    break
            
            # Delete from vector store
            if self.collection:
                # Get all segment IDs for this meeting
                results = self.collection.get(
                    where={"meeting_id": meeting_id}
                )
                
                if results["ids"]:
                    self.collection.delete(ids=results["ids"])
                    logger.info(f"Deleted {len(results['ids'])} segments from vector store")
            
            logger.info(f"Meeting {meeting_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting meeting {meeting_id}: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        stats = {
            "total_meetings": len(self.meetings_index["meetings"]),
            "storage_path": str(self.data_dir),
            "chromadb_available": CHROMADB_AVAILABLE,
            "vector_documents": 0
        }
        
        if self.collection:
            try:
                stats["vector_documents"] = self.collection.count()
            except Exception as e:
                logger.error(f"Error getting vector count: {e}")
        
        return stats

def create_memory_store(data_dir: str = "meeting_data") -> MeetingMemoryStore:
    return MeetingMemoryStore(data_dir)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    store = create_memory_store("test_data")
    
    mock_transcript = {
        "transcription": {
            "segments": [
                {"start": 0, "end": 5, "speaker": "Alice", "text": "Let's start the daily standup"},
                {"start": 5, "end": 10, "speaker": "Bob", "text": "I finished the authentication feature"},
                {"start": 10, "end": 15, "speaker": "Alice", "text": "Great work! What's next?"}
            ],
            "speakers": ["Alice", "Bob"],
            "speaker_count": 2
        }
    }
    
    meeting_id = store.store_meeting("Daily Standup", mock_transcript)
    print(f"Stored meeting: {meeting_id}")
    
    # Search
    results = store.search_meetings("authentication")
    print(f"Search results: {len(results)}")
    for result in results:
        print(f"- {result['text']} (Score: {result['relevance_score']:.2f})")
    
    stats = store.get_statistics()
    print(f"Statistics: {stats}")