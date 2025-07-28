#!/usr/bin/env python3
import re
import time
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MeetingRoastEngine:
    """The ridiculous meeting roast engine 🎭"""
    
    def __init__(self):
        # Buzzword patterns and responses
        self.buzzwords = {
            "synergy": [
                "Synergy count: {count}. Drink! 🍺",
                "We've achieved {count} synergies! The buzzword gods are pleased ⚡",
                "Synergy mentioned {count} times. I'm sensing some serious corporate energy 💼"
            ],
            "leverage": [
                "We're leveraging everything except clarity ({count} times) 🤔",
                "Leverage count: {count}. What are we, a construction company? 🏗️",
                "'Leverage' used {count} times. Someone's been reading business books 📚"
            ],
            "circle back": [
                "Circled back {count} times. We're in orbit now 🪐",
                "Circle back count: {count}. More circles than a geometry class ⭕",
                "We've circled back {count} times. I'm getting dizzy 🌀"
            ],
            "offline": [
                "'Take it offline' count: {count}. Nothing was taken offline. 📡",
                "Offline mentions: {count}. Everything's still very much online 💻",
                "Talked about going offline {count} times but here we still are 🤷"
            ],
            "actionable": [
                "Actionable mentioned {count} times. Very... actionable 🎯",
                "'Actionable' count: {count}. As opposed to inactionable? 🤨",
                "We love actionable things ({count} mentions). Action! ⚡"
            ],
            "bandwidth": [
                "Bandwidth mentioned {count} times. This isn't a network meeting 📡",
                "Human bandwidth: {count} mentions. We're not routers, people! 🤖",
                "Bandwidth count: {count}. Someone's been talking to IT 💾"
            ],
            "pivot": [
                "Pivot count: {count}. More turns than a dance class 💃",
                "We've pivoted {count} times. I'm getting motion sickness 🌀",
                "Pivot mentions: {count}. Are we playing basketball? 🏀"
            ],
            "deep dive": [
                "Deep dive count: {count}. Hope you brought scuba gear 🤿",
                "We're deep diving {count} times. That's a lot of diving 🏊",
                "Deep dive mentions: {count}. Ocean exploration meeting? 🌊"
            ],
            "low hanging fruit": [
                "Low hanging fruit: {count} mentions. Time to pick some apples? 🍎",
                "We love low hanging fruit ({count} times). Easy pickings! 🍒",
                "Fruit picking mentioned {count} times. Wrong meeting? 🍓"
            ],
            "touch base": [
                "Touch base count: {count}. This isn't baseball ⚾",
                "Touched base {count} times. Safe! 🏃‍♂️",
                "Base touching: {count} mentions. Home run! ⚾"
            ]
        }
        
        # Meeting anti-patterns
        self.anti_patterns = {
            "long_pause": "Awkward silence lasted {duration} seconds. New record! 🦗",
            "interrupt": "{person} interrupted {count} times. Going for gold! 🥇",
            "tangent": "Off-topic ratio: {percentage}%. Meeting or podcast? 🎙️",
            "no_agenda": "Meeting without agenda detected. Thoughts and prayers. 🙏",
            "over_time": "Meeting ran {minutes} minutes over. Time is a construct 🕰️",
            "camera_off": "{count} cameras off. Are we sure people are here? 👻",
            "mute_issues": "Mute button confusion: {count} incidents. Technology wins again 🔇"
        }
        
        # Meeting personality archetypes
        self.personalities = {
            "The Synergizer": {
                "description": "Uses buzzwords like punctuation",
                "triggers": ["synergy", "leverage", "actionable"],
                "threshold": 3
            },
            "The Rambler": {
                "description": "15-minute updates on 5-minute tasks",
                "triggers": ["long_speech"],
                "threshold": 1
            },
            "The Silent One": {
                "description": "Camera off, mic off, probably asleep",
                "triggers": ["no_speech"],
                "threshold": 1
            },
            "The Architect": {
                "description": "Every bug needs a complete redesign",
                "triggers": ["architecture", "redesign", "refactor"],
                "threshold": 2
            },
            "The Meeting Master": {
                "description": "Schedules meetings to discuss scheduling meetings",
                "triggers": ["schedule", "calendar", "follow up"],
                "threshold": 3
            }
        }
        
        # Tracking data
        self.reset_tracking()
    
    def reset_tracking(self):
        """Reset all tracking data for a new meeting"""
        self.buzzword_counts = defaultdict(int)
        self.speaker_stats = defaultdict(lambda: {
            'total_time': 0,
            'segments': 0,
            'interruptions': 0,
            'buzzwords': defaultdict(int),
            'longest_speech': 0
        })
        self.meeting_start = time.time()
        self.last_speech_end = 0
        self.silences = []
        self.personalities_detected = defaultdict(int)
        self.meeting_events = []
    
    def analyze_segment(self, segment: Dict) -> Dict[str, Any]:
        """Analyze a single transcript segment for roast-worthy content"""
        speaker = segment.get('speaker', 'Unknown')
        text = segment.get('text', '').lower()
        start_time = segment.get('start', 0)
        end_time = segment.get('end', 0)
        duration = end_time - start_time
        
        analysis = {
            'buzzwords_found': {},
            'personality_clues': [],
            'events': []
        }
        
        # Track buzzwords
        for buzzword in self.buzzwords.keys():
            count = len(re.findall(r'\b' + re.escape(buzzword.replace(' ', r'\s+')) + r'\b', text))
            if count > 0:
                self.buzzword_counts[buzzword] += count
                self.speaker_stats[speaker]['buzzwords'][buzzword] += count
                analysis['buzzwords_found'][buzzword] = count
        
        # Track speaker stats
        self.speaker_stats[speaker]['total_time'] += duration
        self.speaker_stats[speaker]['segments'] += 1
        
        if duration > self.speaker_stats[speaker]['longest_speech']:
            self.speaker_stats[speaker]['longest_speech'] = duration
        
        # Check for long speeches (potential rambler)
        if duration > 120:  # 2 minutes
            analysis['personality_clues'].append('long_speech')
            analysis['events'].append(f"{speaker} spoke for {duration:.1f} seconds straight!")
        
        # Check for silence before this segment
        if self.last_speech_end > 0:
            silence_duration = start_time - self.last_speech_end
            if silence_duration > 5:  # 5+ second pause
                self.silences.append(silence_duration)
                analysis['events'].append(f"Awkward silence: {silence_duration:.1f} seconds")
        
        self.last_speech_end = end_time
        
        # Update personality detection
        for personality, config in self.personalities.items():
            for trigger in config['triggers']:
                if trigger in analysis['personality_clues'] or any(trigger in text for trigger in config['triggers'] if isinstance(trigger, str)):
                    self.personalities_detected[personality] += 1
        
        return analysis
    
    def generate_roast_observations(self) -> List[str]:
        """Generate witty observations about the meeting"""
        observations = []
        
        # Buzzword roasts
        for buzzword, count in self.buzzword_counts.items():
            if count > 0:
                templates = self.buzzwords[buzzword]
                template = templates[count % len(templates)]
                observations.append(template.format(count=count))
        
        # Speaker-specific roasts
        if self.speaker_stats:
            # Find the biggest talker
            biggest_talker = max(self.speaker_stats.items(), key=lambda x: x[1]['total_time'])
            speaker_name, stats = biggest_talker
            
            total_meeting_time = time.time() - self.meeting_start
            talk_percentage = (stats['total_time'] / total_meeting_time) * 100
            
            if talk_percentage > 50:
                observations.append(f"{speaker_name} talked for {talk_percentage:.1f}% of the meeting. Impressive lung capacity! 🫁")
            
            # Longest single speech
            if stats['longest_speech'] > 180:  # 3 minutes
                observations.append(f"{speaker_name}'s longest monologue: {stats['longest_speech']:.1f} seconds. TED talk worthy! 🎤")
        
        # Silence roasts
        if self.silences:
            longest_silence = max(self.silences)
            if longest_silence > 10:
                observations.append(f"Longest awkward silence: {longest_silence:.1f} seconds. You could hear a pin drop 📌")
        
        # Meeting duration roasts
        meeting_duration = (time.time() - self.meeting_start) / 60
        if meeting_duration > 60:
            observations.append(f"Meeting lasted {meeting_duration:.1f} minutes. That's {meeting_duration/30:.1f} sitcom episodes! 📺")
        
        return observations
    
    def generate_personality_profiles(self) -> Dict[str, str]:
        """Generate personality profiles for meeting participants"""
        profiles = {}
        
        for speaker, stats in self.speaker_stats.items():
            profile_hints = []
            
            # Check buzzword usage
            total_buzzwords = sum(stats['buzzwords'].values())
            if total_buzzwords > 5:
                profile_hints.append("The Synergizer")
            
            # Check for long speeches
            if stats['longest_speech'] > 180:
                profile_hints.append("The Rambler")
            
            # Check for minimal participation
            if stats['segments'] < 3 and stats['total_time'] < 30:
                profile_hints.append("The Silent One")
            
            # Assign primary personality
            if profile_hints:
                primary = profile_hints[0]
                description = self.personalities.get(primary, {}).get('description', 'Mysterious meeting participant')
                profiles[speaker] = f"{primary}: {description}"
        
        return profiles
    
    def generate_meeting_report_card(self) -> Dict[str, Any]:
        """Generate a comprehensive meeting report card"""
        meeting_duration = (time.time() - self.meeting_start) / 60
        total_speakers = len(self.speaker_stats)
        total_buzzwords = sum(self.buzzword_counts.values())
        
        # Calculate scores
        efficiency_score = max(0, 100 - (meeting_duration * 2) - (total_buzzwords * 3))
        buzzword_density = total_buzzwords / meeting_duration if meeting_duration > 0 else 0
        
        # Assign letter grade
        if efficiency_score >= 90:
            grade = "A+ (Miracle meeting!)"
        elif efficiency_score >= 80:
            grade = "B+ (Pretty good!)"
        elif efficiency_score >= 70:
            grade = "C (Could be worse)"
        elif efficiency_score >= 60:
            grade = "D (Rough)"
        else:
            grade = "F (Thoughts and prayers)"
        
        # Generate awards
        awards = []
        
        if self.speaker_stats:
            # Biggest talker
            biggest_talker = max(self.speaker_stats.items(), key=lambda x: x[1]['total_time'])
            awards.append(f"🏆 Marathon Speaker: {biggest_talker[0]} ({biggest_talker[1]['total_time']:.0f} seconds)")
            
            # Buzzword champion
            buzzword_champion = max(
                self.speaker_stats.items(),
                key=lambda x: sum(x[1]['buzzwords'].values()),
                default=(None, {'buzzwords': {}})
            )
            if buzzword_champion[0] and sum(buzzword_champion[1]['buzzwords'].values()) > 0:
                awards.append(f"📢 Buzzword Champion: {buzzword_champion[0]} ({sum(buzzword_champion[1]['buzzwords'].values())} corporate clichés)")
            
            # Silent award
            quietest = min(self.speaker_stats.items(), key=lambda x: x[1]['segments'])
            if quietest[1]['segments'] <= 2:
                awards.append(f"👻 Ghost Award: {quietest[0]} (present but invisible)")
        
        if meeting_duration > 60:
            awards.append(f"⏰ Plot Twist Award: Meeting → {meeting_duration:.0f} minutes")
        
        return {
            "efficiency_score": f"{efficiency_score:.0f}/100",
            "grade": grade,
            "buzzword_density": f"{buzzword_density:.1f} per minute",
            "duration": f"{meeting_duration:.1f} minutes",
            "speakers": total_speakers,
            "awards": awards,
            "ai_recommendation": self._get_ai_recommendation(efficiency_score, meeting_duration, total_buzzwords)
        }
    
    def _get_ai_recommendation(self, efficiency_score: float, duration: float, buzzwords: int) -> str:
        """Generate AI recommendation based on meeting metrics"""
        if efficiency_score > 85:
            return "🎉 This meeting was actually productive! Consider it a template."
        elif duration > 60 and buzzwords > 20:
            return "💡 This could have been an email... a very long email."
        elif buzzwords > 15:
            return "📝 Consider a buzzword jar. Fund the next team lunch!"
        elif duration > 90:
            return "⏰ Next time, try the 'one breath rule' - if you can't say it in one breath, it's two meetings."
        else:
            return "🤖 Meeting analysis complete. Results are... interesting."
    
    def get_live_roast_update(self) -> str:
        """Get a quick roast update for live meetings"""
        observations = []
        
        # Quick buzzword check
        recent_buzzwords = [(k, v) for k, v in self.buzzword_counts.items() if v > 0]
        if recent_buzzwords:
            top_buzzword = max(recent_buzzwords, key=lambda x: x[1])
            observations.append(f"🚨 {top_buzzword[0].title()} alert! Count: {top_buzzword[1]}")
        
        # Quick speaker check
        if self.speaker_stats:
            current_time = time.time()
            meeting_duration = (current_time - self.meeting_start) / 60
            
            if meeting_duration > 30:  # After 30 minutes
                biggest_talker = max(self.speaker_stats.items(), key=lambda x: x[1]['total_time'])
                talk_percentage = (biggest_talker[1]['total_time'] / (meeting_duration * 60)) * 100
                
                if talk_percentage > 60:
                    observations.append(f"📊 {biggest_talker[0]} is dominating at {talk_percentage:.0f}% talk time!")
        
        return " | ".join(observations) if observations else "Meeting is surprisingly well-behaved... for now 🤔"

# Example usage
if __name__ == "__main__":
    # Demo the roast engine
    engine = MeetingRoastEngine()
    
    # Simulate meeting segments
    demo_segments = [
        {"speaker": "Alice", "text": "Let's leverage our synergy to create actionable insights", "start": 0, "end": 5},
        {"speaker": "Bob", "text": "I think we need to circle back on this and take it offline", "start": 10, "end": 15},
        {"speaker": "Alice", "text": "Great idea! We should definitely leverage this opportunity for maximum synergy", "start": 20, "end": 25},
        {"speaker": "Charlie", "text": "...", "start": 30, "end": 31},  # Silent Charlie
    ]
    
    for segment in demo_segments:
        analysis = engine.analyze_segment(segment)
        print(f"Segment analysis: {analysis}")
    
    # Generate roast
    observations = engine.generate_roast_observations()
    print(f"\nRoast observations: {observations}")
    
    profiles = engine.generate_personality_profiles()
    print(f"\nPersonality profiles: {profiles}")
    
    report_card = engine.generate_meeting_report_card()
    print(f"\nReport card: {report_card}")