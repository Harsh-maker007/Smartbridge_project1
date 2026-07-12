"""
config/emotions.py
==================
Emotion definitions, keyword dictionaries for rule-based matching,
color mappings for the Streamlit UI, and emoji representations.
"""

# ── Emotion Metadata ──────────────────────────────────────────
EMOTION_META = {
    "bored": {
        "label":       "Bored",
        "emoji":       "😴",
        "color_hex":   "#6B7280",
        "color_rgb":   (107, 114, 128),
        "description": "The student shows disengagement or lack of interest.",
        "icon":        "🌑",
    },
    "confident": {
        "label":       "Confident",
        "emoji":       "😊",
        "color_hex":   "#10B981",
        "color_rgb":   (16, 185, 129),
        "description": "The student feels assured and capable.",
        "icon":        "⭐",
    },
    "confused": {
        "label":       "Confused",
        "emoji":       "😕",
        "color_hex":   "#F59E0B",
        "color_rgb":   (245, 158, 11),
        "description": "The student is uncertain or lost about the topic.",
        "icon":        "❓",
    },
    "curious": {
        "label":       "Curious",
        "emoji":       "🤔",
        "color_hex":   "#3B82F6",
        "color_rgb":   (59, 130, 246),
        "description": "The student is engaged and wants to explore further.",
        "icon":        "💡",
    },
    "frustrated": {
        "label":       "Frustrated",
        "emoji":       "😤",
        "color_hex":   "#EF4444",
        "color_rgb":   (239, 68, 68),
        "description": "The student is stressed, blocked, or overwhelmed.",
        "icon":        "🔥",
    },
}

# ── Rule-Based Keyword Dictionary ────────────────────────────
# Each emotion maps to a list of trigger phrases/keywords.
# Used by src/detection/rule_based.py to boost model confidence.
EMOTION_KEYWORDS = {
    "bored": [
        "boring", "bored", "dull", "not interesting", "uninteresting",
        "tedious", "pointless", "useless", "don't care", "don't see the point",
        "unmotivated", "lazy", "no interest", "not engaging", "lifeless",
        "what's the point", "waste of time", "sleepy", "mind wandering",
        "can't focus", "zoned out", "checked out", "disengaged", "monotonous",
    ],
    "confident": [
        "got it", "understand", "makes sense", "clear", "i know",
        "sure", "confident", "solved", "figured out", "mastered",
        "easy", "simple", "no problem", "i can do this", "ready",
        "good at", "comfortable", "prepared", "strong", "excellent",
        "great", "i think i understand", "finally understand",
    ],
    "confused": [
        "confused", "don't understand", "not sure", "unclear",
        "what does", "what is", "how does", "doesn't make sense",
        "lost", "no idea", "can't follow", "hard to understand",
        "makes no sense", "complicated", "complex", "overwhelmed",
        "mixed up", "unsure", "don't know", "can't figure out",
        "stuck on", "help me understand", "explain", "what exactly",
    ],
    "curious": [
        "wonder", "curious", "interesting", "how does", "why does",
        "want to know", "want to learn", "can you explain", "tell me more",
        "fascinating", "intriguing", "want to explore", "excited to learn",
        "what if", "how about", "is it possible", "i want to try",
        "deeper understanding", "more about this", "what happens when",
    ],
    "frustrated": [
        "frustrated", "frustrated with", "annoyed", "angry", "mad",
        "can't solve", "can't figure", "stuck", "give up", "hopeless",
        "impossible", "too hard", "too difficult", "no matter what",
        "nothing works", "keeps failing", "failing again", "errors",
        "why isn't", "still not working", "tried everything",
        "so hard", "pulling my hair", "want to quit", "hate this",
        "doesn't work", "broken", "wrong", "keeps happening",
    ],
}

# ── Guidance Tone per Emotion ─────────────────────────────────
# Instructions for Gemini on how to frame the response per emotion.
GUIDANCE_TONE = {
    "bored": (
        "energize and re-engage the student. Use exciting real-world examples, "
        "gamification ideas, and connect the topic to things they care about."
    ),
    "confident": (
        "celebrate their progress, present advanced challenges and extension "
        "activities to deepen mastery and maintain momentum."
    ),
    "confused": (
        "be patient, clear, and step-by-step. Break down concepts into simple "
        "analogies, provide worked examples, and recommend clarifying resources."
    ),
    "curious": (
        "fuel their curiosity! Provide deeper insights, related topics, "
        "interesting rabbit holes, project ideas, and advanced resources."
    ),
    "frustrated": (
        "be empathetic and supportive first. Validate their effort, suggest "
        "a short break, then offer a simpler re-entry point or alternative approach."
    ),
}

# ── Plotly Color Sequence (for charts) ───────────────────────
CHART_COLORS = {
    "bored":      "#6B7280",
    "confident":  "#10B981",
    "confused":   "#F59E0B",
    "curious":    "#3B82F6",
    "frustrated": "#EF4444",
}

# ── Ordered Labels (for model output alignment) ───────────────
ORDERED_LABELS = ["bored", "confident", "confused", "curious", "frustrated"]
