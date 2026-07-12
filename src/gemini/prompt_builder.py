"""
src/gemini/prompt_builder.py
==============================
Dynamic prompt construction for Gemini API calls.
Builds context-rich, emotion-aware prompts for personalized guidance.
"""

from config.emotions import EMOTION_META, GUIDANCE_TONE
from src.utils.helpers import truncate_text


def build_guidance_prompt(
    student_text:   str,
    final_emotion:  str,
    confidence:     float,
    is_mixed:       bool,
    mixed_label:    str = "",
    bilstm_emotion: str = "",
    bert_emotion:   str = "",
) -> str:
    """
    Build a detailed, structured Gemini prompt for personalized learning guidance.

    Args:
        student_text:   The student's raw input text.
        final_emotion:  Detected primary emotion (e.g., "frustrated").
        confidence:     Confidence score (0–1) for the final emotion.
        is_mixed:       Whether mixed emotions were detected.
        mixed_label:    Human-readable mixed label (e.g., "Frustrated + Confused").
        bilstm_emotion: BiLSTM's predicted emotion.
        bert_emotion:   BERT's predicted emotion.

    Returns:
        Complete prompt string to send to Gemini.
    """
    emotion_meta = EMOTION_META.get(final_emotion, {})
    label        = emotion_meta.get("label", final_emotion.capitalize())
    emoji        = emotion_meta.get("emoji", "🎓")
    tone         = GUIDANCE_TONE.get(final_emotion, "be supportive and helpful.")

    # Truncate student text for prompt (avoid token limit issues)
    truncated_text = truncate_text(student_text, max_chars=400)

    # Mixed emotion context
    mixed_context = ""
    if is_mixed:
        mixed_context = (
            f"\n**Note:** The student shows MIXED emotions: {mixed_label}. "
            f"Address both emotional states in your guidance."
        )

    # Model agreement context
    model_context = ""
    if bilstm_emotion and bert_emotion:
        if bilstm_emotion == bert_emotion:
            model_context = f"(Both AI models agree on this emotion detection.)"
        else:
            model_context = (
                f"(Models had different views — BiLSTM says '{bilstm_emotion.capitalize()}', "
                f"BERT says '{bert_emotion.capitalize()}'. Final decision based on ensemble.)"
            )

    prompt = f"""You are an expert educational psychologist and personalized learning coach.
A student has described their study situation, and our AI system has detected their emotional state.

---
**STUDENT'S WORDS:**
"{truncated_text}"

---
**EMOTIONAL STATE DETECTED:**
- Primary Emotion: {emoji} {label} (Confidence: {confidence:.0%})
{mixed_context}
{model_context}

---
**YOUR TASK:**
Generate compassionate, actionable, and personalized learning guidance for this student.
Your response should {tone}

**REQUIRED SECTIONS (use these exact headers with emojis):**

## 💭 Understanding Your Feeling
Briefly validate and normalize the student's emotion (2-3 sentences). Make them feel heard.

## 🎯 Immediate Action Steps
Provide 3-4 specific, concrete steps the student can take RIGHT NOW to address their situation.
Format as a numbered list.

## 📚 Learning Strategy
Suggest 2-3 study techniques or approaches tailored to their emotional state and the topic they seem to be working on.

## 💡 Motivational Insight
Share a brief, genuine motivational message or perspective shift (2-3 sentences). Avoid generic clichés.

## 🔗 Helpful Resources
Suggest 2-3 types of resources (e.g., Khan Academy, YouTube, study groups, etc.) relevant to their situation.

---
**GUIDELINES:**
- Be warm, empathetic, and encouraging (not patronizing)
- Keep each section concise and actionable
- Use simple, clear language appropriate for a student
- Total response should be 200-350 words
- Format using markdown for readability
"""

    return prompt


def build_quick_tip_prompt(emotion: str, topic_hint: str = "") -> str:
    """
    Build a shorter prompt for a quick tip card (used in analytics page).

    Args:
        emotion:    Detected emotion string.
        topic_hint: Optional subject/topic from input text.

    Returns:
        Short prompt for a quick study tip.
    """
    label = EMOTION_META.get(emotion, {}).get("label", emotion.capitalize())
    topic_part = f"about {topic_hint}" if topic_hint else ""

    return (
        f"A student feels {label} while studying {topic_part}. "
        f"Give them ONE powerful, specific study tip in 2-3 sentences. "
        f"Be encouraging and practical. No headers needed."
    )
