# Model related
OLLAMA_MODEL = "qwen:1.8b" # "phi:latest"
WHISPER_MODEL = "base" #"base.en"
BARK_MODEL = "suno/bark-small"
BARK_VOICE_PRESET = "v2/en_speaker_1"
PYTTSX3_VOICE_IDX = -2

# Setting
DISABLE_STDOUT = False
RESPONSE_MAX_WORD_NUM = 40
SOUNDDRIVE_DEVICE_ID = 0

VOICE_BEGIN_TEXT = """
你好呀，请问今天需要占卜什么呢？
"""

TEMPLATE = """
You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less 
than 40 words. DO NOT exceed 40 words for your response. Keep Conise. DO NOT ANSWER OTHERS. DO NOT REPEAT INSTRUTS

The conversation transcript is as follows:
{history}

And here is the user's follow-up: {input}

Your response (DO NOT exceed 40 words for your response. Keep Conise. DO NOT ANSWER OTHERS. DO NOT REPEAT INSTRUTS):
"""

