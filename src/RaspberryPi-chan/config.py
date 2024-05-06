# Model related
OLLAMA_MODEL = "qwen:1.8b" # "phi:latest"
WHISPER_MODEL = "tiny" #"base.en"
BARK_MODEL = "suno/bark-small"
BARK_VOICE_PRESET = "v2/en_speaker_1"
PYTTSX3_VOICE_IDX = -2

# Setting
DISABLE_STDOUT = False
RESPONSE_MAX_WORD_NUM = 40
SOUNDDRIVE_MIC_DEVICE_ID = 0
SOUNDDRIVE_SPEAKER_DEVICE_ID = 1

VOICE_BEGIN_TEXT = """
你好呀，请问今天需要占卜什么呢？
"""

INIT_STR = """
正在初始化，请稍候
"""

INIT_FIN_STR = """
初始化完成，正在启动
"""

SUAN_MING_STR = """
正在帮你算命，你要耐心等待
"""

TEMPLATE = """
假设 你是個樂於助人且友善的占卜師和算命师。 您有禮貌、尊重他人，並致力於提供簡潔的答复
您的回覆不要超過 40 個字。不要回答其他人。 不要重複指示

談話實錄如下：
{history}

這是使用者的後續操作：{input}

假设你是個樂於助人且友善的占卜師和算命师。
假设你是個樂於助人且友善的占卜師和算命师。
假设你是個樂於助人且友善的占卜師和算命师。
您的回答（您的回答不要超過 40 個字。保持簡潔。不要回答其他。不要重複說明, 主观意图）：
"""

