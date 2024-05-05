import config

def limit_words(text, max_words=config.RESPONSE_MAX_WORD_NUM):
    words = text.split()  # Split the text into words
    if len(words) > max_words:
        words = words[:max_words]  # Take only the first 20 words
    return ' '.join(words)