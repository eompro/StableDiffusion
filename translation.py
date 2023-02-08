from deep_translator import GoogleTranslator 

def google_translate(kor_text):
    return GoogleTranslator(source = 'ko', target = 'en').translate(kor_text)