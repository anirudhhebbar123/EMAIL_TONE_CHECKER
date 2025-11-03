from flask import Flask, render_template, request, session, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import imaplib
import email
from email.header import decode_header
import os
import re
import json
from openai import OpenAI

# Initialize Flask app with explicit template and static folders
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production-for-local-use')

# Initialize analyzers
vader_analyzer = SentimentIntensityAnalyzer()
sentiment_pipeline = None
toxicity_pipeline = None
gen_pipeline = None

# Initialize OpenAI client for free GPT models
# Using Hugging Face's free inference API or OpenRouter for free GPT access
openai_client = None

def get_openai_client():
    """Initialize OpenAI client with free API alternatives"""
    global openai_client
    if openai_client is not None:
        return openai_client
    
    try:
        # Option 1: Use OpenRouter (supports multiple free models)
        # Get your free API key from https://openrouter.ai/
        api_key = os.environ.get('OPENROUTER_API_KEY', '')
        if api_key:
            openai_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
            return openai_client
        
        # Option 2: Use Hugging Face Inference API (free)
        # Get your token from https://huggingface.co/settings/tokens
        hf_token = os.environ.get('HUGGINGFACE_TOKEN', '')
        if hf_token:
            openai_client = OpenAI(
                base_url="https://api-inference.huggingface.co/v1/",
                api_key=hf_token
            )
            return openai_client
            
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        openai_client = None
    
    return None

def get_generation_pipeline():
    """Lazily load a text-generation pipeline. Returns None on failure.
    Model can be overridden with GEN_MODEL env var (default: google/flan-t5-small).
    """
    global gen_pipeline
    if gen_pipeline is not None:
        return gen_pipeline
    try:
        model_name = os.environ.get('GEN_MODEL', 'google/flan-t5-small')
        gen_pipeline = pipeline('text2text-generation', model=model_name, tokenizer=model_name)
        return gen_pipeline
    except Exception:
        gen_pipeline = None
        return None

def get_pipelines():
    global sentiment_pipeline, toxicity_pipeline
    if sentiment_pipeline is None:
        sentiment_pipeline = pipeline(
            'text-classification',
            model='cardiffnlp/twitter-roberta-base-sentiment-latest',
            tokenizer='cardiffnlp/twitter-roberta-base-sentiment-latest',
            top_k=None
        )
    if toxicity_pipeline is None:
        toxicity_pipeline = pipeline(
            'text-classification',
            model='unitary/toxic-bert',
            tokenizer='unitary/toxic-bert',
            top_k=None
        )
    return sentiment_pipeline, toxicity_pipeline

# Tone detection keywords
RUDE_KEYWORDS = ['stupid', 'idiot', 'idot', 'fool', 'crazy', 'horrible', 'terrible', 'hate', 
                 'damn', 'hell', 'what the', 'ridiculous', 'absurd', 'pathetic', 'useless',
                 'moron', 'dumb', 'dummy']
FORMAL_KEYWORDS = ['respectfully', 'sincerely', 'regards', 'appreciate', 'kindly', 
                   'regarding', 'pursuant', 'herein', 'whereas', 'aforementioned']
FRIENDLY_KEYWORDS = ['hello', 'hi', 'hey', 'thanks', 'thank you', 'please', 'great', 
                     'awesome', 'love', 'happy', 'excited', 'wonderful', 'cheers']

# Aggressive/imperative patterns that suggest rudeness
AGGRESSIVE_PATTERNS = [
    r'\basap\b',
    r'\bnow\b.*\b(send|do|give|get)',
    r'\bsend\b.*\basap\b',
    r'^(hey|hi|hello)\s+(idiot|stupid|moron|dumb)',
    r'\bdo\s+it\s+now\b',
    r'\b(just|only)\s+(send|do|give)',
]

def truncate_text_for_model(text, max_chars=300):
    """Truncate text to safe length for transformer models (avoid token limit errors)."""
    if not text:
        return text
    
    text = text.strip()
    if len(text) <= max_chars:
        return text
    
    truncated = text[:max_chars]
    
    last_period = truncated.rfind('.')
    last_exclamation = truncated.rfind('!')
    last_question = truncated.rfind('?')
    last_newline = truncated.rfind('\n')
    
    last_boundary = max(last_period, last_exclamation, last_question, last_newline)
    
    if last_boundary > max_chars * 0.7:
        truncated = truncated[:last_boundary + 1]
    
    return truncated.strip()

def classify_tone(text):
    """Classify tone using transformer models (sentiment + toxicity) and keyword analysis."""
    text = (text or '').strip()
    if not text:
        return 'formal', 0.5, {'compound': 0.0}

    text_lower = text.lower()
    
    text_for_models = truncate_text_for_model(text, max_chars=400)

    rude_keyword_count = sum(1 for kw in RUDE_KEYWORDS if kw in text_lower)
    formal_keyword_count = sum(1 for kw in FORMAL_KEYWORDS if kw in text_lower)
    friendly_keyword_count = sum(1 for kw in FRIENDLY_KEYWORDS if kw in text_lower)
    
    aggressive_pattern_count = sum(1 for pattern in AGGRESSIVE_PATTERNS if re.search(pattern, text_lower, re.IGNORECASE))
    
    hey_used_rudely = False
    if 'hey' in text_lower:
        hey_pattern = re.search(r'\bhey\b\s+(\w+)', text_lower)
        if hey_pattern:
            next_word = hey_pattern.group(1)
            if next_word in ['idiot', 'idot', 'stupid', 'moron', 'dumb']:
                hey_used_rudely = True
            elif re.search(r'\bhey\b.*\b(send|do|give|get)\b', text_lower) and 'please' not in text_lower:
                hey_used_rudely = True

    vader_scores = vader_analyzer.polarity_scores(text)
    compound_score = vader_scores.get('compound', 0.0)

    sent_pipe, tox_pipe = get_pipelines()

    try:
        sent_result = sent_pipe(text_for_models)[0]
        sent_scores = {item['label'].lower(): float(item['score']) for item in sent_result}
        pos = sent_scores.get('positive', 0.0)
        neg = sent_scores.get('negative', 0.0)
        neu = sent_scores.get('neutral', 0.0)
    except Exception as e:
        pos = max(0.0, compound_score) if compound_score > 0.1 else 0.0
        neg = max(0.0, -compound_score) if compound_score < -0.1 else 0.0
        neu = max(0.0, 1.0 - pos - neg)

    try:
        tox_result = tox_pipe(text_for_models)[0]
        tox_scores = {item['label'].lower(): float(item['score']) for item in tox_result}
        toxic_score = tox_scores.get('toxic', tox_scores.get('toxic/other', 0.0)) or 0.0
    except Exception as e:
        toxic_score = 0.3 if rude_keyword_count > 0 or aggressive_pattern_count > 0 else 0.0

    keyword_rude_boost = min(0.3, rude_keyword_count * 0.15)
    aggressive_boost = min(0.4, aggressive_pattern_count * 0.2)
    if hey_used_rudely:
        aggressive_boost = max(aggressive_boost, 0.35)
    
    keyword_friendly_boost = min(0.2, friendly_keyword_count * 0.1)
    keyword_formal_boost = min(0.2, formal_keyword_count * 0.1)

    rude_score = max(toxic_score, neg) + keyword_rude_boost + aggressive_boost
    if (toxic_score >= 0.4 or 
        (neg > 0.55 and compound_score < -0.2) or 
        (rude_keyword_count > 0 and neg > 0.4) or
        aggressive_pattern_count > 0 or
        hey_used_rudely or
        (rude_keyword_count > 0 and aggressive_pattern_count > 0)):
        tone = 'rude'
        confidence = min(0.98, 0.55 + rude_score * 0.4)
    elif not hey_used_rudely and ((pos >= max(neg, neu) and pos > 0.4) or (friendly_keyword_count > 0 and pos > 0.35) or compound_score > 0.2):
        tone = 'friendly'
        friendly_score = pos + keyword_friendly_boost + max(0, compound_score * 0.5)
        confidence = min(0.95, 0.5 + friendly_score * 0.4)
    else:
        tone = 'formal'
        formal_score = neu + keyword_formal_boost + (1.0 - abs(compound_score) * 0.5)
        confidence = max(0.6, 0.5 + formal_score * 0.3)

    return tone, round(confidence, 3), {'compound': compound_score, 'toxic': toxic_score, 'pos': pos, 'neg': neg, 'neu': neu}

def generate_gpt_suggestions(original_text):
    """Use free GPT models to generate polite rewrites for rude emails."""
    client = get_openai_client()
    
    if client is None:
        return None
    
    try:
        # Prepare the prompt
        prompt = f"""You are a professional email tone advisor. Rewrite the following rude or unprofessional email into two polite versions:

1. A friendly, warm version
2. A formal, professional version

CRITICAL RULES:
- Remove ALL profanity, insults, and offensive language
- Replace aggressive commands with polite requests
- Add proper greetings and closings
- Maintain the core message while improving tone

Original Email:
{original_text}

Return your response in this exact JSON format:
{{
  "friendly": "the friendly rewrite here",
  "formal": "the formal rewrite here"
}}"""

        # Try OpenRouter with free models first
        models_to_try = [
            "meta-llama/llama-3.2-3b-instruct:free",  # Free Llama model
            "qwen/qwen-2-7b-instruct:free",  # Free Qwen model
            "microsoft/phi-3-mini-128k-instruct:free",  # Free Phi model
        ]
        
        for model in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful email tone advisor. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                
                content = response.choices[0].message.content.strip()
                
                # Try to parse JSON from response
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    result = json.loads(json_match.group(0))
                    if result.get('friendly') and result.get('formal'):
                        return result
                        
            except Exception as e:
                print(f"Model {model} failed: {e}")
                continue
        
        return None
        
    except Exception as e:
        print(f"GPT generation failed: {e}")
        return None

def suggest_polite_rewrite(original_text, tone):
    """Suggest polite alternatives using GPT for rude emails, fallback to rule-based."""
    text = (original_text or '').strip()
    if not text:
        return [{
            'original': original_text,
            'rewritten': original_text,
            'change': 'No content provided'
        }]

    suggestions = []
    
    # Use GPT for rude emails
    if tone == 'rude':
        gpt_result = generate_gpt_suggestions(text)
        
        if gpt_result:
            friendly = gpt_result.get('friendly', '').strip()
            formal = gpt_result.get('formal', '').strip()
            
            if friendly:
                suggestions.append({
                    'original': original_text,
                    'rewritten': friendly,
                    'change': 'AI-powered friendly rewrite'
                })
            if formal:
                suggestions.append({
                    'original': original_text,
                    'rewritten': formal,
                    'change': 'AI-powered formal rewrite'
                })
            
            if suggestions:
                return suggestions
    
    # Fallback to rule-based for all cases
    rule_suggestions = create_rule_based_rewrites(original_text, tone)
    if rule_suggestions:
        return rule_suggestions
    
    # Final fallback
    redacted_body = 'I have some concerns regarding parts of the message and would like to discuss constructive next steps.'
    friendly_fb = 'Hi,\n\n' + redacted_body + '\n\nCould we align on next steps?\n\nThanks,'
    formal_fb = 'Hello,\n\n' + redacted_body + '\n\nPlease let me know how we can move forward.\n\nRegards,'
    suggestions.append({'original': original_text, 'rewritten': friendly_fb, 'change': 'Friendly fallback (redacted)'})
    suggestions.append({'original': original_text, 'rewritten': formal_fb, 'change': 'Formal fallback (redacted)'})
    return suggestions

def remove_offensive_words(text, tox_pipe):
    """Use toxicity pipeline to identify and remove offensive words/phrases."""
    if not tox_pipe:
        return text
    
    def check_toxicity(text_to_check, threshold=0.15):
        try:
            text_safe = truncate_text_for_model(text_to_check, max_chars=300)
            res = tox_pipe(text_safe)[0]
            scores = {item['label'].lower(): float(item['score']) for item in res}
            tox_score = 0.0
            for key in ['toxic', 'toxicity', 'abusive', 'obscene']:
                if key in scores:
                    tox_score = max(tox_score, scores[key])
            return tox_score >= threshold
        except Exception:
            return False
    
    words = text.split()
    cleaned_words = []
    skip_next = False
    
    for i, word in enumerate(words):
        if skip_next:
            skip_next = False
            continue
            
        word_clean = re.sub(r'[^\w]', '', word.lower())
        if not word_clean:
            cleaned_words.append(word)
            continue
        
        if check_toxicity(word_clean, threshold=0.15):
            continue
        
        if i < len(words) - 1:
            next_word_clean = re.sub(r'[^\w]', '', words[i+1].lower())
            phrase = word_clean + ' ' + next_word_clean
            if check_toxicity(phrase, threshold=0.15):
                skip_next = True
                continue
        
        cleaned_words.append(word)
    
    result = ' '.join(cleaned_words)
    result = re.sub(r'\s+', ' ', result).strip()
    return result

def create_rule_based_rewrites(text, tone):
    """Create context-aware rewrites using rule-based transformations."""
    suggestions = []
    text_lower = text.lower()
    
    _, tox_pipe = get_pipelines()
    
    rude_replacements = {
        r'\bstupid\b': 'not ideal',
        r'\bidiot\b': '',
        r'\bidot\b': '',
        r'\bfool\b': 'unwise',
        r'\bcrazy\b': 'unexpected',
        r'\bhorrible\b': 'concerning',
        r'\bterrible\b': 'needs improvement',
        r'\bhate\b': 'prefer not to',
        r'\bdamn\b': '',
        r'\bhell\b': '',
        r'\bwhat the\b': 'I am surprised that',
        r'\bridiculous\b': 'unexpected',
        r'\babsurd\b': 'unusual',
        r'\bpathetic\b': 'disappointing',
        r'\buseless\b': 'needs revision',
        r'\bcrap\b': 'suboptimal',
        r'\bsucks\b': 'needs work',
    }
    
    command_replacements = {
        r'\basap\b': 'at your earliest convenience',
        r'\bsend\s+asap\b': 'please send when you have a chance',
        r'\bnow\s+(send|do|give)\b': r'please \1',
        r'\bgive\s+me\b': 'could you please send me',
        r'\bgive\b': 'please send',
    }
    
    if tone == 'rude':
        friendly_text = text
        
        friendly_text = remove_offensive_words(friendly_text, tox_pipe)
        friendly_text = re.sub(r'\b(damn|hell|damned)\b', '', friendly_text, flags=re.IGNORECASE)
        friendly_text = re.sub(r'\b(idiot|idot)\b', '', friendly_text, flags=re.IGNORECASE)
        friendly_text = re.sub(r'\s+', ' ', friendly_text)
        
        for pattern, replacement in command_replacements.items():
            friendly_text = re.sub(pattern, replacement, friendly_text, flags=re.IGNORECASE)
        
        for pattern, replacement in rude_replacements.items():
            if replacement:
                friendly_text = re.sub(pattern, replacement, friendly_text, flags=re.IGNORECASE)
        
        if re.search(r'\bsend\b', friendly_text.lower()) and 'please' not in friendly_text.lower():
            friendly_text = re.sub(r'\b(send)\b', 'please send', friendly_text, flags=re.IGNORECASE, count=1)
        
        friendly_text = re.sub(r'\b(very|extremely|incredibly)\s+(bad|wrong|awful)', 'concerning', friendly_text, flags=re.IGNORECASE)
        friendly_text = re.sub(r'\bis\s+(terrible|horrible|awful|bad)\b', 'needs improvement', friendly_text, flags=re.IGNORECASE)
        friendly_text = re.sub(r'\s+', ' ', friendly_text).strip()
        
        if re.search(r'\bhey\s+(idiot|idot|stupid)', text_lower):
            friendly_text = re.sub(r'\bhey\b', '', friendly_text, flags=re.IGNORECASE)
            friendly_text = re.sub(r'\s+', ' ', friendly_text).strip()
        
        if any(word in text_lower for word in ['hate', 'stupid', 'idiot', 'idot', 'terrible', 'horrible', 'ridiculous']):
            if not any(starter in friendly_text.lower()[:50] for starter in ['i wanted', 'i hope', 'i appreciate', 'hi', 'hello']):
                friendly_text = "I wanted to reach out about this. " + friendly_text
        
        if not re.match(r'^(hi|hello|dear|greetings)', friendly_text.lower().strip()):
            friendly_text = "Hi,\n\n" + friendly_text
        else:
            friendly_text = re.sub(r'^(hey)\s*[,\-]?\s*', 'Hi,\n\n', friendly_text, flags=re.IGNORECASE)
        
        if not any(word in friendly_text.lower()[-100:] for word in ['thanks', 'thank you', 'regards', 'sincerely', 'best']):
            friendly_text += "\n\nThanks!"
        
        suggestions.append({
            'original': text,
            'rewritten': friendly_text,
            'change': 'Friendly rewrite (softened language)'
        })
        
        formal_text = text
        formal_text = remove_offensive_words(formal_text, tox_pipe)
        formal_text = re.sub(r'\b(damn|hell|damned)\b', '', formal_text, flags=re.IGNORECASE)
        formal_text = re.sub(r'\b(idiot|idot)\b', '', formal_text, flags=re.IGNORECASE)
        formal_text = re.sub(r'\s+', ' ', formal_text)
        
        for pattern, replacement in command_replacements.items():
            formal_text = re.sub(pattern, replacement, formal_text, flags=re.IGNORECASE)
        
        for pattern, replacement in rude_replacements.items():
            if replacement:
                formal_text = re.sub(pattern, replacement, formal_text, flags=re.IGNORECASE)
        
        if re.search(r'\bsend\b', formal_text.lower()) and 'please' not in formal_text.lower():
            formal_text = re.sub(r'\b(send)\b', 'please send', formal_text, flags=re.IGNORECASE, count=1)
        
        formal_text = re.sub(r'\b(very|extremely|incredibly)\s+(bad|wrong|awful)', 'concerning', formal_text, flags=re.IGNORECASE)
        formal_text = re.sub(r'\s+', ' ', formal_text).strip()
        
        if re.search(r'\bhey\s+(idiot|idot|stupid)', text_lower):
            formal_text = re.sub(r'\bhey\b', '', formal_text, flags=re.IGNORECASE)
            formal_text = re.sub(r'\s+', ' ', formal_text).strip()
        
        if not re.match(r'^(dear|hello|greetings)', formal_text.lower().strip()):
            formal_text = "Hello,\n\n" + formal_text
        
        if 'regards' not in formal_text.lower() and 'sincerely' not in formal_text.lower():
            formal_text += "\n\nBest regards,"
        
        suggestions.append({
            'original': text,
            'rewritten': formal_text,
            'change': 'Formal rewrite (professional tone)'
        })
        
    elif tone == 'formal':
        text_lower_check = text.lower()
        has_rude_content = (
            any(kw in text_lower_check for kw in RUDE_KEYWORDS) or
            any(re.search(pattern, text_lower_check, re.IGNORECASE) for pattern in AGGRESSIVE_PATTERNS) or
            (re.search(r'\bhey\b\s+(idiot|idot|stupid)', text_lower_check, re.IGNORECASE))
        )
        
        if has_rude_content:
            friendly_text = text
            friendly_text = remove_offensive_words(friendly_text, tox_pipe)
            friendly_text = re.sub(r'\b(damn|hell|damned)\b', '', friendly_text, flags=re.IGNORECASE)
            friendly_text = re.sub(r'\b(idiot|idot)\b', '', friendly_text, flags=re.IGNORECASE)
            friendly_text = re.sub(r'\s+', ' ', friendly_text).strip()
            
            for pattern, replacement in command_replacements.items():
                friendly_text = re.sub(pattern, replacement, friendly_text, flags=re.IGNORECASE)
            
            if re.search(r'\bsend\b', friendly_text.lower()) and 'please' not in friendly_text.lower():
                friendly_text = re.sub(r'\b(send)\b', 'please send', friendly_text, flags=re.IGNORECASE, count=1)
            
            if re.search(r'\bhey\s+(idiot|idot|stupid)', text_lower_check):
                friendly_text = re.sub(r'\bhey\b', '', friendly_text, flags=re.IGNORECASE)
                friendly_text = re.sub(r'\s+', ' ', friendly_text).strip()
            
            if not re.match(r'^(hi|hello|dear)', friendly_text.lower().strip()):
                friendly_text = "Hi,\n\n" + friendly_text
            else:
                friendly_text = re.sub(r'^(hey)\s*[,\-]?\s*', 'Hi,\n\n', friendly_text, flags=re.IGNORECASE)
            
            if not any(word in friendly_text.lower()[-100:] for word in ['thanks', 'thank you', 'regards', 'sincerely']):
                friendly_text += "\n\nThanks!"
            
            suggestions.append({
                'original': text,
                'rewritten': friendly_text,
                'change': 'Polite rewrite (removed offensive language)'
            })
        else:
            warm_text = text
            if re.match(r'^(dear|hello)', warm_text.lower()):
                warm_text = re.sub(r'^Dear\s+', 'Hi ', warm_text, flags=re.IGNORECASE)
                warm_text = re.sub(r'^Hello\s*,\s*', 'Hi,\n\n', warm_text, flags=re.IGNORECASE)
            
            if 'sincerely' in warm_text.lower() or 'respectfully' in warm_text.lower():
                warm_text = re.sub(r'(?i)\s*(sincerely|respectfully|yours sincerely)[,.]?\s*$', '\n\nBest,', warm_text)
            
            suggestions.append({
                'original': text,
                'rewritten': warm_text,
                'change': 'Warmer version (maintains professionalism)'
            })
        
    else:
        professional_text = text
        if 'cheers' in professional_text.lower() and 'regards' not in professional_text.lower():
            professional_text = re.sub(r'(?i)\s*cheers\s*[,.]?\s*$', '\n\nBest regards,', professional_text)
        
        suggestions.append({
            'original': text,
            'rewritten': professional_text,
            'change': 'Maintains friendly tone with professionalism'
        })
    
    return suggestions

# ... (rest of the routes remain the same)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/analyze_email', methods=['POST'])
def analyze_email():
    data = request.get_json()
    email_text = data.get('email_text', '')
    
    if not email_text:
        return jsonify({'error': 'No email text provided'}), 400
    
    tone, confidence, vader_scores = classify_tone(email_text)
    suggestions = suggest_polite_rewrite(email_text, tone)
    
    clean_suggestions = []
    for s in (suggestions or []):
        rew = s.get('rewritten', '') or ''
        rew = re.sub(r"'{0,2}---VERSION_SEPARATOR---'{0,2}", '', rew, flags=re.IGNORECASE)
        rew = rew.replace('---VERSION_SEPARATOR---', '')
        rew = rew.replace('\r', '')
        rew = rew.strip()
        if not rew:
            base = (email_text or '').strip()
            if base:
                rew = 'Hi,\n\n' + base + '\n\nThanks,'
            else:
                rew = base
        s['rewritten'] = rew
        clean_suggestions.append(s)
    suggestions = clean_suggestions
    
    recommendations = []
    if tone == 'rude':
        recommendations.append('Consider using softer language to maintain professionalism')
        recommendations.append('Avoid negative words that might offend the recipient')
        recommendations.append('Focus on solutions rather than problems')
    elif tone == 'formal':
        recommendations.append('Tone is appropriate for professional communication')
        recommendations.append('Consider adding a friendly greeting if appropriate')
    else:
        recommendations.append('Friendly tone is great for maintaining relationships')
        recommendations.append('Maintain professionalism while being warm')
    
    return jsonify({
        'tone': tone,
        'confidence': round(confidence, 2),
        'vader_scores': vader_scores,
        'suggestions': suggestions,
        'recommendations': recommendations
    })

@app.route('/fetch_emails', methods=['POST'])
def fetch_emails():
    data = request.get_json()
    email_address = data.get('email')
    password = data.get('password')
    
    if not email_address or not password:
        return jsonify({'error': 'Email and password required'}), 400
    
    try:
        mail = imaplib.IMAP4_SSL('imap.gmail.com', 993)
        mail.login(email_address, password)
        mail.select('inbox')
        
        try:
            limit = int(data.get('limit', 20))
        except Exception:
            limit = 20

        status, messages = mail.search(None, 'UNSEEN')
        email_ids = messages[0].split() if status == 'OK' else []
        if not email_ids:
            status, messages = mail.search(None, 'ALL')
            email_ids = messages[0].split() if status == 'OK' else []

        emails = []
        for email_id in email_ids[-limit:]:
            status, msg_data = mail.fetch(email_id, '(RFC822)')
            
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    
                    subject_raw = msg['Subject']
                    subject = ''
                    if subject_raw:
                        decoded_parts = decode_header(subject_raw)
                        for part, enc in decoded_parts:
                            if isinstance(part, bytes):
                                try:
                                    subject += part.decode(enc or 'utf-8', errors='ignore')
                                except Exception:
                                    subject += part.decode('utf-8', errors='ignore')
                            else:
                                subject += part
                    if not subject:
                        subject = 'No Subject'
                    
                    sender = msg['From']
                    
                    body = ''
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            content_disposition = str(part.get('Content-Disposition') or '')
                            if content_type == "text/plain" and 'attachment' not in content_disposition:
                                try:
                                    payload = part.get_payload(decode=True)
                                    charset = part.get_content_charset() or 'utf-8'
                                    body = (payload or b'').decode(charset, errors='ignore')
                                    if body:
                                        break
                                except Exception:
                                    continue
                    else:
                        if msg.get_content_type() == "text/plain":
                            payload = msg.get_payload(decode=True)
                            try:
                                body = (payload or b'').decode(msg.get_content_charset() or 'utf-8', errors='ignore')
                            except Exception:
                                body = (payload or b'').decode('utf-8', errors='ignore')
                    
                    try:
                        tone, confidence, _ = classify_tone(body)
                    except Exception as e:
                        body_lower = body.lower()
                        if any(kw in body_lower for kw in RUDE_KEYWORDS):
                            tone = 'rude'
                            confidence = 0.7
                        elif any(kw in body_lower for kw in FRIENDLY_KEYWORDS):
                            tone = 'friendly'
                            confidence = 0.6
                        else:
                            tone = 'formal'
                            confidence = 0.5
                    
                    emails.append({
                        'id': email_id.decode(),
                        'subject': subject,
                        'sender': sender,
                        'body': body[:200] + '...' if len(body) > 200 else body,
                        'full_body': body,
                        'tone': tone,
                        'confidence': round(confidence, 2)
                    })
        
        mail.close()
        mail.logout()
        
        return jsonify({'emails': emails})
        
    except imaplib.IMAP4.error as e:
        return jsonify({'error': 'Invalid credentials. Please enable "Less secure app access" or use an App Password.'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)