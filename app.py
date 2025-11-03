from flask import Flask, render_template, request, session, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import imaplib
import email
from email.header import decode_header
import os
import re
import json

# Initialize Flask app with explicit template and static folders
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production-for-local-use')

# Initialize analyzers
vader_analyzer = SentimentIntensityAnalyzer()

# Lazy loading for heavy models - only load if needed
sentiment_pipeline = None
toxicity_pipeline = None

def get_pipelines():
    """Lazy load pipelines only when needed - disabled for free tier"""
    global sentiment_pipeline, toxicity_pipeline
    # Disable heavy models on free tier to save memory
    USE_TRANSFORMERS = os.environ.get('USE_TRANSFORMERS', 'false').lower() == 'true'
    
    if not USE_TRANSFORMERS:
        return None, None
    
    try:
        if sentiment_pipeline is None:
            from transformers import pipeline
            sentiment_pipeline = pipeline(
                'text-classification',
                model='cardiffnlp/twitter-roberta-base-sentiment-latest',
                tokenizer='cardiffnlp/twitter-roberta-base-sentiment-latest',
                top_k=None
            )
        if toxicity_pipeline is None:
            from transformers import pipeline
            toxicity_pipeline = pipeline(
                'text-classification',
                model='unitary/toxic-bert',
                tokenizer='unitary/toxic-bert',
                top_k=None
            )
        return sentiment_pipeline, toxicity_pipeline
    except Exception as e:
        print(f"Failed to load transformers: {e}")
        return None, None

# Tone detection keywords
RUDE_KEYWORDS = ['stupid', 'idiot', 'idot', 'fool', 'crazy', 'horrible', 'terrible', 'hate', 
                 'damn', 'hell', 'what the', 'ridiculous', 'absurd', 'pathetic', 'useless',
                 'moron', 'dumb', 'dummy', 'wtf', 'bullshit', 'crap', 'sucks']
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

def classify_tone(text):
    """Classify tone using VADER and keyword analysis (lightweight version)."""
    text = (text or '').strip()
    if not text:
        return 'formal', 0.5, {'compound': 0.0}

    text_lower = text.lower()
    
    # Count keywords
    rude_keyword_count = sum(1 for kw in RUDE_KEYWORDS if kw in text_lower)
    formal_keyword_count = sum(1 for kw in FORMAL_KEYWORDS if kw in text_lower)
    friendly_keyword_count = sum(1 for kw in FRIENDLY_KEYWORDS if kw in text_lower)
    
    # Count aggressive patterns
    aggressive_pattern_count = sum(1 for pattern in AGGRESSIVE_PATTERNS if re.search(pattern, text_lower, re.IGNORECASE))
    
    # Check for "hey" used rudely
    hey_used_rudely = False
    if 'hey' in text_lower:
        hey_pattern = re.search(r'\bhey\b\s+(\w+)', text_lower)
        if hey_pattern:
            next_word = hey_pattern.group(1)
            if next_word in ['idiot', 'idot', 'stupid', 'moron', 'dumb']:
                hey_used_rudely = True
            elif re.search(r'\bhey\b.*\b(send|do|give|get)\b', text_lower) and 'please' not in text_lower:
                hey_used_rudely = True

    # VADER sentiment analysis
    vader_scores = vader_analyzer.polarity_scores(text)
    compound_score = vader_scores.get('compound', 0.0)
    
    # Calculate sentiment scores
    pos = vader_scores.get('pos', 0.0)
    neg = vader_scores.get('neg', 0.0)
    neu = vader_scores.get('neu', 0.0)
    
    # Boost scores based on keywords
    keyword_rude_boost = min(0.3, rude_keyword_count * 0.15)
    aggressive_boost = min(0.4, aggressive_pattern_count * 0.2)
    if hey_used_rudely:
        aggressive_boost = max(aggressive_boost, 0.35)
    
    keyword_friendly_boost = min(0.2, friendly_keyword_count * 0.1)
    keyword_formal_boost = min(0.2, formal_keyword_count * 0.1)

    # Calculate final rude score
    rude_score = neg + keyword_rude_boost + aggressive_boost
    
    # Determine tone
    if (neg > 0.3 and compound_score < -0.2) or \
       (rude_keyword_count > 0 and neg > 0.2) or \
       aggressive_pattern_count > 0 or \
       hey_used_rudely or \
       (rude_keyword_count >= 2):
        tone = 'rude'
        confidence = min(0.98, 0.55 + rude_score * 0.4)
    elif not hey_used_rudely and \
         ((pos >= max(neg, neu) and pos > 0.4) or \
          (friendly_keyword_count > 0 and pos > 0.35) or \
          compound_score > 0.2):
        tone = 'friendly'
        friendly_score = pos + keyword_friendly_boost + max(0, compound_score * 0.5)
        confidence = min(0.95, 0.5 + friendly_score * 0.4)
    else:
        tone = 'formal'
        formal_score = neu + keyword_formal_boost + (1.0 - abs(compound_score) * 0.5)
        confidence = max(0.6, 0.5 + formal_score * 0.3)

    return tone, round(confidence, 3), {
        'compound': compound_score, 
        'pos': pos, 
        'neg': neg, 
        'neu': neu,
        'rude_keywords': rude_keyword_count,
        'aggressive_patterns': aggressive_pattern_count
    }

def suggest_polite_rewrite(original_text, tone):
    """Suggest polite alternatives using rule-based transformations."""
    text = (original_text or '').strip()
    if not text:
        return [{
            'original': original_text,
            'rewritten': original_text,
            'change': 'No content provided'
        }]

    suggestions = []
    text_lower = text.lower()
    
    # Rude word replacements
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
        r'\bwtf\b': 'I am confused about',
        r'\bbullshit\b': 'inaccurate',
    }
    
    # Command replacements
    command_replacements = {
        r'\basap\b': 'at your earliest convenience',
        r'\bsend\s+asap\b': 'please send when you have a chance',
        r'\bnow\s+(send|do|give)\b': r'please \1',
        r'\bgive\s+me\b': 'could you please send me',
        r'\bgive\b': 'please send',
    }
    
    if tone == 'rude':
        # Friendly version
        friendly_text = text
        
        # Remove profanity
        for pattern, replacement in rude_replacements.items():
            friendly_text = re.sub(pattern, replacement, friendly_text, flags=re.IGNORECASE)
        
        # Fix commands
        for pattern, replacement in command_replacements.items():
            friendly_text = re.sub(pattern, replacement, friendly_text, flags=re.IGNORECASE)
        
        # Add "please" if sending/requesting
        if re.search(r'\bsend\b', friendly_text.lower()) and 'please' not in friendly_text.lower():
            friendly_text = re.sub(r'\b(send)\b', 'please send', friendly_text, flags=re.IGNORECASE, count=1)
        
        # Clean up extra spaces
        friendly_text = re.sub(r'\s+', ' ', friendly_text).strip()
        
        # Remove rude greetings
        if re.search(r'\bhey\s+(idiot|idot|stupid)', text_lower):
            friendly_text = re.sub(r'\bhey\b', '', friendly_text, flags=re.IGNORECASE)
            friendly_text = re.sub(r'\s+', ' ', friendly_text).strip()
        
        # Add polite opening if missing
        if not re.match(r'^(hi|hello|dear|greetings)', friendly_text.lower().strip()):
            friendly_text = "Hi,\n\n" + friendly_text
        else:
            friendly_text = re.sub(r'^(hey)\s*[,\-]?\s*', 'Hi,\n\n', friendly_text, flags=re.IGNORECASE)
        
        # Add polite closing if missing
        if not any(word in friendly_text.lower()[-100:] for word in ['thanks', 'thank you', 'regards', 'sincerely', 'best']):
            friendly_text += "\n\nThanks!"
        
        suggestions.append({
            'original': text,
            'rewritten': friendly_text,
            'change': 'Friendly rewrite (softened language)'
        })
        
        # Formal version
        formal_text = text
        
        # Remove profanity
        for pattern, replacement in rude_replacements.items():
            formal_text = re.sub(pattern, replacement, formal_text, flags=re.IGNORECASE)
        
        # Fix commands
        for pattern, replacement in command_replacements.items():
            formal_text = re.sub(pattern, replacement, formal_text, flags=re.IGNORECASE)
        
        # Add "please"
        if re.search(r'\bsend\b', formal_text.lower()) and 'please' not in formal_text.lower():
            formal_text = re.sub(r'\b(send)\b', 'please send', formal_text, flags=re.IGNORECASE, count=1)
        
        # Clean up
        formal_text = re.sub(r'\s+', ' ', formal_text).strip()
        
        # Remove rude greetings
        if re.search(r'\bhey\s+(idiot|idot|stupid)', text_lower):
            formal_text = re.sub(r'\bhey\b', '', formal_text, flags=re.IGNORECASE)
            formal_text = re.sub(r'\s+', ' ', formal_text).strip()
        
        # Add formal opening
        if not re.match(r'^(dear|hello|greetings)', formal_text.lower().strip()):
            formal_text = "Hello,\n\n" + formal_text
        
        # Add formal closing
        if 'regards' not in formal_text.lower() and 'sincerely' not in formal_text.lower():
            formal_text += "\n\nBest regards,"
        
        suggestions.append({
            'original': text,
            'rewritten': formal_text,
            'change': 'Formal rewrite (professional tone)'
        })
        
    elif tone == 'formal':
        # Make it warmer
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
        
    else:  # friendly
        # Make it more professional
        professional_text = text
        if 'cheers' in professional_text.lower() and 'regards' not in professional_text.lower():
            professional_text = re.sub(r'(?i)\s*cheers\s*[,.]?\s*$', '\n\nBest regards,', professional_text)
        
        suggestions.append({
            'original': text,
            'rewritten': professional_text,
            'change': 'Maintains friendly tone with professionalism'
        })
    
    return suggestions

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/analyze_email', methods=['POST'])
def analyze_email():
    try:
        data = request.get_json()
        email_text = data.get('email_text', '')
        
        if not email_text:
            return jsonify({'error': 'No email text provided'}), 400
        
        # Classify tone
        tone, confidence, vader_scores = classify_tone(email_text)
        
        # Get suggestions
        suggestions = suggest_polite_rewrite(email_text, tone)
        
        # Clean suggestions
        clean_suggestions = []
        for s in (suggestions or []):
            rew = s.get('rewritten', '') or ''
            rew = re.sub(r'\s+', ' ', rew).strip()
            if not rew:
                base = (email_text or '').strip()
                if base:
                    rew = 'Hi,\n\n' + base + '\n\nThanks,'
                else:
                    rew = base
            s['rewritten'] = rew
            clean_suggestions.append(s)
        suggestions = clean_suggestions
        
        # Generate recommendations
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
    except Exception as e:
        print(f"Error in analyze_email: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/fetch_emails', methods=['POST'])
def fetch_emails():
    try:
        data = request.get_json()
        email_address = data.get('email')
        password = data.get('password')
        
        if not email_address or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
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
                    
                    # Get subject
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
                    
                    # Get body
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
                    
                    # Classify tone
                    try:
                        tone, confidence, _ = classify_tone(body)
                    except Exception as e:
                        print(f"Error classifying tone: {e}")
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
        print(f"Error in fetch_emails: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'version': '2.0'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)