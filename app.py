import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from markupsafe import Markup
import matplotlib
matplotlib.use('Agg')  # Ensures non-interactive backend for server rendering
from matplotlib import pyplot as plt
import spacy
import stanza
import re
import io
import pandas as pd
from textblob import TextBlob
from langdetect import detect
from collections import Counter
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from spacy import displacy
import seaborn as sns
import stanza

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "your-secret-key")

# --- NLP Model Loaders ---
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"Failed to load spaCy model: {e}")
        return None

def load_stanza_pipeline():
    try:
        if not os.path.exists(os.path.join(stanza.download_directory, 'en')):
            stanza.download('en')
        return stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    except Exception as e:
        print(f"Failed to load stanza pipeline: {e}")
        return None

# Load models
nlp_spacy = load_spacy_model()
nlp_stanza = load_stanza_pipeline()

# --- Utility Functions ---
def clean_text(text):
    """Lowercase, remove non-word characters."""
    return re.sub(r'\W+', ' ', text.lower())

def extract_keywords(text):
    words = clean_text(text).split()
    return [w for w, _ in Counter(words).most_common(10)]

def perform_topic_modeling(text, num_topics=3, num_words=5):
    vectorizer = CountVectorizer(stop_words='english')
    dtm = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)

    topics = []
    strengths = []
    words = vectorizer.get_feature_names_out()

    for i, topic in enumerate(lda.components_):
        top_idx = topic.argsort()[-num_words:][::-1]
        top_words = [words[j] for j in top_idx]
        topics.append(f"Topic {i+1}: " + ", ".join(top_words))
        strengths.append(float(sum(topic[top_idx])))

    return topics, [f"Topic {i+1}" for i in range(num_topics)], strengths

def is_valid_license(key):
    valid_keys = {
        "MBC6-7JMP-W09O-OH87", "37XG-ZA39-2DK1-1NKG", "DYI3-L6V9-TKIO-OBGI",
        "49PZ-I8PI-MSG2-XFAK", "PSKL-4FVA-AEVV-9I6S", "I8QT-MNI7-AE4J-JK0V",
        "3GRR-56KS-O3PL-VMHW", "R4M6-V3JV-EON4-LBEJ", "Y2LX-79OT-XZXS-IC5S",
        "CILQ-WOAJ-67IC-F7YT", "I7IF-KQ4G-2G1A-MKJ8", "30VV-9JAX-HHH8-4SHM",
        "MH65-EIOZ-SC1B-TLOX", "AG0N-OCLT-LQ66-ZAF0", "L6UP-W6SA-7BY1-HRFL"
    }
    return key.strip() in valid_keys

@app.route('/')
def home():
    return "It works!"

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html', premium=session.get('premium', False))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        session['name'] = request.form['name']
        session['email'] = request.form['email']
        session['premium'] = is_valid_license(request.form['license_key'])

        if session['premium']:
            flash("✅ Registered successfully as a premium user!", "success")
        else:
            flash("❌ Invalid license key. Free version activated.", "warning")

        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['email'] = request.form['email']
        session['premium'] = is_valid_license(request.form['license_key'])

        if session['premium']:
            flash("✅ Logged in as premium user.", "success")
        else:
            flash("⚠️ Logged in as free user. License key invalid or missing.", "warning")

        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("👋 You have been logged out.", "info")
    return redirect(url_for('index'))

@app.route("/result")
def result():
    return render_template(
        "result.html",
        text=..., token_count=..., type_count=..., ttr=...,
        pos_labels=..., pos_values=...,
        entities=..., lemmas=...,
        dep_tree_html=..., cons_tree_html=...,
        readability=..., sentiment=...,
        language=..., premium=...,
        topics=..., topic_labels=..., topic_strengths=...
    )

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    doc = nlp_spacy(text)

    non_space_tokens = [token for token in doc if not token.is_space]
    tokens = [token.text for token in non_space_tokens]
    types = list(set(token.text.lower() for token in doc if not token.is_space))
    token_count = len(tokens)
    type_count = len(types)
    ttr = round(type_count / token_count, 3) if token_count else 0

    pos_counts = Counter(token.pos_ for token in doc)
    pos_labels = list(pos_counts.keys())
    pos_values = list(pos_counts.values())

    entities = [(ent.text, ent.label_) for ent in doc.ents]
    lemmas = [(token.text, token.lemma_) for token in doc]
    dependencies = [(token.text, token.dep_, token.head.text) for token in doc]

    premium = session.get('premium', False)

    readability = {'flesch_reading_ease': 'N/A', 'smog_index': 'N/A'}
    sentiment = {'polarity': 'N/A', 'subjectivity': 'N/A'}
    dep_tree_html = "<pre>Upgrade to Premium to view this.</pre>"
    cons_tree_html = "<pre>Upgrade to Premium to view this.</pre>"
    language = 'Unknown'
    topics = []
    topic_labels = []
    topic_strengths = []

    if premium:
        readability = {
        }

        blob = TextBlob(text)
        sentiment = {
            'polarity': blob.polarity,
            'subjectivity': blob.subjectivity
        }

        language = detect(text)
        dep_tree_html = displacy.render(doc, style='dep', page=True)
        stanza_doc = nlp_stanza(text)
        cons_tree_html = "".join(f"<pre>{sent.constituency}</pre>" for sent in stanza_doc.sentences)

        topics, topic_labels, topic_strengths = perform_topic_modeling(text)

    csv_io = io.StringIO()
    pd.DataFrame({
        'Token': [token.text for token in non_space_tokens],
        'Lemma': [token.lemma_ for token in non_space_tokens],
        'POS': [token.pos_ for token in non_space_tokens]
    }).to_csv(csv_io, index=False)
    session['data'] = csv_io.getvalue()

    return render_template('result.html',
        text=text,
        tokens=tokens,
        token_count=token_count,
        type_count=type_count,
        ttr=ttr,
        pos_labels=pos_labels,
        pos_values=pos_values,
        entities=entities,
        lemmas=lemmas,
        dependencies=dependencies,
        readability=readability,
        sentiment=sentiment,
        dep_tree_html=Markup(dep_tree_html),
        cons_tree_html=Markup(cons_tree_html),
        language=language,
        topics=topics,
        topic_labels=topic_labels,
        topic_strengths=topic_strengths,
        premium=premium
    )

@app.route('/download/<string:type>')
def download(type):
    if not session.get('premium', False):
        return "Upgrade to premium for download access.", 403

    csv_data = session.get('data')
    if not csv_data:
        return "No analysis data found.", 404

    df = pd.read_csv(io.StringIO(csv_data))

    if type == 'csv':
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='analysis.csv')

    elif type == 'excel':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        output.seek(0)
        return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='analysis.xlsx')

    return "Invalid format requested", 400

@app.route('/visualize')
def visualize():
    if not session.get('premium', False):
        return "Upgrade to premium to access visualization.", 403

    csv_data = session.get('data')
    if not csv_data:
        return "No data available to visualize", 404

    df = pd.read_csv(io.StringIO(csv_data))
    if 'Token' not in df.columns:
        return "Token column not found", 500

    df['word_length'] = df['Token'].apply(len)
    fig, ax = plt.subplots()
    sns.histplot(df['word_length'], bins=10, kde=True, ax=ax)
    ax.set_title('Token Length Distribution')
    ax.set_xlabel('Length')
    ax.set_ylabel('Frequency')

    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)