import os
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
import PyPDF2
import pandas as pd
from bs4 import BeautifulSoup
import markdown

# Load environment variables
load_dotenv()

# Retrieve API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_AI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is not set in the .env file")
if not GEMINI_API_KEY:
    raise ValueError("Gemini API key is not set in the .env file")

# Initialize API clients
client_openai = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)

# Configurations
MAX_FILE_SIZE = 1024 * 1024 * 10  # 10MB
MAX_WORDS_PER_CHUNK = 800
SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.html', '.md', '.csv']

def split_text_into_chunks(text, max_words=MAX_WORDS_PER_CHUNK):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def extract_text_from_file(file):
    """Extract text content from different file formats"""
    filename = file.filename.lower()
    
    try:
        if filename.endswith('.txt'):
            return extract_text_plain(file)
        
        elif filename.endswith('.pdf'):
            return extract_text_from_pdf(file)
        
        elif filename.endswith('.html'):
            return extract_text_from_html(file)
        
        elif filename.endswith('.md'):
            return extract_text_from_markdown(file)
        
        elif filename.endswith('.csv'):
            return extract_text_from_csv(file)
        
        else:
            raise ValueError(f"Type de fichier non supporté: {filename}")
    
    except Exception as e:
        raise Exception(f"Erreur lors de l'extraction du texte de {filename}: {str(e)}")

def extract_text_plain(file):
    """Extract text from plain text files"""
    try:
        text = file.read().decode('utf-8')
        if not text.strip():
            raise Exception("Le fichier texte est vide")
        return text
    except UnicodeDecodeError:
        try:
            file.seek(0)
            text = file.read().decode('latin-1')
            return text
        except Exception as e:
            raise Exception(f"Impossible de décoder le fichier texte: {str(e)}")

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        if not text.strip():
            raise Exception("Aucun texte trouvé dans le fichier PDF")
        
        return text
    
    except Exception as e:
        raise Exception(f"Erreur lors de la lecture du PDF: {str(e)}")

def extract_text_from_html(file):
    """Extract text from HTML file"""
    try:
        html_content = file.read().decode('utf-8')
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Supprimer les scripts et styles
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extraire le texte
        text = soup.get_text()
        
        # Nettoyer le texte
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        if not text.strip():
            raise Exception("Aucun texte trouvé dans le fichier HTML")
        
        return text
    
    except UnicodeDecodeError:
        try:
            file.seek(0)
            html_content = file.read().decode('latin-1')
            soup = BeautifulSoup(html_content, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            return text
        except Exception as e:
            raise Exception(f"Erreur lors de la lecture du HTML: {str(e)}")
    except Exception as e:
        raise Exception(f"Erreur lors de la lecture du HTML: {str(e)}")

def extract_text_from_markdown(file):
    """Extract text from Markdown file"""
    try:
        md_content = file.read().decode('utf-8')
        
        # Convertir markdown en HTML puis en texte plain
        html = markdown.markdown(md_content)
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
        
        # Ajouter aussi le contenu markdown brut pour préserver la structure
        text += "\n\nContenu Markdown original:\n" + md_content
        
        if not text.strip():
            raise Exception("Aucun texte trouvé dans le fichier Markdown")
        
        return text
    
    except UnicodeDecodeError:
        try:
            file.seek(0)
            md_content = file.read().decode('latin-1')
            html = markdown.markdown(md_content)
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text()
            text += "\n\nContenu Markdown original:\n" + md_content
            return text
        except Exception as e:
            raise Exception(f"Erreur lors de la lecture du Markdown: {str(e)}")
    except Exception as e:
        raise Exception(f"Erreur lors de la lecture du Markdown: {str(e)}")

def extract_text_from_csv(file):
    """Extract text from CSV file"""
    try:
        # Essayer différents encodages et séparateurs
        encodings = ['utf-8', 'latin-1', 'cp1252']
        separators = [',', ';', '\t']
        
        df = None
        for encoding in encodings:
            for sep in separators:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=encoding, sep=sep)
                    if len(df.columns) > 1:  # Si plus d'une colonne, le séparateur est probablement correct
                        break
                except:
                    continue
            if df is not None and len(df.columns) > 1:
                break
        
        if df is None:
            file.seek(0)
            df = pd.read_csv(file)  # Tentative avec les paramètres par défaut
        
        # Créer un résumé du contenu CSV
        text = f"=== ANALYSE DU FICHIER CSV ===\n\n"
        text += f"Nombre de lignes: {len(df)}\n"
        text += f"Nombre de colonnes: {len(df.columns)}\n"
        text += f"Colonnes: {', '.join(df.columns)}\n\n"
        
        # Ajouter un échantillon des données
        text += "=== ÉCHANTILLON DES DONNÉES (10 premières lignes) ===\n"
        sample_rows = min(10, len(df))
        for i in range(sample_rows):
            text += f"Ligne {i+1}:\n"
            for col in df.columns:
                value = str(df.iloc[i][col])
                if pd.notna(df.iloc[i][col]):
                    text += f"  - {col}: {value}\n"
            text += "\n"
        
        # Ajouter les statistiques pour les colonnes numériques
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            text += "=== STATISTIQUES DES COLONNES NUMÉRIQUES ===\n"
            stats = df[numeric_cols].describe()
            text += stats.to_string()
            text += "\n\n"
        
        # Ajouter des informations sur les colonnes de texte
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            text += "=== INFORMATIONS SUR LES COLONNES TEXTE ===\n"
            for col in text_cols:
                unique_values = df[col].nunique()
                most_common = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
                text += f"Colonne '{col}':\n"
                text += f"  - Valeurs uniques: {unique_values}\n"
                text += f"  - Valeur la plus fréquente: {most_common}\n"
                
                # Échantillon des valeurs uniques
                if unique_values <= 10:
                    unique_vals = df[col].unique()
                    text += f"  - Toutes les valeurs: {', '.join(map(str, unique_vals))}\n"
                text += "\n"
        
        if not text.strip():
            raise Exception("Aucune donnée trouvée dans le fichier CSV")
        
        return text
    
    except Exception as e:
        raise Exception(f"Erreur lors de la lecture du CSV: {str(e)}")

# --- OpenAI Functions ---
def create_chat_completion_openai(prompt, max_tokens=300):
    try:
        response = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"Erreur OpenAI: {str(e)}")

def summarize_chunk_openai(chunk):
    prompt = f"""Detect the language of this text and summarize it in 3-5 key bullet points using EXACTLY the same language:

- If the text is in FRENCH, respond ONLY in French
- If the text is in ENGLISH, respond ONLY in English  
- If the text is in ARABIC, respond ONLY in Arabic

Do not mix languages. Use the detected language throughout your response.

Text to analyze:
{chunk}"""
    return create_chat_completion_openai(prompt)

def summarize_long_text_openai(text):
    chunks = split_text_into_chunks(text)
    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        try:
            summary = summarize_chunk_openai(chunk)
            chunk_summaries.append(summary)
            print(f"Résumé du chunk {i+1}/{len(chunks)} terminé")
        except Exception as e:
            print(f"Erreur chunk {i+1}: {str(e)}")
            continue

    if not chunk_summaries:
        raise Exception("Aucun résumé n'a pu être généré")

    combined_summaries = "\n\n".join(chunk_summaries)
    final_prompt = f"""Summarize the following text in 5-7 clear bullet points and reply with the same language provided in the text. step1:start with a general overall title that describes the content of the text just once,step2:summarize the text as demanded:\n\n{combined_summaries}"""
    return create_chat_completion_openai(final_prompt, max_tokens=200)

# --- Gemini Functions ---
def create_chat_completion_gemini(prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise Exception(f"Erreur Gemini: {str(e)}")

def summarize_chunk_gemini(chunk):
    prompt = f"""Summarize the following text in 5-7 clear bullet points and reply with the same language provided in the text. step1:start with a general overall title that describes the content of the text just once,step2:summarize the text as demanded:\n\n
{chunk}"""
    return create_chat_completion_gemini(prompt)

def summarize_long_text_gemini(text):
    chunks = split_text_into_chunks(text)
    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        try:
            summary = summarize_chunk_gemini(chunk)
            chunk_summaries.append(summary)
            print(f"Résumé du chunk {i+1}/{len(chunks)} terminé")
        except Exception as e:
            print(f"Erreur chunk {i+1}: {str(e)}")
            continue

    if not chunk_summaries:
        raise Exception("Aucun résumé n'a pu être généré")

    combined_summaries = "\n\n".join(chunk_summaries)
    final_prompt = f"""Summarize the following text in 5-7 clear bullet points and reply with the same language provided in the text. step1:start with a general overall title that describes the content of the text just once,step2:summarize the text as demanded:\n\n {combined_summaries}"""
    return create_chat_completion_gemini(final_prompt)

# --- Flask Routes ---
@app.route("/")
def home():
    return send_file("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Aucun fichier fourni"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Aucun fichier sélectionné"}), 400

        # Vérifier l'extension du fichier
        file_extension = os.path.splitext(file.filename.lower())[1]
        if file_extension not in SUPPORTED_EXTENSIONS:
            return jsonify({
                "error": f"Format de fichier non supporté. Formats supportés: {', '.join(SUPPORTED_EXTENSIONS)}"
            }), 400

        # Vérifier la taille du fichier
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)

        if file_size > MAX_FILE_SIZE:
            return jsonify({
                "error": f"Fichier trop volumineux (max {MAX_FILE_SIZE/1024/1024}MB)"
            }), 400

        print(f"Extraction du texte du fichier: {file.filename}")
        # Extraire le texte du fichier selon son format
        text = extract_text_from_file(file)

        if not text.strip():
            return jsonify({"error": "Le fichier semble être vide ou ne contient pas de texte lisible"}), 400

        print(f"Texte extrait: {len(text)} caractères")

        # Récupérer le choix du modèle
        model_choice = request.form.get("model_choice", "openai").lower()

        print(f"Génération du résumé avec le modèle: {model_choice}")
        # Générer le résumé avec le modèle sélectionné
        if model_choice == "openai":
            summary = summarize_long_text_openai(text)
        elif model_choice == "gemini":
            summary = summarize_long_text_gemini(text)
        else:
            return jsonify({"error": "Choix de modèle invalide"}), 400

        return jsonify({
            "success": True,
            "summary": summary,
            "model_used": model_choice,
            "file_type": file_extension,
            "text_length": len(text),
            "filename": file.filename
        })

    except Exception as e:
        print(f"Erreur serveur: {str(e)}")
        return jsonify({
            "error": f"Erreur de traitement: {str(e)}",
            "success": False
        }), 500

if __name__ == "__main__":
    app.run(debug=True)