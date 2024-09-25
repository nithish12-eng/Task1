from flask import Flask, request, render_template, jsonify
import pandas as pd
import os
import language_tool_python
import spacy

app = Flask(__name__)

# Initialize the grammar checking tool and NLP model
tool = language_tool_python.LanguageTool('en-US')
nlp = spacy.load('en_core_web_sm')

# Directory where datasets are stored
data_dir = "Task7"

# Load datasets
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

preprocessed_train_set = load_data(os.path.join(data_dir, 'preprocessed_train_set.csv'))
preprocessed_valid_set = load_data(os.path.join(data_dir, 'preprocessed_valid_set.csv'))
scored_train_set = load_data(os.path.join(data_dir, 'scored_train_set.csv'))
scored_valid_set = load_data(os.path.join(data_dir, 'scored_valid_set.csv'))

# Preprocessing function
def preprocess_essay(text):
    if isinstance(text, str):
        return text.lower().strip()
    return ''

# Function to check grammatical mistakes
def check_grammar(essay):
    matches = tool.check(essay)
    num_errors = len(matches)
    return num_errors

# Function to calculate the number of lines in the essay
def count_lines(essay):
    lines = essay.splitlines()  # Split the text by actual line breaks
    num_lines = len([line for line in lines if line.strip()])  # Count non-empty lines
    return num_lines

# Function to extract main concept of the essay using Named Entity Recognition (NER)
def extract_concept(essay):
    doc = nlp(essay)
    # Extract entities or keywords as the main concept (e.g., a named entity like an organization, person, etc.)
    concepts = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PERSON', 'GPE', 'NORP', 'PRODUCT', 'EVENT']]
    return concepts if concepts else None

# Function to check if the entire essay relates to the main concept
def is_essay_relevant_to_concept(essay, concepts):
    if not concepts:
        return True  # If no concept was identified, we cannot penalize it

    doc = nlp(essay)
    matches = 0
    for concept in concepts:
        if concept.lower() in doc.text.lower():
            matches += 1

    # If less than half of the essay matches the main concept, we penalize
    total_sentences = len(list(doc.sents))
    if matches < total_sentences // 2:
        return False
    return True

# Function to compare essays with preprocessed datasets and provide a score
def get_score_from_dataset(essay):
    # Preprocess the input essay for comparison
    processed_essay = preprocess_essay(essay)
    
    # Check if the essay exists in preprocessed_train_set or preprocessed_valid_set
    if preprocessed_train_set is not None and processed_essay in preprocessed_train_set['essay'].values:
        matched_row = preprocessed_train_set[preprocessed_train_set['essay'] == processed_essay]
        return int(matched_row['score'].values[0])
    
    if preprocessed_valid_set is not None and processed_essay in preprocessed_valid_set['essay'].values:
        matched_row = preprocessed_valid_set[preprocessed_valid_set['essay'] == processed_essay]
        return int(matched_row['score'].values[0])

    # Default score of 3 if no match found
    return None

# Scoring function
def score_essay(essay):
    essay = preprocess_essay(essay)
    
    # Try to fetch a score from the preprocessed datasets
    dataset_score = get_score_from_dataset(essay)
    if dataset_score is not None:
        return dataset_score
    
    # Get the number of lines
    num_lines = count_lines(essay)
    
    # Get the number of grammatical errors
    num_grammar_errors = check_grammar(essay)

    # Debugging outputs
    print(f"Processed Essay: {essay}")
    print(f"Number of lines: {num_lines}")
    print(f"Number of grammatical errors: {num_grammar_errors}")
    
    # Base score initialization
    score = 3  # Start with a default score of 3

    # Adjust score based on the number of lines
    if num_lines >= 10:
        score += 2
    elif num_lines >= 5:
        score += 1
    else:
        score -= 1  # Penalize for short essays

    # Adjust score based on grammatical mistakes
    if num_grammar_errors > 10:
        score -= 2
    elif num_grammar_errors > 5:
        score -= 1
    else:
        score += 1  # Reward fewer grammar mistakes

    # Check if the essay is relevant to the identified concept
    concepts = extract_concept(essay)
    is_relevant = is_essay_relevant_to_concept(essay, concepts)
    
    if not is_relevant:
        score -= 3  # Penalize for being off-topic

    # Ensure score is between 1 and 6
    score = max(1, min(score, 6))

    print(f"Final Score: {score}")  # Debugging: check the final score
    return score

@app.route('/')
def index():
    return render_template('Task7.html')

@app.route('/submit', methods=['POST'])
def submit():
    essay = request.form.get('essay')
    if not essay:
        return jsonify({'error': 'No essay provided'}), 400
    
    try:
        score = score_essay(essay)
        return jsonify({'score': score})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Change the host and port for wider access if needed
    app.run(host='0.0.0.0', port=5001, debug=True)
