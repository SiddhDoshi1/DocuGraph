import os
import time
import threading
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import math
from collections import defaultdict
import community as community_louvain
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from collections import defaultdict
import re
import joblib

# Domain dictionary (can be expanded)
DOMAIN_DICT = {
    'Artificial Intelligence': {
        'keywords': ['artificial intelligence', 'machine learning', 'deep learning', 
                    'neural network', 'reinforcement learning', 'supervised learning',
                    'unsupervised learning', 'computer vision', 'natural language processing'],
        'keyphrases': ['convolutional neural network', 'recurrent neural network', 
                      'generative adversarial network', 'transfer learning',
                      'transformer models', 'attention mechanism']
    },
    'Computer Science': {
        'keywords': ['algorithm', 'data structure', 'computational complexity',
                    'programming', 'software engineering', 'operating system',
                    'distributed systems', 'computer architecture'],
        'keyphrases': ['object-oriented programming', 'design patterns',
                      'big O notation', 'cloud computing', 'edge computing']
    },
    'Cybersecurity': {
        'keywords': ['cybersecurity', 'encryption', 'authentication', 'firewall',
                    'malware', 'ransomware', 'phishing', 'vulnerability'],
        'keyphrases': ['public key infrastructure', 'intrusion detection system',
                      'penetration testing', 'zero trust architecture']
    },
    'Data Science': {
        'keywords': ['data analysis', 'data visualization', 'predictive modeling',
                    'statistical analysis', 'data mining', 'big data'],
        'keyphrases': ['machine learning pipeline', 'feature engineering',
                      'exploratory data analysis', 'A/B testing']
    },
    'Mathematics': {
        'keywords': ['algebra', 'calculus', 'geometry', 'topology', 'statistics',
                    'probability', 'number theory', 'linear algebra'],
        'keyphrases': ['differential equations', 'abstract algebra',
                      'real analysis', 'complex analysis']
    },
    'Physics': {
        'keywords': ['quantum mechanics', 'thermodynamics', 'electromagnetism',
                    'relativity', 'particle physics', 'astrophysics'],
        'keyphrases': ['standard model', 'quantum field theory',
                      'general relativity', 'string theory']
    },
    'Chemistry': {
        'keywords': ['organic chemistry', 'inorganic chemistry', 'physical chemistry',
                    'analytical chemistry', 'biochemistry', 'polymer chemistry'],
        'keyphrases': ['periodic table trends', 'chemical bonding',
                      'reaction mechanisms', 'spectroscopic analysis']
    },
    'Biology': {
        'keywords': ['genetics', 'molecular biology', 'cell biology', 'ecology',
                    'evolution', 'microbiology', 'zoology', 'botany'],
        'keyphrases': ['central dogma', 'DNA replication',
                      'natural selection', 'ecosystem dynamics']
    },
    'Medicine': {
        'keywords': ['anatomy', 'physiology', 'pharmacology', 'pathology',
                    'immunology', 'neuroscience', 'cardiology'],
        'keyphrases': ['evidence-based medicine', 'clinical trials',
                      'drug metabolism', 'medical imaging']
    },
    'Engineering': {
        'keywords': ['mechanical engineering', 'electrical engineering',
                    'civil engineering', 'chemical engineering', 'aerospace'],
        'keyphrases': ['finite element analysis', 'control systems',
                      'structural design', 'thermodynamic cycles']
    },
    'Economics': {
        'keywords': ['microeconomics', 'macroeconomics', 'econometrics',
                    'game theory', 'behavioral economics'],
        'keyphrases': ['supply and demand', 'market equilibrium',
                      'monetary policy', 'fiscal policy']
    },
    'Psychology': {
        'keywords': ['cognitive psychology', 'developmental psychology',
                    'social psychology', 'clinical psychology', 'neuroscience'],
        'keyphrases': ['classical conditioning', 'cognitive behavioral therapy',
                      'working memory', 'attachment theory']
    },
    'Environmental Science': {
        'keywords': ['climate change', 'sustainability', 'conservation',
                    'pollution', 'renewable energy'],
        'keyphrases': ['carbon footprint', 'ecosystem services',
                      'life cycle assessment', 'greenhouse gas emissions']
    },
    'Business': {
        'keywords': ['marketing', 'finance', 'accounting', 'management',
                    'entrepreneurship', 'supply chain'],
        'keyphrases': ['return on investment', 'market segmentation',
                      'business model canvas', 'SWOT analysis']
    }
}

# Initialize BERT model for text embedding
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

app = Flask(__name__)
CORS(app)

# In-memory document storage
documents_collection = {}
graph_lock = threading.Lock()

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.int_, np.intc, np.intp, np.int8, np.int16, 
                         np.int32, np.int64, np.uint8, np.uint16, 
                         np.uint32, np.uint64)):
            return int(o)
        if isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, float) and math.isnan(o):
            return None
        return super(JSONEncoder, self).default(o)

app.json_encoder = JSONEncoder

def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def get_domain_keywords():
    """Extract all keywords from domain dictionary"""
    return {kw: domain for domain, keywords in DOMAIN_DICT.items() for kw in keywords}

def extract_domain_from_content(text):
    if not text or not isinstance(text, str):
        return 'Other'
    
    text = text.lower()
    domain_scores = defaultdict(int)
    
    # Score domains
    for domain, patterns in DOMAIN_DICT.items():
        # Score keywords (exact matches)
        for keyword in patterns['keywords']:
            domain_scores[domain] += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text))
        
        # Score keyphrases (weighted higher)
        for phrase in patterns['keyphrases']:
            domain_scores[domain] += 5 * len(re.findall(re.escape(phrase), text))
    
    if domain_scores:
        best_domain, best_score = max(domain_scores.items(), key=lambda x: x[1])
        return best_domain if best_score >= 0.5 else 'Other'  # Lowered threshold
    return 'Other'

class GNNClassifier(nn.Module):
    """Simple GNN for document classification"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNClassifier, self).__init__()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, edge_index):
        # Simple message passing
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        return F.log_softmax(x, dim=1)

def get_text_embedding(text):
    """Get BERT embedding for text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding='max_length')
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def create_gnn_data(graph_data):
    """Convert graph data to PyTorch Geometric format using content analysis"""
    node_features = []
    labels = []
    node_id_to_idx = {node['id']: idx for idx, node in enumerate(graph_data['nodes'])}
    
    for node in graph_data['nodes']:
        doc = documents_collection.get(node['id'])
        if doc:
            # Use content-based domain detection
            domain = extract_domain_from_content(doc.get('text', ''))
            labels.append(domain)
            
            # Get embedding from content
            embedding = get_text_embedding(doc['text'][:512])  # First 512 chars
            node_features.append(embedding)
    
    if not node_features:
        return None
    
    # Create edge indices
    edge_indices = []
    for edge in graph_data['edges']:
        src = node_id_to_idx[edge['from']]
        dst = node_id_to_idx[edge['to']]
        edge_indices.append([src, dst])
        edge_indices.append([dst, src])  # Undirected graph
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    
    le = LabelEncoder()
    y = torch.tensor(le.fit_transform(labels), dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, y=y), le

def train_gnn_model(data, le, epochs=50):
    """Train the GNN model"""
    model = GNNClassifier(input_dim=data.x.shape[1], 
                         hidden_dim=128, 
                         output_dim=len(le.classes_))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
    
    return model

@app.route('/train_gnn', methods=['POST'])
def train_gnn():
    try:
        graph_data = compute_similarity_graph()
        data, le = create_gnn_data(graph_data)
        
        if data is None:
            return jsonify({'error': 'Not enough data for GNN training'}), 400
            
        model = train_gnn_model(data, le)
        
        # Save model and label encoder (simplified example)
        torch.save(model.state_dict(), 'gnn_model.pth')
        joblib.dump(le, 'label_encoder.pkl')
        
        return jsonify({'message': 'GNN trained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_domains', methods=['GET'])
def predict_domains():
    try:
        # Load model and encoder
        model = GNNClassifier(input_dim=768, hidden_dim=128, output_dim=len(DOMAIN_DICT)+1)
        model.load_state_dict(torch.load('gnn_model.pth'))
        model.eval()
        
        le = joblib.load('label_encoder.pkl')
        
        # Get current graph data
        graph_data = compute_similarity_graph()
        data, _ = create_gnn_data(graph_data)
        
        # Predict
        with torch.no_grad():
            predictions = model(data.x, data.edge_index)
            predicted_labels = predictions.argmax(dim=1)
            domains = le.inverse_transform(predicted_labels.numpy())
        
        # Prepare results
        results = []
        for node, domain in zip(graph_data['nodes'], domains):
            results.append({
                'document_id': node['id'],
                'document_name': node['name'],
                'predicted_domain': domain
            })
        
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classify_domains', methods=['GET'])
def classify_domains():
    try:
        # Step 1: Compute the similarity graph
        graph_data = compute_similarity_graph()
        if not graph_data['nodes']:
            return jsonify({'error': 'No documents available'}), 400

        # Step 2: Prepare GNN data
        gnn_data, label_encoder = create_gnn_data(graph_data)
        if not gnn_data:
            return jsonify({'error': 'Failed to create graph data'}), 400

        # Step 3: Train the GNN
        model = train_gnn_model(gnn_data, label_encoder)

        # Step 4: Get predictions
        model.eval()
        with torch.no_grad():
            log_probs = model(gnn_data.x, gnn_data.edge_index)
            preds = log_probs.argmax(dim=1)
            domains = label_encoder.inverse_transform(preds.numpy())
            confidences = torch.exp(log_probs.max(dim=1)[0]).numpy()

        # Step 5: Prepare results and domain distribution
        results = []
        domain_counts = defaultdict(int)
        
        for i, node in enumerate(graph_data['nodes']):
            doc = documents_collection.get(node['id'])
            if doc:
                domain = domains[i]
                results.append({
                    'document_id': node['id'],
                    'document_name': doc['filename'],
                    'domain': domain,
                    'confidence': float(confidences[i])
                })
                domain_counts[domain] += 1

        return jsonify({
            'message': 'GNN classification completed',
            'results': results,
            'domain_distribution': dict(domain_counts)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def store_document(filename, text):
    doc_id = str(uuid.uuid4())
    doc_data = {
        'filename': filename,
        'text': text,
        'upload_date': datetime.now().isoformat(),
        'processed': True
    }
    documents_collection[doc_id] = doc_data
    return doc_id

def compute_similarity_graph():
    documents = list(documents_collection.values())
    if len(documents) < 2:
        return {"nodes": [], "edges": []}
    
    doc_ids = list(documents_collection.keys())
    doc_texts = [doc["text"] for doc in documents]
    doc_names = [doc["filename"] for doc in documents]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(doc_texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    G = nx.Graph()
    for idx, (doc_id, doc_name) in enumerate(zip(doc_ids, doc_names)):
        G.add_node(doc_id, label=doc_name)
    
    threshold = 0.3
    for i in range(len(doc_ids)):
        for j in range(i + 1, len(doc_ids)):
            similarity = similarity_matrix[i, j]
            if similarity > threshold:
                G.add_edge(doc_ids[i], doc_ids[j], weight=float(similarity))
    
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    pagerank = nx.pagerank(G)
    core_numbers = nx.core_number(G)

    # Calculate basic metrics
    average_degree = sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    density = nx.density(G)
    
    # Create a mapping from doc_id to doc_name
    doc_name_map = {doc_id: name for doc_id, name in zip(doc_ids, doc_names)}
    
    graph_data = {
        "nodes": [{
            "id": n, 
            "label": doc_name_map.get(n, f"Doc {n[:4]}"),  # Use document name
            "name": doc_name_map.get(n, f"Doc {n[:4]}"),  # Store name separately
            "degree": degree_centrality.get(n, 0),
            "betweenness": betweenness_centrality.get(n, 0),
            "closeness": closeness_centrality.get(n, 0),
            "eigenvector": eigenvector_centrality.get(n, 0),
            "pagerank": pagerank.get(n, 0)
        } for n in G.nodes()],
        "edges": [{"from": u, "to": v, "weight": G[u][v]['weight']} for u, v in G.edges()],
        "metrics": {
            "degree_centrality": degree_centrality,
            "betweenness_centrality": betweenness_centrality,
            "closeness_centrality": closeness_centrality,
            "eigenvector_centrality": eigenvector_centrality,
            "pagerank": pagerank,
            "core_numbers": core_numbers,
            "doc_name_map": doc_name_map,
            "average_degree": average_degree,
            "density": density,
            "number_of_nodes": G.number_of_nodes(),
            "number_of_edges": G.number_of_edges()
        }
    }
    return graph_data

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return jsonify({'error': 'No files selected'}), 400
    
    results = []
    for file in files:
        text = extract_text_from_pdf(file)
        doc_id = store_document(file.filename, text)
        results.append({'filename': file.filename, 'doc_id': str(doc_id)})
    
    graph_data = compute_similarity_graph()
    return jsonify({
        'message': 'Files are processed',
        'results': results,
        'graph': graph_data
    })

@app.route('/documents', methods=['GET'])
def get_documents():
    docs = [{'id': k, 'filename': v['filename'], 'upload_date': v['upload_date']} 
            for k, v in documents_collection.items()]
    return jsonify(docs)

@app.route('/delete_document/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    try:
        if doc_id in documents_collection:
            del documents_collection[doc_id]
            return jsonify({'message': 'Document deleted successfully'})
        return jsonify({'error': 'Document not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/graph_metrics', methods=['GET'])
def get_graph_metrics():
    graph_data = compute_similarity_graph()
    return jsonify(graph_data)

@app.route('/analyze_pair', methods=['GET'])
def analyze_pair():
    doc_id_1 = request.args.get('doc_id_1')
    doc_id_2 = request.args.get('doc_id_2')
    
    if not doc_id_1 or not doc_id_2:
        return jsonify({'error': 'Please provide two document IDs'}), 400
    
    try:
        doc_1 = documents_collection.get(doc_id_1)
        doc_2 = documents_collection.get(doc_id_2)
        
        if not doc_1 or not doc_2:
            return jsonify({'error': 'One or both documents not found'}), 404
        
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([doc_1['text'], doc_2['text']])
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()[0]
        
        return jsonify({
            'similarity_score': similarity_score,
            'doc1': doc_1['filename'],
            'doc2': doc_2['filename']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check_plagiarism', methods=['POST'])
def check_plagiarism():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Extract text from uploaded file
        query_text = extract_text_from_pdf(file)
        
        # Get all documents from database
        documents = list(documents_collection.values())
        
        if not documents:
            return jsonify({'error': 'No documents in database to compare with'}), 400
            
        # Prepare documents for comparison
        doc_texts = [doc['text'] for doc in documents]
        doc_ids = list(documents_collection.keys())
        doc_names = [doc['filename'] for doc in documents]
        
        # Add query document to comparison set
        all_texts = [query_text] + doc_texts
        
        # Calculate TF-IDF and similarities
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Compare query document with all others
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        # Prepare results
        results = []
        potential_plagiarism = False
        plagiarism_threshold = 0.7  # Consider scores above this as potential plagiarism
        
        for i, score in enumerate(similarity_scores):
            # Convert numpy float to native Python float
            score_float = float(score)
            # Convert numpy bool to native Python bool
            is_plag = bool(score_float > plagiarism_threshold)
            
            if is_plag:
                potential_plagiarism = True
                
            results.append({
                'document_id': doc_ids[i],
                'document_name': doc_names[i],
                'similarity_score': score_float,
                'is_plagiarism': is_plag
            })
        
        # Sort results by similarity score (descending)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return jsonify({
            'potential_plagiarism': potential_plagiarism,
            'results': results,
            'query_filename': file.filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)