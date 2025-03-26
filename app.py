from flask import Flask, request, jsonify 
from flask_cors import CORS
import requests
import os 
from dotenv import load_dotenv
import numpy as np

app = Flask(__name__)
CORS(app)

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
llm_models = ["4o", "o1", "Claude", "Gemini", "Llama", "R1", "V3", "Nova", "Qwen", "Ernie"]
question_topics = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government"]

class LLMSelector:
    def __init__(self, llms, topics, alpha=0.1, gamma=0.98, beta=0.9, temperature=0.1):
        """
        Initializes the LLM selector with default weight optimization parameters.

        Parameters:
        - llms: List of available LLMs
        - topics: List of predefined topics
        - alpha: Learning rate for weight updates
        - gamma: Decay factor for unselected LLMs
        - beta: Smoothing factor for selected LLMs
        - temperature: Softmax temperature for probabilistic selection
        """
        self.llms = llms
        self.topics = topics
        self.weights = {topic: {llm: 1 / len(llms) for llm in llms} for topic in topics}
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.temperature = temperature

    def select_llms(self, topic, k=3):
        """Selects `k` LLMs based on topic-specific softmax-weighted probabilities."""
        if topic not in self.weights:
            topic = "general"  # Default case

        llm_weights = np.array(list(self.weights[topic].values()))
        probs = np.exp(llm_weights / self.temperature)
        probs /= probs.sum()  # Normalize

        return np.random.choice(list(self.weights[topic].keys()), size=k, p=probs, replace=False)

    def update_weights(self, topic, selected_llm):
        """Updates the weights of the LLMs based on user selection."""
        if topic not in self.weights:
            self.weights[topic] = {llm: 1 / len(self.llms) for llm in self.llms}

        for llm in self.weights[topic]:
            if llm == selected_llm:
                self.weights[topic][llm] = self.beta * self.weights[topic][llm] + (1 - self.beta)  # Reward
            else:
                self.weights[topic][llm] *= self.gamma  # Apply decay to unselected LLMs

class LLMSystem:
    def __init__(self):
        """Initialize the LLM selector"""
        self.llm_selector = LLMSelector(llm_models, question_topics)

    def get_topic(self, question):
        # result = classifier(question, candidate_labels=question_topics)
        payload = {"inputs": question, "parameters": {"candidate_labels": question_topics}}
        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()
        
        max_index = result['scores'].index(max(result['scores']))
        return result['labels'][max_index]

    def select_llm(self, question):
        """Processes a new question and selects LLMs"""
        topic = self.get_topic(question)
        selected_llms = self.llm_selector.select_llms(topic)
        print(f"Selected LLMs for topic '{topic}': {selected_llms}")
        return selected_llms, topic

    def update_system(self, user_selected_llm, topic):
        # Simulating user choice
        # user_selected_llm = np.random.choice(selected_llms)
        print(f"User selected response from: {user_selected_llm}")

        # Update weights based on user selection
        self.llm_selector.update_weights(topic, user_selected_llm)
        # print(f"Updated weights for topic '{topic}': {self.llm_selector.weights[topic]}")
        print(f"Updated weights for topic '{topic}': {dict(sorted(self.llm_selector.weights[topic].items(), key=lambda item: item[1], reverse=True))}")

llm_system = LLMSystem()

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/select-llms", methods=["GET"])
def respond_select_llms():
    question = request.args.get('question')
    llms, topic = llm_system.select_llm(question)
    return jsonify({'llms': llms.tolist(), 'topic': topic})

@app.route("/update-system", methods=["GET"])
def respond_update_system():
    user_selected_llm = request.args.get('llm')
    topic = request.args.get('topic')
    llm_system.update_system(user_selected_llm, topic)
    return jsonify({'topic': topic})