import random
from locust import HttpUser, task, between

API_HOST = (
    "https://yanlufpsad.execute-api.us-east-1.amazonaws.com"
)

API_ENDPOINT_PATH = "/sentimentAnalysisONNX"


class EmotionRecognitionUser(HttpUser):
    """
    Simulates a user sending text to the emotion recognition API.
    """

    host = API_HOST

    wait_time = between(1, 3)

    def on_start(self):
        """
        Called when a new user starts. Prepares a pool of realistic text samples.
        This represents varied user inputs.
        """
        self.text_samples = [
            "I am incredibly happy and excited about the new project!",  # Positive, short
            "This is the best day of my life, everything is going perfectly.",  # Positive, medium
            "I'm feeling a bit down today, nothing seems to be going right.",  # Negative, medium
            "The service was terrible and I am extremely disappointed with the outcome.",  # Negative, long
            "I am absolutely furious about the recent changes. This is unacceptable.",  # Anger
            "The upcoming presentation is making me very anxious and nervous.",  # Fear/Anxiety
            "The movie was okay, neither good nor bad. It was just average.",  # Neutral
            "The weather is 72 degrees and sunny.",  # Neutral
            "This is a sentence.",  # Neutral, short
            "Locust is a powerful open-source load testing tool written in Python, which allows you to define user behaviour with Python code and swarm your system with millions of simultaneous users.",  # Neutral, long
        ]

    @task
    def analyze_emotion(self):
        """
        Picks a random text sample and sends it to the API endpoint for analysis.
        """
        text_to_analyze = random.choice(self.text_samples)

        payload = {"text": text_to_analyze}

        self.client.post(API_ENDPOINT_PATH, json=payload, name="Analyze Emotion API")
