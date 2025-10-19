import re
import torch
import emoji
import torch.nn.functional as F
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import nltk

class SentimentAnalyser:
    def __init__(self, model="unitary/toxic-bert"):
        self.model = model

    def compute_emoji_score(self, text):
        """
        Compute an additional sentiment score based on emojis and emoticons.
        Adjust these values based on domain knowledge or experiments.
        """
        demojized = emoji.demojize(text)
        emoji_score = 0.0

        # Emoji sentiment mapping (ensure keys match demojize output!)
        emoji_sentiment = {
            # Positive
            ":smiley:": 0.4,
            ":smile:": 0.3,
            ":grinning:": 0.4,
            ":grin:": 0.5,
            ":blush:": 0.4,
            ":innocent:": 0.3,
            ":heart_eyes:": 0.6,
            ":kissing_heart:": 0.4,
            ":kissing:": 0.3,
            ":star_struck:": 0.5,
            ":yum:": 0.3,
            ":stuck_out_tongue:": 0.3,
            ":zany_face:": 0.4,
            ":wink:": 0.2,
            ":+1:": 0.4,
            ":clap:": 0.4,
            ":pray:": 0.3,
            ":fire:": 0.7,
            ":sparkles:": 0.4,
            ":100:": 0.6,
            ":tada:": 0.5,
            ":trophy:": 0.5,
            ":partying_face:": 0.5,
            ":sun_with_face:": 0.4,
            ":rocket:": 0.5,
            ":boom:": 0.4,
            ":laughing:": 0.5,
            ":rofl:": 0.6,
            ":joy:": 0.6,

            # Neutral
            ":neutral_face:": 0.0,
            ":expressionless:": 0.0,
            ":thinking:": 0.0,
            ":no_mouth:": 0.0,
            ":raised_eyebrow:": 0.0,
            ":monocle_face:": 0.0,
            ":zipper_mouth_face:": 0.0,

            # Negative
            ":pensive:": -0.3,
            ":disappointed:": -0.3,
            ":worried:": -0.3,
            ":confused:": -0.3,
            ":slightly_frowning_face:": -0.3,
            ":frowning_face:": -0.4,
            ":persevere:": -0.4,
            ":cry:": -0.4,
            ":sob:": -0.5,
            ":angry:": -0.6,
            ":symbols_over_mouth:": -0.7,
            ":rage:": -0.8,
            ":pouting_face:": -0.8,
            ":scream:": -0.6,
            ":head_bandage:": -0.5,
            ":exploding_head:": -0.6,
            ":skull:": -0.6,
            ":poop:": -0.5,
            ":face_vomiting:": -0.6,
            ":thermometer_face:": -0.5,
            ":nauseated_face:": -0.5,
        }

        for emoji_alias, score in emoji_sentiment.items():
            count = demojized.count(emoji_alias)
            if count > 0:
                emoji_score += score * count

        # Common ASCII emoticons
        if re.search(r'(:\)|:-\))', text):
            emoji_score += 0.1
        if re.search(r'(:\(|:-\()', text):
            emoji_score -= 0.1
        if re.search(r'(:D|:-D)', text):
            emoji_score += 0.1

        return emoji_score


    def compute_question_mark_score(self, text):
        """
        Compute penalty for repeated question marks.
        """
        qm_count = text.count('?')
        penalty = -0.05 * max(qm_count - 1.0, 0.0)
        return max(penalty, -0.3)  # cap penalty


    def compute_combined_score(self, text):
        """
        Return a sentiment score in [0, 5] integrating multiple signals.
        """
        try:
            # Lazy init models
            if not hasattr(self, "sia"):
                self.sia = SentimentIntensityAnalyzer()
            if not hasattr(self, "tox_model"):
                self.tox_tokenizer = AutoTokenizer.from_pretrained(self.model)
                self.tox_model = AutoModelForSequenceClassification.from_pretrained(self.model)

            sia = self.sia
            tox_model_local = self.tox_model
            tox_tokenizer_local = self.tox_tokenizer


            # Text sentiment
            text_sentiment = sia.polarity_scores(text)["compound"]

            # Emoji sentiment
            emoji_sent = self.compute_emoji_score(text)
            if (text_sentiment < 0 and emoji_sent > 0) or (text_sentiment > 0 and emoji_sent < 0):
                adjusted_emoji_sent = emoji_sent * 0.5
            else:
                adjusted_emoji_sent = emoji_sent * 2  # reduced scaling

            # Other features
            length_score = min(len(text) / 500, 1.0)
            exclam_score = min(text.count("!") / 5.0, 1.0)
            qm_score = self.compute_question_mark_score(text)

            # Toxicity
            inputs = tox_tokenizer_local(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                logits = tox_model_local(**inputs).logits
            probs = F.softmax(logits, dim=1)
            toxicity_prob = probs[0][1].item()
            toxicity_score = (toxicity_prob - 0.5) * 2  # normalize to [-1,1]

            # Weights
            w_text, w_length, w_exclam, w_emoji, w_toxicity, w_qm = 0.45, 0.01, 0.05, 0.2, 0.25, 0.05

            # Redistribute if signal = 0
            if length_score == 0: w_text += w_length; w_length = 0
            if exclam_score == 0: w_text += w_exclam; w_exclam = 0
            if adjusted_emoji_sent == 0: w_text += w_emoji; w_emoji = 0
            if toxicity_score == 0: w_text += w_toxicity; w_toxicity = 0
            if qm_score == 0: w_text += w_qm; w_qm = 0

            # Weighted sum
            raw = (w_text * text_sentiment +
                w_length * length_score +
                w_exclam * exclam_score +
                w_emoji * adjusted_emoji_sent -
                w_toxicity * toxicity_score +
                w_qm * qm_score)

            # Gentle amplification
            if text_sentiment > 0 and (adjusted_emoji_sent > 0 or exclam_score > 0 or toxicity_score <= 0):
                raw *= 1.2
            elif text_sentiment < 0 and (adjusted_emoji_sent < 0 or toxicity_score > 0 or qm_score < 0):
                raw *= 1.2

            # Clamp and scale to [0,5]
            raw = max(min(raw, 1), -1)
            scaled = (raw + 1) * 2.5
            return max(min(scaled, 5.0), 0.0)

        except Exception as e:
            print("Error in compute_combined_score:", e)
            return 2.5
