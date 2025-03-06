import nest_asyncio

nest_asyncio.apply()

from flask import Flask, request, render_template, jsonify
import yaml
import threading
import imaplib
import email
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)


# Configuration
class Config:
    CREDENTIALS_FILE = "credentials.yml"
    EMAILS_FILE = "emails.yml"
    IMAP_URL = "imap.gmail.com"
    DATASET_FILE = "mail_datasets.csv"


# Logging setup
logging.basicConfig(level=logging.INFO)


# Load credentials
def load_credentials():
    try:
        with open(Config.CREDENTIALS_FILE, "r") as f:
            credentials = yaml.safe_load(f) or {}
        return credentials.get("user"), credentials.get("password")
    except FileNotFoundError:
        logging.error("Credentials file not found!")
        return None, None


# Load target email
def load_target_email():
    try:
        with open(Config.EMAILS_FILE, "r") as f:
            data = yaml.safe_load(f) or {}
            return data.get("target_email", "No email set")
    except FileNotFoundError:
        return "No email set"


# Update target email
def update_target_email(new_email):
    try:
        with open(Config.EMAILS_FILE, "r") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        data = {}

    data["target_email"] = new_email
    with open(Config.EMAILS_FILE, "w") as f:
        yaml.dump(data, f)

    logging.info(f"Updated target email: {new_email}")


# Fetch latest email from target email
def fetch_latest_email(target_email):
    user, password = load_credentials()
    if not user or not password:
        logging.error("Missing email credentials.")
        return []

    try:
        with imaplib.IMAP4_SSL(Config.IMAP_URL) as my_mail:
            my_mail.login(user, password)
            my_mail.select("Inbox")

            search_criterion = f'FROM "{target_email}"'
            _, data = my_mail.search(None, search_criterion)

            mail_ids = data[0].split()
            if not mail_ids:
                logging.info(f"No emails found from {target_email}.")
                return []

            latest_email_id = mail_ids[-1]
            _, msg_data = my_mail.fetch(latest_email_id, "(RFC822)")
            email_bodies = []

            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    my_msg = email.message_from_bytes(response_part[1])
                    for part in my_msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode(part.get_content_charset(), errors="ignore")
                            email_bodies.append(body.strip())

            return email_bodies

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return []


# Train the spam detection model
def train_model():
    df = pd.read_csv(Config.DATASET_FILE)
    data = df.where(pd.notnull(df), '')

    data.loc[data['Category'] == 'spam', 'Category'] = 0
    data.loc[data['Category'] == 'ham', 'Category'] = 1

    X = data['Message']
    Y = data['Category'].astype('int')

    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_features = feature_extraction.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_features, Y)

    return model, feature_extraction


# Run model prediction
def classify_emails(email_bodies, model, feature_extraction):
    results = []
    for body in email_bodies:
        input_data_features = feature_extraction.transform([body])
        prediction = model.predict(input_data_features)[0]
        results.append({"prediction": "Ham ✅" if prediction == 1 else "Spam 🚨", "email": body[:200]})
    return results


# Train model once on startup
spam_model, vectorizer = train_model()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        target_email = request.form["mixedInput"]
        update_target_email(target_email)

        emails = fetch_latest_email(target_email)
        predictions = classify_emails(emails, spam_model, vectorizer)

        return jsonify({"target_email": target_email, "predictions": predictions})

    target_email = load_target_email()
    return render_template("index.html", target_email=target_email)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
