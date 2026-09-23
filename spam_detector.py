import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only the required columns
data = data.iloc[:, :2]
data.columns = ["label", "message"]

# Convert labels to numbers
data["label"] = data["label"].map({"ham": 0, "spam": 1})
extra_spam = [
    "Congratulations! You have won a £1000 cash prize. Click here to claim now!",
    "URGENT! Your account has been selected for a special reward. Claim now!",
    "You have won an iPhone! Click the link to receive your free prize.",
    "FREE MONEY! Get £5000 instantly. Apply now!",
    "Winner! You have been selected for a cash reward. Click here.",
    "Claim your FREE gift card now! Limited time offer.",
    "You have been selected to receive a £500 Amazon voucher. Claim today!",
    "Congratulations! You are today's lucky winner. Collect your prize now.",
    "URGENT: Your bank account has been suspended. Click here to verify.",
    "Your PayPal account has been locked. Verify your details immediately.",
    "Security alert! Confirm your banking information now to avoid suspension.",
    "Your account will be closed unless you verify your information today.",
    "Unusual activity detected. Click the link to secure your account.",
    "Your payment failed. Update your card details immediately.",
    "You are eligible for an instant loan. No credit check required!",
    "Get rich fast! Earn £500 per day working from home.",
    "Exclusive investment opportunity! Double your money today.",
    "You have been approved for a £10,000 loan. Click to receive funds.",
    "Limited offer! Buy now and receive 90% discount.",
    "ACT NOW! This amazing offer expires today. Click here!",
    "Free entry into our £1 million prize draw. Enter now!",
    "You've won! Send your bank details to receive your prize.",
    "Congratulations winner! Pay a small processing fee to claim your reward.",
    "Click this link immediately to receive your free bonus.",
    "Dear customer, your parcel is waiting. Pay £1.99 to arrange delivery.",
    "Your package could not be delivered. Click here to reschedule.",
    "HMRC refund available! Enter your bank details to receive your tax refund.",
    "You are entitled to a tax refund. Claim your money now!",
    "FINAL WARNING: Your account will be deleted. Verify now!",
    "Your subscription has expired. Click here to restore access immediately."
]
extra_data = pd.DataFrame({
"label":[1] * len(extra_spam),
"message": extra_spam
})
data = pd.concat([data, extra_data], ignore_index=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    data["message"],
    data["label"],
    test_size=0.2,
    random_state=42
)

# Convert email text into numbers
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the spam detection model
model = MultinomialNB(alpha=0.5)
model.fit(X_train_vectorized, y_train)

joblib.dump(model, "spam_model.pk1")
joblib.dump(vectorizer, "vectorizer.pk1")

# Test accuracy
predictions = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, predictions)

print("Spam Email Blocker")
print("-------------------")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Check emails continuously
while True:
    email = input("\nEnter an email message (or type quit): ")

    if email.lower() == "quit":
        print("Program closed.")
        break

    email_vectorized = vectorizer.transform([email])
    prediction = model.predict(email_vectorized)

    if prediction[0] == 1:
        print("SPAM EMAIL")
    else:
        print("NOT SPAM")