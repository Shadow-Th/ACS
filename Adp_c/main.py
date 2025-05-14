from scripts.fetch_emails import extract_email_features
from scripts.preprocess import preprocess_data
from model.train import train_model
from model.evaluate import evaluate_model
from scripts.migrate import migrate_data

def main():
    # Step 1: Extract real emails and save features to data/emails.csv
    print("🔐 Fetching emails from your inbox...")
    extract_email_features("thasneemmohammmed.64@gmail.com", "xopp isrh rdlk juea")  

    # Step 2: Preprocess the extracted features
    print("⚙️ Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data("data/emails.csv")

    # Step 3: Train the neural network
    print("🧠 Training model...")
    model = train_model(X_train, y_train)

    # Step 4: Evaluate the model
    print("📊 Evaluating performance...")
    evaluate_model(model, X_test, y_test)

    # Step 5: Migrate storage based on prediction results
    print("📦 Migrating data to storage...")
    migrate_data()

    print("✅ All steps completed.")

if __name__ == "__main__":
    main()
