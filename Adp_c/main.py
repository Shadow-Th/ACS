from scripts.fetch_emails import extract_email_features
from scripts.preprocess import preprocess_data
from model.train import train_model
from model.evaluate import evaluate_model
from scripts.migrate import migrate_data

def main():
    print("🔐 Fetching emails from your inbox...")
    extract_email_features("", "")  
    
    print("⚙️ Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data("data/emails.csv")


    print("🧠 Training model...")
    model = train_model(X_train, y_train)

    
    print("📊 Evaluating performance...")
    evaluate_model(model, X_test, y_test)

    
    print("📦 Migrating data to storage...")
    migrate_data()

    print("✅ All steps completed.")

if __name__ == "__main__":
    main()
