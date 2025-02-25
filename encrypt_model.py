from cryptography.fernet import Fernet

# Generate a secret key (only do this once!)
key = Fernet.generate_key()
with open("secret.key", "wb") as key_file:
    key_file.write(key)

# Encrypt the model
cipher = Fernet(key)
with open("stock_predictor_model.h5", "rb") as file:
    model_data = file.read()

encrypted_data = cipher.encrypt(model_data)

with open("encrypted_model.enc", "wb") as encrypted_file:
    encrypted_file.write(encrypted_data)

print("✅ Model successfully encrypted! Delete 'stock_predictor_model.h5' now.")
