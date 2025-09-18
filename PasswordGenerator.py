import string
import random

def generate_password(length):
    """Generate a random password of the given length."""
    if length < 3:
        print("Minimum password length should be 3.")
        return None
    characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(characters) for _ in range(length))

def main():
    try:
        num_passwords = int(input("How many passwords do you want to generate? "))
        if num_passwords <= 0:
            print("Please enter a positive number.")
            return

        print(f"Generating {num_passwords} passwords")

        for i in range(1, num_passwords + 1):
            while True:
                try:
                    length = int(input(f"Enter the length of Password #{i}: "))
                    if length < 3:
                        print("Minimum length of password should be 3. Please try again.")
                        continue
                    password = generate_password(length)
                    if password:
                        print(f"Password #{i}: {password}")
                        break
                except ValueError:
                    print("Please enter a valid number.")
    except ValueError:
        print("Please enter a valid number of passwords.")

if __name__ == "__main__":
    main()
