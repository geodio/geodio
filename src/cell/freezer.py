import os
import base64
import json
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from src.cell.cell import t_cell, Cell


class InvalidFrozenCell(Exception):
    pass


class KeyNotFound(Exception):
    pass


KEY_FILE = "encryption_key.json"
# todo generate random salt
SALT = b'some_salt'


def load_key():
    if not os.path.exists(KEY_FILE):
        raise KeyNotFound("Encryption key not found, generating a new key.")
    with open(KEY_FILE, "rb") as f:
        return f.read()


def save_key(key):
    with open(KEY_FILE, "wb") as f:
        f.write(key)


def generate_key():
    backend = default_backend()
    salt = os.urandom(16)
    kdf = Scrypt(
        salt=salt,
        length=32,
        n=2 ** 14,
        r=8,
        p=1,
        backend=backend
    )
    key = kdf.derive(SALT)
    print(f"SALT: {SALT}")
    save_key(json.dumps({"key": base64.b64encode(key).decode(),
                         "salt": base64.b64encode(salt).decode()}))
    return key


def get_key():
    try:
        key_data = load_key()
    except KeyNotFound:
        print("KEY NOT FOUND, GENERATING NEW KEYS")
        key_data = generate_key()
    key_data = json.loads(key_data)
    return base64.b64decode(key_data["key"]), base64.b64decode(
        key_data["salt"])


def encrypt(data, key, salt):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = kdf.derive(key)
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv),
                    backend=default_backend())
    encryptor = cipher.encryptor()
    return iv + encryptor.update(data) + encryptor.finalize()


def decrypt(data, key, salt):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = kdf.derive(key)
    iv = data[:16]
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv),
                    backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(data[16:]) + decryptor.finalize()


def defroze(frozen: str) -> t_cell:
    key, salt = get_key()
    encrypted_bytes = base64.b64decode(frozen)
    serialized_cell = decrypt(encrypted_bytes, key, salt)
    return Cell.from_bytes(serialized_cell)


def freeze(x: t_cell) -> str:
    if t_cell.frozen is not None:
        return t_cell.frozen
    clone = x.clone()
    for weight in clone.get_weights():
        weight.set_to_zero()
    serialized_cell = clone.to_bytes()
    key, salt = get_key()
    encrypted_bytes = encrypt(serialized_cell, key, salt)
    frozen = base64.b64encode(encrypted_bytes).decode('utf-8')
    t_cell.frozen = frozen
    return frozen
