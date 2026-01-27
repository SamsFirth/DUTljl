import logging
import os
import hashlib
logger = logging.getLogger('verify-checksum-models')

def sha256sum(file):
    block_size = 16 * 1024 * 1024
    b = bytearray(block_size)
    file_hash = hashlib.sha256()
    mv = memoryview(b)
    with open(file, 'rb', buffering=0) as f:
        while True:
            n = f.readinto(mv)
            if not n:
                break
            file_hash.update(mv[:n])
    return file_hash.hexdigest()
llama_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
hash_list_file = os.path.join(llama_path, 'SHA256SUMS')
if not os.path.exists(hash_list_file):
    logger.error(f'Hash list file not found: {hash_list_file}')
    exit(1)
with open(hash_list_file, 'r') as f:
    hash_list = f.read().splitlines()
results = []
for line in hash_list:
    hash_value, filename = line.split('  ')
    file_path = os.path.join(llama_path, filename)
    logger.info(f'Verifying the checksum of {file_path}')
    if os.path.exists(file_path):
        file_hash = sha256sum(file_path)
        if file_hash == hash_value:
            valid_checksum = 'V'
            file_missing = ''
        else:
            valid_checksum = ''
            file_missing = ''
    else:
        valid_checksum = ''
        file_missing = 'X'
    results.append({'filename': filename, 'valid checksum': valid_checksum, 'file missing': file_missing})
print('filename'.ljust(40) + 'valid checksum'.center(20) + 'file missing'.center(20))
print('-' * 80)
for r in results:
    print(f"{r['filename']:40} {r['valid checksum']:^20} {r['file missing']:^20}")