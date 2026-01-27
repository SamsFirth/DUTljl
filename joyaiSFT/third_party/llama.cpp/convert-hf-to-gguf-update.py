import logging
import os
import pathlib
import re
import requests
import sys
import json
from hashlib import sha256
from enum import IntEnum, auto
from transformers import AutoTokenizer
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('convert-hf-to-gguf-update')
sess = requests.Session()

class TOKENIZER_TYPE(IntEnum):
    SPM = auto()
    BPE = auto()
    WPM = auto()
chktxt = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u200d🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````""""......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'
if len(sys.argv) == 2:
    token = sys.argv[1]
    if not token.startswith('hf_'):
        logger.info('Huggingface token seems invalid')
        logger.info('Usage: python convert-hf-to-gguf-update.py <huggingface_token>')
        sys.exit(1)
else:
    logger.info('Usage: python convert-hf-to-gguf-update.py <huggingface_token>')
    sys.exit(1)
models = [{'name': 'llama-spm', 'tokt': TOKENIZER_TYPE.SPM, 'repo': 'https://huggingface.co/meta-llama/Llama-2-7b-hf'}, {'name': 'llama-bpe', 'tokt': TOKENIZER_TYPE.BPE, 'repo': 'https://huggingface.co/meta-llama/Meta-Llama-3-8B'}, {'name': 'phi-3', 'tokt': TOKENIZER_TYPE.SPM, 'repo': 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct'}, {'name': 'deepseek-llm', 'tokt': TOKENIZER_TYPE.BPE, 'repo': 'https://huggingface.co/deepseek-ai/deepseek-llm-7b-base'}, {'name': 'deepseek-coder', 'tokt': TOKENIZER_TYPE.BPE, 'repo': 'https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base'}, {'name': 'falcon', 'tokt': TOKENIZER_TYPE.BPE, 'repo': 'https://huggingface.co/tiiuae/falcon-7b'}, {'name': 'bert-bge', 'tokt': TOKENIZER_TYPE.WPM, 'repo': 'https://huggingface.co/BAAI/bge-small-en-v1.5'}, {'name': 'mpt', 'tokt': TOKENIZER_TYPE.BPE, 'repo': 'https://huggingface.co/mosaicml/mpt-7b'}, {'name': 'starcoder', 'tokt': TOKENIZER_TYPE.BPE, 'repo': 'https://huggingface.co/bigcode/starcoder2-3b'}, {'name': 'gpt-2', 'tokt': TOKENIZER_TYPE.BPE, 'repo': 'https://huggingface.co/openai-community/gpt2'}, {'name': 'stablelm2', 'tokt': TOKENIZER_TYPE.BPE, 'repo': 'https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b'}, {'name': 'refact', 'tokt': TOKENIZER_TYPE.BPE, 'repo': 'https://huggingface.co/smallcloudai/Refact-1_6-base'}, {'name': 'command-r', 'tokt': TOKENIZER_TYPE.BPE, 'repo': 'https://huggingface.co/CohereForAI/c4ai-command-r-v01'}, {'name': 'qwen2', 'tokt': TOKENIZER_TYPE.BPE, 'repo': 'https://huggingface.co/Qwen/Qwen1.5-7B'}, {'name': 'olmo', 'tokt': TOKENIZER_TYPE.BPE, 'repo': 'https://huggingface.co/allenai/OLMo-1.7-7B-hf'}, {'name': 'dbrx', 'tokt': TOKENIZER_TYPE.BPE, 'repo': 'https://huggingface.co/databricks/dbrx-base'}, {'name': 'jina-v2-en', 'tokt': TOKENIZER_TYPE.WPM, 'repo': 'https://huggingface.co/jinaai/jina-embeddings-v2-base-en'}, {'name': 'jina-v2-es', 'tokt': TOKENIZER_TYPE.BPE, 'repo': 'https://huggingface.co/jinaai/jina-embeddings-v2-base-es'}, {'name': 'jina-v2-de', 'tokt': TOKENIZER_TYPE.BPE, 'repo': 'https://huggingface.co/jinaai/jina-embeddings-v2-base-de'}, {'name': 'smaug-bpe', 'tokt': TOKENIZER_TYPE.BPE, 'repo': 'https://huggingface.co/abacusai/Smaug-Llama-3-70B-Instruct'}, {'name': 'poro-chat', 'tokt': TOKENIZER_TYPE.BPE, 'repo': 'https://huggingface.co/LumiOpen/Poro-34B-chat'}, {'name': 'jina-v2-code', 'tokt': TOKENIZER_TYPE.BPE, 'repo': 'https://huggingface.co/jinaai/jina-embeddings-v2-base-code'}]

def download_file_with_auth(url, token, save_path):
    headers = {'Authorization': f'Bearer {token}'}
    response = sess.get(url, headers=headers)
    response.raise_for_status()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    logger.info(f'File {save_path} downloaded successfully')

def download_model(model):
    name = model['name']
    repo = model['repo']
    tokt = model['tokt']
    os.makedirs(f'models/tokenizers/{name}', exist_ok=True)
    files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
    if tokt == TOKENIZER_TYPE.SPM:
        files.append('tokenizer.model')
    for file in files:
        save_path = f'models/tokenizers/{name}/{file}'
        if os.path.isfile(save_path):
            logger.info(f'{name}: File {save_path} already exists - skipping')
            continue
        download_file_with_auth(f'{repo}/resolve/main/{file}', token, save_path)
for model in models:
    try:
        download_model(model)
    except Exception as e:
        logger.error(f"Failed to download model {model['name']}. Error: {e}")
src_ifs = ''
for model in models:
    name = model['name']
    tokt = model['tokt']
    if tokt == TOKENIZER_TYPE.SPM:
        continue
    if not os.path.exists(f'models/tokenizers/{name}'):
        logger.warning(f'Directory for tokenizer {name} not found. Skipping...')
        continue
    try:
        tokenizer = AutoTokenizer.from_pretrained(f'models/tokenizers/{name}')
    except OSError as e:
        logger.error(f'Error loading tokenizer for model {name}. The model may not exist or is not accessible with the provided token. Error: {e}')
        continue
    chktok = tokenizer.encode(chktxt)
    chkhsh = sha256(str(chktok).encode()).hexdigest()
    logger.info(f'model: {name}')
    logger.info(f'tokt: {tokt}')
    logger.info(f"repo: {model['repo']}")
    logger.info(f'chktok: {chktok}')
    logger.info(f'chkhsh: {chkhsh}')
    with open(f'models/tokenizers/{name}/tokenizer.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
        normalizer = cfg['normalizer']
        logger.info('normalizer: ' + json.dumps(normalizer, indent=4))
        pre_tokenizer = cfg['pre_tokenizer']
        logger.info('pre_tokenizer: ' + json.dumps(pre_tokenizer, indent=4))
        if 'ignore_merges' in cfg['model']:
            logger.info('ignore_merges: ' + json.dumps(cfg['model']['ignore_merges'], indent=4))
    logger.info('')
    src_ifs += f'        if chkhsh == "{chkhsh}":\n'
    src_ifs += f"            # ref: {model['repo']}\n"
    src_ifs += f'            res = "{name}"\n'
src_func = f"""\n    def get_vocab_base_pre(self, tokenizer) -> str:\n        # encoding this string and hashing the resulting tokens would (hopefully) give us a unique identifier that\n        # is specific for the BPE pre-tokenizer used by the model\n        # we will use this unique identifier to write a "tokenizer.ggml.pre" entry in the GGUF file which we can\n        # use in llama.cpp to implement the same pre-tokenizer\n\n        chktxt = {repr(chktxt)}\n\n        chktok = tokenizer.encode(chktxt)\n        chkhsh = sha256(str(chktok).encode()).hexdigest()\n\n        logger.debug(f"chktok: {{chktok}}")\n        logger.debug(f"chkhsh: {{chkhsh}}")\n\n        res = None\n\n        # NOTE: if you get an error here, you need to update the convert-hf-to-gguf-update.py script\n        #       or pull the latest version of the model from Huggingface\n        #       don't edit the hashes manually!\n{src_ifs}\n        if res is None:\n            logger.warning("\\n")\n            logger.warning("**************************************************************************************")\n            logger.warning("** WARNING: The BPE pre-tokenizer was not recognized!")\n            logger.warning("**          There are 2 possible reasons for this:")\n            logger.warning("**          - the model has not been added to convert-hf-to-gguf-update.py yet")\n            logger.warning("**          - the pre-tokenization config has changed upstream")\n            logger.warning("**          Check your model files and convert-hf-to-gguf-update.py and update them accordingly.")\n            logger.warning("** ref:     https://github.com/ggerganov/llama.cpp/pull/6920")\n            logger.warning("**")\n            logger.warning(f"** chkhsh:  {{chkhsh}}")\n            logger.warning("**************************************************************************************")\n            logger.warning("\\n")\n            raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")\n\n        logger.debug(f"tokenizer.ggml.pre: {{repr(res)}}")\n        logger.debug(f"chkhsh: {{chkhsh}}")\n\n        return res\n"""
convert_py_pth = pathlib.Path('convert-hf-to-gguf.py')
convert_py = convert_py_pth.read_text()
convert_py = re.sub('(# Marker: Start get_vocab_base_pre)(.+?)( +# Marker: End get_vocab_base_pre)', lambda m: m.group(1) + src_func + m.group(3), convert_py, flags=re.DOTALL | re.MULTILINE)
convert_py_pth.write_text(convert_py)
logger.info('+++ convert-hf-to-gguf.py was updated')
tests = ['ied 4 ½ months', 'Führer', '', ' ', '  ', '   ', '\t', '\n', '\n\n', '\n\n\n', '\t\n', 'Hello world', ' Hello world', 'Hello World', ' Hello World', ' Hello World!', 'Hello, world!', ' Hello, world!', ' this is 🦙.cpp', 'w048 7tuijk dsdfhu', 'нещо на Български', 'កាន់តែពិសេសអាចខលចេញ', '🚀 (normal) 😶\u200d🌫️ (multiple emojis concatenated) ✅ (only emoji that has its own token)', 'Hello', ' Hello', '  Hello', '   Hello', '    Hello', '    Hello\n    Hello', ' (', '\n =', "' era", "Hello, y'all! How are you 😁 ?我想在apple工作1314151天～", '3', '33', '333', '3333', '33333', '333333', '3333333', '33333333', '333333333', chktxt]
for model in models:
    name = model['name']
    tokt = model['tokt']
    if not os.path.exists(f'models/tokenizers/{name}'):
        logger.warning(f'Directory for tokenizer {name} not found. Skipping...')
        continue
    try:
        tokenizer = AutoTokenizer.from_pretrained(f'models/tokenizers/{name}')
    except OSError as e:
        logger.error(f'Failed to load tokenizer for model {name}. Error: {e}')
        continue
    with open(f'models/ggml-vocab-{name}.gguf.inp', 'w', encoding='utf-8') as f:
        for text in tests:
            f.write(f'{text}')
            f.write('\n__ggml_vocab_test__\n')
    with open(f'models/ggml-vocab-{name}.gguf.out', 'w') as f:
        for text in tests:
            res = tokenizer.encode(text, add_special_tokens=False)
            for r in res:
                f.write(f' {r}')
            f.write('\n')
    logger.info(f'Tests for {name} written in ./models/ggml-vocab-{name}.gguf.*')
logger.info('\nRun the following commands to generate the vocab files for testing:\n')
for model in models:
    name = model['name']
    print(f'python3 convert-hf-to-gguf.py models/tokenizers/{name}/ --outfile models/ggml-vocab-{name}.gguf --vocab-only')
logger.info('\n')