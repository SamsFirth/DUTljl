from typing import Any, List, Optional, Set

class TextStreamer:

    def __init__(self, tokenizer: 'AutoTokenizer', skip_prompt: bool=False, **decode_kwargs):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True

    def reset(self):
        self.token_cache = []
        self.print_len = 0

    def put(self, value) -> Optional[str]:
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if not isinstance(value, int):
            raise ValueError('TextStreamer only supports batch size 1, and int type input')
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return None
        self.token_cache.append(value)
        text = self.tokenizer.decode(self.token_cache, skip_special_tokens=True, **self.decode_kwargs)
        if text.endswith('\n'):
            printable_text = text[self.print_len:]
            self.reset()
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len:]
            self.print_len += len(printable_text)
        else:
            printable_text = text[self.print_len:text.rfind(' ') + 1]
            self.print_len += len(printable_text)
        return printable_text

    def end(self) -> Optional[str]:
        """Flushes any remaining cache and prints a newline to stdout."""
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache, skip_special_tokens=True, **self.decode_kwargs)
            printable_text = text[self.print_len:]
            self.reset()
        else:
            printable_text = ''
        self.next_tokens_are_prompt = True
        return printable_text

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        if cp >= 19968 and cp <= 40959 or (cp >= 13312 and cp <= 19903) or (cp >= 131072 and cp <= 173791) or (cp >= 173824 and cp <= 177983) or (cp >= 177984 and cp <= 178207) or (cp >= 178208 and cp <= 183983) or (cp >= 63744 and cp <= 64255) or (cp >= 194560 and cp <= 195103):
            return True
        return False