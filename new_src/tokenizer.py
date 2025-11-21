# src/tokenizer.py

import json
import re

class Tokenizer:
    """
    Simple vocabulary-based tokenizer compatible with your normalized traces.
    - Uses a fixed vocab loaded from JSON.
    - Splits on whitespace.
    - Unknown tokens -> <unk> id (1).
    - Padding token id = 0.

    Input text is already normalized (timestamp|TIMESTAMP|PID|...)
    so this tokenizer ONLY splits token-by-token and maps using vocab.
    """

    def __init__(self, vocab_path):
        raw = json.load(open(vocab_path, "r", encoding="utf-8"))

        # Accept both dict and list
        if isinstance(raw, dict):
            self.token2id = raw
        elif isinstance(raw, list):
            self.token2id = {tok: i for i, tok in enumerate(raw)}
        else:
            raise TypeError(f"Unexpected vocab type: {type(raw)}")

        self.vocab = self.token2id 

        # Required special tokens
        self.pad_id = self.vocab.get("<pad>", 0)
        self.unk_id = self.vocab.get("<unk>", 1)

        # reverse lookup for decode()
        self.inv_vocab = {int(v): k for k, v in self.vocab.items()}

    # ---------------------------------------------------------
    # Basic text → tokens
    # ---------------------------------------------------------
    def tokenize(self, text):
        """
        Tokenize a normalized trace.
        Normalized format already breaks structure into tokens
        separated by whitespace and '|||' markers.
        Example token:
          'module|KERNEL32.dll|function|Sleep'
        """
        if not isinstance(text, str):
            return []

        # Split simply on whitespace — normalization ensures good boundaries
        toks = text.strip().split()
        return toks

    # ---------------------------------------------------------
    # Map tokens → ids
    # ---------------------------------------------------------
    def encode(self, text, max_len):
        """
        Convert a trace string to a padded list of token IDs.
        """
        tokens = self.tokenize(text)
        ids = [self.vocab.get(tok, self.unk_id) for tok in tokens]

        # Truncate
        if len(ids) > max_len:
            ids = ids[:max_len]

        # Pad to max_len
        while len(ids) < max_len:
            ids.append(self.pad_id)

        return ids

    # ---------------------------------------------------------
    # Map ids → tokens
    # ---------------------------------------------------------
    def decode(self, ids):
        toks = []
        for i in ids:
            tok = self.inv_vocab.get(int(i), "<unk>")
            toks.append(tok)
        return toks

    # ---------------------------------------------------------
    # For debugging: return tokens without converting to ids
    # ---------------------------------------------------------
    def tokens_from_text(self, text):
        return self.tokenize(text)
