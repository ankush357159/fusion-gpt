import tiktoken


class TiktokenTokenizer:
    def __init__(
        self,
        encoding_name="gpt2",
        bos_token_id=50256,
        eos_token_id=50256,
        pad_token_id=50256,
    ):
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab

        # GPT-2 uses <|endoftext|> for BOS/EOS/PAD conventionally
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def encode(
        self,
        text,
        add_bos=False,
        add_eos=False,
        allowed_special={"<|endoftext|>"},
        disallowed_special=(),
    ):
        tokens = self.enc.encode(
            text,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
        )

        if add_bos:
            tokens = [self.bos_token_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_token_id]

        return tokens

    def decode(self, tokens):
        return self.enc.decode(tokens)

    def encode_batch(self, texts, **kwargs):
        return [self.encode(text, **kwargs) for text in texts]

    def decode_batch(self, batch_tokens):
        return [self.decode(tokens) for tokens in batch_tokens]

    def truncate(self, tokens, max_length):
        return tokens if max_length is None else tokens[:max_length]

    def pad_to_max_length(self, tokens, max_length):
        if max_length is None:
            return tokens
        if len(tokens) >= max_length:
            return tokens[:max_length]
        return tokens + [self.pad_token_id] * (max_length - len(tokens))
