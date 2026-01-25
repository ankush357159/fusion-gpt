import tiktoken


class TiktokenTokenizer:
    def __init__(
        self,
        encoding_name="cl100k_base",
        bos_token_id=None,
        eos_token_id=None,
    ):
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def encode(
        self,
        text,
        add_bos=False,
        add_eos=False,
        allowed_special=(),
        disallowed_special="all",
    ):
        tokens = self.enc.encode(
            text,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
        )

        if add_bos:
            if self.bos_token_id is None:
                raise ValueError("bos_token_id is not set")
            tokens = [self.bos_token_id] + tokens

        if add_eos:
            if self.eos_token_id is None:
                raise ValueError("eos_token_id is not set")
            tokens = tokens + [self.eos_token_id]

        return tokens

    def decode(self, tokens):
        return self.enc.decode(tokens)

    def encode_batch(
        self,
        texts,
        add_bos=False,
        add_eos=False,
        allowed_special=(),
        disallowed_special="all",
    ):
        return [
            self.encode(
                text,
                add_bos=add_bos,
                add_eos=add_eos,
                allowed_special=allowed_special,
                disallowed_special=disallowed_special,
            )
            for text in texts
        ]

    def decode_batch(self, batch_tokens):
        return [self.decode(tokens) for tokens in batch_tokens]

    def truncate(self, tokens, max_length):
        if max_length is None:
            return tokens
        return tokens[:max_length]

    def pad_to_max_length(self, tokens, max_length, pad_token_id=0):
        if max_length is None:
            return tokens
        if len(tokens) >= max_length:
            return tokens[:max_length]
        return tokens + [pad_token_id] * (max_length - len(tokens))
