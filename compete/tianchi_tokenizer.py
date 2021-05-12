
import transformers
class TTokenizer(transformers.PreTrainedTokenizer):
    def __init__(self,vocab_to_id,vocab_to_str, bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="<unk>",
        sep_token="[SEP]",
        pad_token="<pad>",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs):
        super().__init__(bos_token=bos_token,
                        eos_token=eos_token,
                        unk_token=unk_token,
                        sep_token=sep_token,
                        pad_token=pad_token,
                        cls_token=cls_token,
                        mask_token=mask_token,
                        **kwargs)
        self.tokens_encode = vocab_to_id
        self.tokens_decode = vocab_to_str
        self.sanitize_special_tokens()


    @property
    def vocab_size(self):
        return len(self.tokens_encode)

    def _tokenize(self, text, **kwargs):
        text = text.split(";")[0]
        text = text.strip()
        split_text = text.split(" ")
        return split_text
    
    def _convert_id_to_token(self, index: int):
        if index not in self.tokens_decode.keys():
            return None
        return self.tokens_decode[index]

    def _convert_token_to_id(self, token):
        if token not in self.tokens_encode.keys():
            return None
        return self.tokens_encode[token]

    def save_vocabulary(self, save_directory: str, filename_prefix: str):
        return ("",)

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    