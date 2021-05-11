
import transformers
class TTokenizer(transformers.PreTrainedTokenizer):
    def __init__(self,vocab_to_id,vocab_to_str,**kwargs):
        super().__init__(**kwargs)
        self.tokens_encode = vocab_to_id
        self.tokens_decode = vocab_to_str
        # self.added_tokens_encoder = {"<mask>":1001, "0":0}
        # self.added_tokens_decoder = {1001:"<mask>", 0:"0"}

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
