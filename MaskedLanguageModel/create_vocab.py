import sentencepiece as spm
data_path = "/home/hedan/tools/Github/NLP_Based_Transformer/data/wikitext-2/train.txt"
save_path = "/home/hedan/tools/Github/NLP_Based_Transformer/MaskedLanguageModel/VocabModel/vocab"
spm.SentencePieceTrainer.train(input=data_path, model_prefix=save_path,model_type="bpe",vocab_size=10000)
