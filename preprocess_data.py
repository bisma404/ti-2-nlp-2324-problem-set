import pickle
import numpy as np
from aksara import MultiwordTokenizer, BaseTokenizer

class Preprocess:
  def __init__(self) -> None:
    self.data: list[str] = None
    self.train_data: list[str] = None
    self.test_data: list[str] = None
  
  """
  - Parameter path merupakan lokasi data diletakkan.
  - Data disarankan untuk disimpan di dalam folder data yang telah disediakan
  - Di dalam folder tersebut tersedia tiga jenis data, yakni idwiki-corpus.txt, idwiki-train.txt, dan idwiki-test.txt
  - Rasio antara train:test yang terdapat pada file idwiki-train.txt dan idwiki-test.txt merupakan 80:20
  - idwiki-corpus.txt merupakan gabungan antara idwiki-train.txt dan idwiki-test.txt
  - idwiki-corpus.txt disediakan untuk Anda yang ingin bereksperimen dengan rasio train:test atau sejenisnya.
  """
  def load_data(self, path: str) -> None:
    with open(path, 'r', encoding='utf-8') as f:
      data = f.readlines()
    
    data_type = path.split('/')[-1].split('-')[-1].split('.')[0]
    if data_type == 'train':
      self.train_data = data
    elif data_type == 'test':
      self.test_data = data

  def get_train_data(self) -> list[str]:
    return self.train_data
  
  def get_test_data(self) -> list[str]:
    return self.test_data
  
  """
  - Fungsionalitas method di bawah ini adalah melakukan pembersihan kalimat dari simbol newline ('\n').
  - Di samping itu, method diharapkan dapat melakukan trimming whitespace pada awal dan akhir kalimat.
  - Output format yang diharapkan berupa list of strings.
  """
  def split_sentences(self, data: list[str]) -> list[str]:
    # TODO: Implement based on the given description
    self.splitted_sentences = [sentence.strip() for sentence in data]
    return self.splitted_sentences
  
  """
  - Fungsionalitas method di bawah ini adalah melakukan tokenisasi pada kalimat.
  - Sebelum mengerjakan method ini, pastikan Anda telah melakukan eksplorasi pada dataset.
  - Perlu diperhatikan terdapat dua macam parameter yang disediakan, yakni data dan lower. Parameter lower digunakan untuk melakukan eksperimen lowercasing dan non-lowercasing.
  - Character/kata pada dataset tidak seluruhnya ascii/latin. Lakukan penanganan/filtering terhadap character/kata non-ascii.
  - Perlu diperhatikan bahwa n-gram merupakan case-sensitive model. Kata "saya" dan "Saya" merupakan dua hal yang berbeda. Silakan Anda tentukan penanganan yang Anda rasa terbaik.
  - Output format yang diharapkan adalah list of lists of strings.
  """
  def tokenize_sentences(self, data: list[str], lower: bool) -> list[list[str]]:
    # TODO: Implement based on the given description
    import re

    multiword_tokenizer = MultiwordTokenizer()
    self.tokenized_data = []

    for sentence in data:
      if not lower:
        sentence = re.sub(r'[^\x00-\x7F]+', '', sentence)
      else:
        sentence = re.sub(r'[^\x00-\x7F]+', '', sentence.lower())

      tokenized_sentence = multiword_tokenizer.tokenize(sentence)
      self.tokenized_data.append(tokenized_sentence)

    return self.tokenized_data
  
  def get_tokenized_data(self, data: list[str], lower: bool) -> list[list[str]]:
    splitted: list[str] = self.split_sentences(data)
    tokenized: list[str] = self.tokenize_sentences(splitted, lower)
    return tokenized

  """
  - Fungsionalitas pada method di bawah ini adalah menghitung kemunculan kata di dalam corpus.
  - Output format yang diharapkan berupa dictionary dengan pasangan key berupa kata/token dan value berupa jumlah kemunculan kata/token tsb.
  """
  def word_map(self, data: list[list[str]]) -> dict:
    # TODO: Implement based on the given description
    self.word_count = {}

    for sentence in data:
      for word in sentence:
        self.word_count[word] = self.word_count.get(word, 0) + 1

    return self.word_count
  
  """
  - Fungsionalitas pada method di bawah ini adalah melakukan filtering terhadap kata yang kemunculannya di bawah threshold/batasan tertentu.
  - Misalkan Anda menetapkan setiap kata yang kemunculannya di bawah 5 tidak perlu diproses (threshold = 5).
  - Anda diharapkan memanfaatkan fungsionalitas method word_map yang telah diimplementasikan sebelumnya.
  - Expected output berupa kumpulan kata yang kemunculannya di atas atau sama dengan (greater than or equal to) threshold.
  - Output format berupa list of strings.
  """
  def filter_vocab_by_threshold(self, data: list[list[str]], num_threshold: int) -> list[str]:
    # TODO: Implement based on the given description
    word_count = self.word_map(data)
    self.filtered_vocab = []

    for vocab, count in word_count.items():
      if count >= num_threshold: self.filtered_vocab.append(vocab)

    return self.filtered_vocab
  
  """
  - Fungsionalitas pada method ini adalah mengganti kata-kata yang kemunculannya di bawah threshold menjadi simbol <unk>.
  - Simbol <unk> melambangkan kata tersebut merupakan OOV (Out of Vocabulary).
  - Anda diharapkan memahami alur method ini dengan memanfaatkan method preprocess_raw_data.
  - Output format berupa list of lists of strings.
  """
  def handle_oov_with_unk(self, data: list[list[str]], vocab: list[str], unknown_token='<unk>') -> list[list[str]]:
    # TODO: Implement based on the given description
    self.oov_handled = []
    
    for sentence in data:
      oov_sentence = []
      for word in sentence:
        word = unknown_token if word not in vocab else word
        oov_sentence.append(word)
      self.oov_handled.append(oov_sentence)

    return self.oov_handled
  
  def preprocess_raw_data(self, train, test, threshold):
    vocab = self.filter_vocab_by_threshold(train, threshold)
    train_handled = self.handle_oov_with_unk(train, vocab)
    test_handled = self.handle_oov_with_unk(test, vocab)
    return vocab, train_handled, test_handled
  
  def save_to_pickle(self, vocab: list[str], train: list[list[str]], test: list[list[str]], lower: bool) -> None:
    filename = {'vocab': vocab, 'train': train, 'test': test}
    for key, value in filename.items():
      filename = f'./data/idwiki-{key}-uncased.pkl'
      if lower:
        filename = f'./data/idwiki-{key}-cased.pkl'
      with open(filename, 'wb+') as f:
        pickle.dump(value, f)
  
  def load_from_pickle(self, lower: bool):
    suffix = 'uncased'
    if lower:
      suffix = 'cased'
    with open(f'./data/idwiki-vocab-{suffix}.pkl', 'rb') as f:
      vocab = pickle.load(f)
    with open(f'./data/idwiki-train-{suffix}.pkl', 'rb') as f:
      train = pickle.load(f)
    with open(f'./data/idwiki-test-{suffix}.pkl', 'rb') as f:
      test = pickle.load(f)
    return vocab, train, test
  
def main():
  preprocess = Preprocess()

  print('Loading data...')
  preprocess.load_data('./data/idwiki-train.txt')
  preprocess.load_data('./data/idwiki-test.txt')

  train = preprocess.get_train_data()
  test = preprocess.get_test_data()

  """
  EXAMPLE:
  - Pada contoh ini menggunakan skenario no lowercasing (cased)
  """
  lowercase: bool = False
  tokenized_train = preprocess.get_tokenized_data(train, lowercase)
  tokenized_test = preprocess.get_tokenized_data(test, lowercase)

  """
  EXAMPLE:
  - Pada contoh ini menggunakan skenario lowercasing
  """
  # tokenized_train_lower = preprocess.get_tokenized_data(train, lower=True)
  # tokenized_test_lower = preprocess.get_tokenized_data(test, lower=True)

  """
  EXAMPLE:
  - Pada contoh ini ditetapkan threshold sebesar 3.
  - Artinya setiap kata yang kemunculannya di bawah 3 akan dihilangkan dari koleksi.
  """
  print('Pre-processing data...')
  vocab, train_handled, test_handled = preprocess.preprocess_raw_data(tokenized_train, tokenized_test, 3)
  # vocab_lower, train_handled_lower, test_handled_lower = preprocess.preprocess_raw_data(tokenized_train_lower, tokenized_test_lower, 3)

  print('Saving data...')
  preprocess.save_to_pickle(vocab, train_handled, test_handled, lowercase)
  # preprocess.save_to_pickle(vocab_lower, train_handled_lower, test_handled_lower, lower=True)
  print('Finish!')

if __name__ == "__main__":
  main()
