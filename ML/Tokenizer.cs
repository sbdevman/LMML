namespace ML;

public class Tokenizer
{
    private Dictionary<string, int> vocab = new Dictionary<string, int>();
    private int vocabSize = 0;

    public Tokenizer(string[] corpus)
    {
        BuildVocabulary(corpus);
    }

    private void BuildVocabulary(string[] corpus)
    {
        foreach (var sentence in corpus)
        {
            foreach (var word in sentence.Split(' '))
            {
                if (!vocab.ContainsKey(word))
                {
                    vocab[word] = vocabSize++;
                }
            }
        }
    }
    
    public int[] Encode(string sentence)
    {
        List<int> tokens = new List<int>();
        foreach (var word in sentence.Split(' '))
        {
            if (vocab.ContainsKey(word))
            {
                tokens.Add(vocab[word]);
            }
        }
        return tokens.ToArray();
    }

    public string Decode(int[] tokens)
    {
        Dictionary<int, string> reverseVocab = new Dictionary<int, string>();
        foreach (var kv in vocab)
        {
            reverseVocab[kv.Value] = kv.Key;
        }
        List<string> words = new List<string>();
        foreach (var token in tokens)
        {
            words.Add(reverseVocab[token]);
        }
        return string.Join(" ", words);
    }
}