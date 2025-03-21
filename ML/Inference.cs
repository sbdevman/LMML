namespace ML;

public class Inference
{
    public static void Predict(Transformer transformer, Tokenizer tokenizer, string sentence)
    {
        int[] inputTokens = tokenizer.Encode(sentence);
        double[,] inputEmbeddings = Transformer.OneHotEncode(inputTokens, 64);
        double[,] prediction = transformer.Forward(inputEmbeddings);
        Console.WriteLine(prediction[0, 0] > 0.5 ? "Positive" : "Negative");
    }
}
