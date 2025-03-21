namespace ML;

public class Training
{
    public static void Train(Transformer transformer, Tokenizer tokenizer, string[] trainingData, int[] labels, int epochs, double learningRate)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalLoss = 0;
            for (int i = 0; i < trainingData.Length; i++)
            {
                int[] tokens = tokenizer.Encode(trainingData[i]);
                double[,] input = Transformer.OneHotEncode(tokens, 64);
                double[,] output = transformer.Forward(input);
                double loss = Transformer.ComputeLoss(output, new double[,] {{ labels[i] }});
                totalLoss += loss;
                transformer.AdjustWeights(output, learningRate);
            }
            Console.WriteLine($"Epoch {epoch + 1}, Loss: {totalLoss}");
        }
    }
}