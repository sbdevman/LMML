namespace ML;
public class TransformerEncoder
{
    private double[,] weights;
    private double[,] attentionWeights;

    public TransformerEncoder(int inputSize, int outputSize)
    {
        weights = new double[inputSize, outputSize];
        attentionWeights = new double[inputSize, inputSize];
        Random rand = new Random();

        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
                weights[i, j] = rand.NextDouble();
            for (int j = 0; j < inputSize; j++)
                attentionWeights[i, j] = rand.NextDouble();
        }
    }

    private double[,] ComputeSelfAttention(double[,] input)
    {
        int size = input.GetLength(0);
        double[,] attentionScores = new double[size, size];
        double[,] output = new double[size, input.GetLength(1)];

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                attentionScores[i, j] = 0;
                for (int k = 0; k < input.GetLength(1); k++)
                    attentionScores[i, j] += input[i, k] * input[j, k];
            }
        }

        for (int i = 0; i < size; i++)
        {
            double sum = attentionScores[i, 0];
            for (int j = 1; j < size; j++)
                sum += attentionScores[i, j];
            for (int j = 0; j < size; j++)
                attentionScores[i, j] /= sum;
        }

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < input.GetLength(1); j++)
            {
                output[i, j] = 0;
                for (int k = 0; k < size; k++)
                    output[i, j] += attentionScores[i, k] * input[k, j];
            }
        }
        return output;
    }

    public double[,] Forward(double[,] input)
    {
        double[,] attentionOutput = ComputeSelfAttention(input);
        int rows = input.GetLength(0);
        int cols = weights.GetLength(1);
        double[,] output = new double[rows, cols];
        
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                for (int k = 0; k < input.GetLength(1); k++)
                    output[i, j] += attentionOutput[i, k] * weights[k, j];
        
        return output;
    }

    public void AdjustWeights(double[,] gradients, double learningRate)
    {
        for (int i = 0; i < weights.GetLength(0); i++)
        {
            for (int j = 0; j < weights.GetLength(1); j++)
            {
                weights[i, j] -= learningRate * gradients[i, j];
            }
        }
    }
    
    public  double ComputeLoss(double[,] output, double[,] target)
    {
        double loss = 0;
        int rows = output.GetLength(0);
        int cols = output.GetLength(1);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double diff = output[i, j] - target[i, j];
                loss += diff * diff;
            }
        }
        return loss / (rows * cols);
    }
}