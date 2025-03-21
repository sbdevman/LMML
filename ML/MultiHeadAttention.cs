namespace ML;

public class MultiHeadAttention
{
    private int numHeads, dModel;
    private SelfAttention[] heads;

    public MultiHeadAttention(int numHeads, int dModel)
    {
        this.numHeads = numHeads;
        this.dModel = dModel;
        heads = new SelfAttention[numHeads];

        for (int i = 0; i < numHeads; i++)
            heads[i] = new SelfAttention(dModel / numHeads);
    }

    public double[,] ComputeMultiHead(double[,] input)
    {
        double[][,] outputs = new double[numHeads][,];
        
        for (int i = 0; i < numHeads; i++)
            outputs[i] = heads[i].ComputeAttention(input);

        return Concatenate(outputs);
    }
    private static double[,] Concatenate(double[][,] matrices)
    {
        int rows = matrices[0].GetLength(0), cols = matrices.Sum(m => m.GetLength(1));
        double[,] concatenated = new double[rows, cols];

        int colOffset = 0;
        foreach (var matrix in matrices)
        {
            for (int i = 0; i < rows; i++)
            for (int j = 0; j < matrix.GetLength(1); j++)
                concatenated[i, colOffset + j] = matrix[i, j];

            colOffset += matrix.GetLength(1);
        }

        return concatenated;
    }
}