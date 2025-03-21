namespace ML;

public class SelfAttention
{
    private int dModel;
    private double[,] Wq, Wk, Wv;

    public SelfAttention(int dModel)
    {
        this.dModel = dModel;
        Wq = MathUtils.RandomMatrix(dModel, dModel);
        Wk = MathUtils.RandomMatrix(dModel, dModel);
        Wv = MathUtils.RandomMatrix(dModel, dModel);
    }
    
    public double[,] ComputeAttention(double[,] input)
    {
        var Q = MathUtils.Dot(input, Wq);
        var K = MathUtils.Dot(input, Wk);
        var V = MathUtils.Dot(input, Wv);

        var scores = MathUtils.Dot(Q, Transpose(K));
        var scaledScores = Scale(scores, Math.Sqrt(dModel));

        // for (int i = 0; i < scaledScores.GetLength(0); i++)
        //     scaledScores[i] = MathUtils.Softmax(scaledScores[i]);

        return MathUtils.Dot(scaledScores, V);
    }

    private static double[,] Transpose(double[,] matrix)
    {
        int rows = matrix.GetLength(0), cols = matrix.GetLength(1);
        double[,] transposed = new double[cols, rows];

        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            transposed[j, i] = matrix[i, j];

        return transposed;
    }
    
    private static double[,] Scale(double[,] matrix, double factor)
    {
        int rows = matrix.GetLength(0), cols = matrix.GetLength(1);
        double[,] scaled = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            scaled[i, j] = matrix[i, j] / factor;

        return scaled;
    }
}