namespace ML;

public class FeedForward
{
    private double[,] W1, W2;
    private int dModel;

    public FeedForward(int dModel)
    {
        this.dModel = dModel;
        W1 = MathUtils.RandomMatrix(dModel, 4 * dModel);
        W2 = MathUtils.RandomMatrix(4 * dModel, dModel);
    }

    public double[,] Forward(double[,] input)
    {
        var hidden = Relu(MathUtils.Dot(input, W1));
        return MathUtils.Dot(hidden, W2);
    }

    private static double[,] Relu(double[,] matrix)
    {
        int rows = matrix.GetLength(0), cols = matrix.GetLength(1);
        double[,] activated = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            activated[i, j] = Math.Max(0, matrix[i, j]);

        return activated;
    }
}