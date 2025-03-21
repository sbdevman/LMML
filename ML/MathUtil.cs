namespace ML;

public static class MathUtils 
{
    public static double[,] RandomMatrix(int rows, int cols)
    {
        var rand = new Random();
        double[,] matrix = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i, j] = rand.NextDouble();
        return matrix;
    }
    
    public static double[,] Dot(double[,] a, double[,] b)
    {
        int aRows = a.GetLength(0), aCols = a.GetLength(1);
        int bRows = b.GetLength(0), bCols = b.GetLength(1);
        
        double[,] result = new double[aRows, bCols];
        
        for (int i = 0; i < aRows; i++)
        for (int j = 0; j < bCols; j++)
        for (int k = 0; k < aCols; k++)
            result[i, j] += a[i, k] * b[k, j];

        return result;
    }
    
    public static double[] Softmax(double[] vector)
    {
        double max = vector.Max();
        double sum = vector.Sum(v => Math.Exp(v - max));
        return vector.Select(v => Math.Exp(v - max) / sum).ToArray();
    }
}