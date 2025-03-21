namespace ML;

public class VectorSearch
{
    private readonly List<(string Id, string Content, List<float> Embedding)> _indexedDocuments = new List<(string, string, List<float>)>();

    public void IndexDocuments(List<(string Id, string Content, List<float> Embedding)> documents)
    {
        _indexedDocuments.AddRange(documents);
    }

    public string RetrieveMostRelevantDocument(List<float> queryEmbedding)
    {
        return _indexedDocuments
            .OrderByDescending(doc => CosineSimilarity(queryEmbedding, doc.Embedding))
            .Select(doc => doc.Content)
            .FirstOrDefault();
    }

    private float CosineSimilarity(List<float> vecA, List<float> vecB)
    {
        float dotProduct = 0, normA = 0, normB = 0;
        for (int i = 0; i < vecA.Count; i++)
        {
            dotProduct += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }
        return dotProduct / (float)(Math.Sqrt(normA) * Math.Sqrt(normB));
    }
}