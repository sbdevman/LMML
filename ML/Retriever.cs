namespace ML;


public class Retriever
{
    private readonly MilvusHelper _milvus;
    private readonly EmbeddingService _embedder;
    private readonly DocumentDatabase _docDb;

    public Retriever(MilvusHelper milvus, EmbeddingService embedder, DocumentDatabase docDb)
    {
        _milvus = milvus;
        _embedder = embedder;
        _docDb = docDb;
    }

    public async Task<string> RetrieveRelevantContextAsync(string query)
    {
        var queryEmbedding = await _embedder.GetEmbeddingAsync(query);
        var retrievedIds = await _milvus.SearchAsync(queryEmbedding);

        if (retrievedIds.Count > 0)
        {
            return _docDb.GetDocumentById(retrievedIds[0]);
        }

        return "No relevant information found.";
    }
}