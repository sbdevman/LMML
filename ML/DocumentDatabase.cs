namespace ML;

public class DocumentDatabase
{
    private List<(long Id, string Content)> _documents = new List<(long, string)>
    {
        (1, "Boyner is expanding into AI-driven retail innovation."),
        (2, "E-commerce in Turkey is rapidly growing."),
        (3, "Boyner's AI chatbot helps customers find fashion products."),
        (4, "AI is revolutionizing the retail sector with personalized recommendations.")
    };

    private MilvusHelper _milvus;
    private EmbeddingService _embedder;

    public DocumentDatabase(MilvusHelper milvus, EmbeddingService embedder)
    {
        _milvus = milvus;
        _embedder = embedder;
    }

    public async Task IndexDocumentsAsync()
    {
        foreach (var doc in _documents)
        {
            var embedding = await _embedder.GetEmbeddingAsync(doc.Content);
            await _milvus.InsertDocumentAsync(doc.Id, embedding);
        }
    }

    public string GetDocumentById(long id)
    {
        return _documents.Find(doc => doc.Id == id).Content;
    }
}