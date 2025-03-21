namespace ML;

class Program
{
    static async Task Main(string[] args)
    {
        var milvus = new MilvusHelper();
        var embedder = new EmbeddingService();
        var docDb = new DocumentDatabase(milvus, embedder);
        var retriever = new Retriever(milvus, embedder, docDb);
        var llm = new LLMService();

        // Index Documents (Run this once)
        await docDb.IndexDocumentsAsync();

        // Get user question
        Console.WriteLine("Enter your question:");
        string question = Console.ReadLine();

        // Retrieve relevant context
        string context = await retriever.RetrieveRelevantContextAsync(question);
        Console.WriteLine($"🔍 Retrieved Context: {context}");

        // Generate response
        string answer = await llm.GenerateResponseAsync(question, context);
        Console.WriteLine($"🤖 Answer: {answer}");
        
        // var db = new DocumentDatabase();
        // var embedder = new EmbeddingService();
        // var search = new VectorSearch();
        // var llm = new LLMService();
        //
        // // Index documents
        // List<(string Id, string Content, List<float> Embedding)> indexedDocs = new();
        // foreach (var doc in db.Documents)
        // {
        //     var embedding = await embedder.GetEmbeddingAsync(doc.Content);
        //     indexedDocs.Add((doc.Id, doc.Content, embedding));
        // }
        // search.IndexDocuments(indexedDocs);
        //
        // // Get user question
        // Console.WriteLine("Enter your question:");
        // string question = Console.ReadLine();
        //
        // // Get query embedding
        // var queryEmbedding = await embedder.GetEmbeddingAsync(question);
        //
        // // Retrieve relevant document
        // string relevantDoc = search.RetrieveMostRelevantDocument(queryEmbedding);
        // Console.WriteLine($"🔍 Retrieved Context: {relevantDoc}");
        //
        // // Generate response
        // string answer = await llm.GenerateResponseAsync(question, relevantDoc);
        // Console.WriteLine($"🤖 Answer: {answer}");
        
        // string[] trainingData = {
        //     "hello world",
        //     "good morning",
        //     "machine learning",
        //     "deep learning"
        // };
        //
        // Console.WriteLine(trainingData.Select(x => x.Length).Sum());
        //
        // int[] labels = { 1, 1, 0, 0 }; // Dummy labels (1 = positive, 0 = negative)
        //
        // Tokenizer tokenizer = new Tokenizer(trainingData);
        // Transformer transformer = new Transformer(64, 1); // 64 vocab size, 1 output
        //
        // Training.Train(transformer, tokenizer, trainingData, labels, epochs: 10, learningRate: 0.01);
        //
        // Console.WriteLine("Training complete!");
        //
        // Inference.Predict(transformer, tokenizer, "good");
        
    }
}