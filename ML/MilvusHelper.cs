using System.ComponentModel.DataAnnotations;
using Milvus.Client;

namespace ML;


public class MilvusHelper
{
    private readonly MilvusClient _client;
    private readonly string _collectionName = "document_vectors";

    public MilvusHelper(string milvusHost = "127.0.0.1", int milvusPort = 19530)
    {
        _client = new MilvusClient(milvusHost, milvusPort);
        InitializeCollection().Wait();
    }

    private async Task InitializeCollection()
    {
        if (!await _client.HasCollectionAsync(_collectionName))
        {
            await _client.CreateCollectionAsync(_collectionName, 
                new List<FieldSchema>
                {
                     FieldSchema.Create("id", MilvusDataType.Int64, isPrimaryKey: true),
                     FieldSchema.Create("vector", MilvusDataType.FloatVector,  true)
                }
            ));
        }
    }

    public async Task InsertDocumentAsync(long id, List<float> embedding)
    {
        var entities = new List<Entity>
        {
            new Entity("id", id),
            new Entity("vector", embedding)
        };

        await _client.GetCollection(_collectionName).InsertAsync( entities.AsReadOnly());
    }

    public async Task<List<long>> SearchAsync(List<float> queryVector, int topK = 1)
    {
        var searchResults = await _client.GetCollection(
            _collectionName).SearchAsync(
            new List<string> { "vector" },
            new List<string> { "id" },
            queryVector,
            topK
        );

        return searchResults.Select(res => (long)res["id"]).ToList();
    }
}