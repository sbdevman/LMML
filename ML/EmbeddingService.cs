using System.Text;
using Newtonsoft.Json;

namespace ML;

public class EmbeddingService
{
    private static readonly string apiKey = "YOUR_OPENAI_API_KEY";
    private static readonly HttpClient client = new HttpClient();

    public async Task<List<float>> GetEmbeddingAsync(string text)
    {
        var payload = new
        {
            input = text,
            model = "text-embedding-ada-002"
        };

        var requestContent = new StringContent(JsonConvert.SerializeObject(payload), Encoding.UTF8, "application/json");
        client.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");

        var response = await client.PostAsync("https://api.openai.com/v1/embeddings", requestContent);
        var responseBody = await response.Content.ReadAsStringAsync();
        dynamic jsonResponse = JsonConvert.DeserializeObject(responseBody);
        
        return jsonResponse.data[0].embedding.ToObject<List<float>>();
    }
}