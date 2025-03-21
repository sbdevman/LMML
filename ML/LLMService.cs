using System.Text;
using Newtonsoft.Json;

namespace ML;

public class LLMService
{
    private static readonly string apiKey = "YOUR_OPENAI_API_KEY";
    private static readonly HttpClient client = new HttpClient();

    public async Task<string> GenerateResponseAsync(string question, string context)
    {
        var prompt = $"Context: {context}\n\nQuestion: {question}\n\nAnswer:";
        var payload = new
        {
            model = "gpt-4",
            prompt = prompt,
            max_tokens = 150
        };

        var requestContent = new StringContent(JsonConvert.SerializeObject(payload), Encoding.UTF8, "application/json");
        client.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");

        var response = await client.PostAsync("https://api.openai.com/v1/completions", requestContent);
        var responseBody = await response.Content.ReadAsStringAsync();
        dynamic jsonResponse = JsonConvert.DeserializeObject(responseBody);

        return jsonResponse.choices[0].text;
    }
}