from google import genai

# Cache a single client instance and reuse it across calls.
_client: genai.Client | None = None


def get_client() -> genai.Client:
    """Return a connected Vertex AI client, reusing the existing one if available."""
    global _client
    if _client is None:
        print("Connecting to Gemini on Vertex AI...")
        _client = genai.Client(
            vertexai=True,
            project='pid-gousenad-enen-gpaig-01',  # Your organization project ID
            location='global'  # Baseline cloud region
        )
    else:
        print("Reusing existing Gemini client...")
    return _client


try:
    client = get_client()

    # Generate content using the enterprise-scale model engine
    response = client.models.generate_content(
        model='gemini-3.8-flash',
        contents='what is kafka',
    )

    print("\n🎉 Connection Successful! Response:")
    print(response.text)

except Exception as e:
    print("\n❌ Pipeline Execution Failed:")
    print(e)
