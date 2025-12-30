from openai import OpenAI

# Initialize client (make sure OPENAI_API_KEY is set in environment)
client = OpenAI(
    # base_url= "http://35.220.164.252:3888/v1/",
    base_url="http://34.13.73.248:3888/v1", # 谷歌负载均衡网络，全球节点，适合国外访问
    # base_url="https://api.boyuerichdata.opensphereai.com/v1", # 直连香港https，数据加密
    api_key="sk-KEY"
)

# Test input
test_text = "Summarize this text in one sentence."

# List of models from most powerful → least powerful
models_to_test = [
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5.2-thinking",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-3.5-turbo"
]

for model_name in models_to_test:
    try:
        print(f"\nTesting model: {model_name}")
        response = client.responses.create(
            model=model_name,
            input=test_text,
            temperature=0.0
        )
        print("Success! Model works.")
        print("Output:", response.output_text)
    except Exception as e:
        print(f"Failed: {e}")
