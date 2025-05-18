import requests

def ollama_code_review_generator(model: str = "mistral", prompt_template: str = "Explain the following Java code:\n{}"):
    """
    Returns a generator function that can be called repeatedly to send code snippets
    to Ollama and yield results.
    """
    def send_snippet(snippet: str):
        prompt = prompt_template.format(snippet)
        print(prompt)
        response = requests.post(
            "http://localhost:11435/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        if response.status_code == 200:
            yield response.json()['response']
        else:
            yield f"[Error {response.status_code}]: {response.text}"

    return send_snippet
