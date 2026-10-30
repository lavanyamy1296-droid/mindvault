import requests
from .config import OLLAMA_URL, OLLAMA_MODEL

def generate_answer(question: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join(
        f"[Source: {item['filename']}] {item['text']}" for item in contexts
    )
    prompt = f"""You are MindVault, a careful personal knowledge assistant.
Answer the user's question using only the supplied context.
If the context does not contain the answer, say that the uploaded documents do not provide enough information.
Do not invent facts. Keep the answer clear and structured.
At the end, mention the source filenames used.

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.RequestException:
        if not contexts:
            return "I couldn't find relevant information, and the local Ollama model is not available. Start Ollama and try again."
        excerpts = "\n\n".join(
            f"From {item['filename']}: {item['text']}" for item in contexts[:3]
        )
        return (
            "Ollama is not available, so here are the closest matching excerpts instead of a generated answer:\n\n"
            + excerpts
        )
