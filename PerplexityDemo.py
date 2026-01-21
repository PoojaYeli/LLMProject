from dotenv import load_dotenv
import os
import requests
import csv
from openai import OpenAI

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

max_results = 10
conversation_history = []

def google_search(query, num=max_results):
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": google_api_key,
        "Content-Type": "application/json"
    }
    payload = {"q": query, "num": num}
    r = requests.post(url, json=payload, headers=headers)
    r.raise_for_status()
    return r.json()["organic"]  # Skip  the ad sites


def save_to_csv(query, results, filename="serper_results.csv"):
    # Write header if file does not exist
    write_header = not os.path.exists(filename)

    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["query", "title", "link", "snippet"])
        for item in results:
            writer.writerow([query, item["title"], item["link"], item["snippet"]])


def ask_chatgpt(data, question):
    global conversation_history
    system_prompt = f"""You are a precise, professional research assistant.

    TASK:
    Answer the question using the provided data as supporting context. If the data is insufficient, use your own knowledge.

    CONTEXT RULES:
    - Treat this as a continuous discussion.
    - Build on previous answers.
    - Do NOT repeat previously stated facts.
    - Add only new, relevant information.
    - If a question depends on prior context, use it implicitly.
    - Maintain logical continuity across turns.

    FORMAT RULES:
    1. SUMMARY must be high-level and abstract.
    2. BULLET POINTS must contain ONLY new details.
    3. No overlap between summary, bullets, or prior answers.
    4. Each bullet must be unique.
    5. Provide a citation for each bullet.
    6. Self-check for repetition before responding.
    """

    user_prompt = f"""
    NEW DATA (if relevant):{data}\n\nCurrent question:{question}\n\nAnswer:
    """

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.2
    )

    answer = response.choices[0].message.content.strip()
    print(answer)

    # Save conversation context
    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": answer})

    next_question = input("Is there anything further I can assist you with? y/n:")

    if (next_question == 'y' or next_question == "Y"):
        program_flow()


def program_flow():
    user_question = input("Ask anything: ")
    results_list = google_search(user_question)

    ask_chatgpt(results_list, user_question)


program_flow()

# save_to_csv(results_list, user_question)
# print(f"Saved {len(results_list)} results for user_question: '{user_question}'")