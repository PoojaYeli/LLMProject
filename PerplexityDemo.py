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
# query = "unpopular south indian boy names"
# query = "pooja veeresh yeli"
# query = "shreyas hampali"
# query = "Is Zurich the largest train station in Switzerland?"
# query = "which is the largest train station in Switzerland?"
# query = "best vegas hotels for families"
# query = "openai api python"
# print(google_api_key)

def google_search(query, num=max_results):
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": google_api_key,
        "Content-Type": "application/json"
    }
    payload = {"q": query, "num": num}
    r = requests.post(url, json=payload, headers=headers)
    r.raise_for_status()
    return r.json()["organic"] #Skip  the ad sites

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
    prompt = f"You are an assistant. You need to answer the given question. Use the provided data as additional information to answer the question. " \
             f"If the provided data does not answer the question, then please use your intellect to answer the question. Provide the answer in the bullet format" \
             f"and do not repeat the answer. Also, provide citations to each bullet point. First provide a short summary, followed by details in bullet points." \
             f" The answer should be accurate and positive.:\n\n{data}\n\nQuestion: {question}\nAnswer:"
    # prompt = f"Use the following data to answer the question:\n\n{data}\n\nQuestion: {question}\nAnswer:"

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    print(response.choices[0].message.content.strip())
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