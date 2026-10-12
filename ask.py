import chromadb
# from openai import OpenAI
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = 'chroma_db'

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="growing_vegetables")

user_query = input("What do you want to know about vegetables? \n\n")

results = collection.query(
    query_texts=user_query,
    n_results=1
)

# client = OpenAI()
client = genai.Client()

system_prompt = """
You are a helpful assistant. You answer questions about growing vegetables in Florida. 
But you only answer based on knowledge I'm providing you. You don't use your internal 
knowledge and you don't make thins up.
If you don't know the answer, just say: I don't know
--------------------

The Data:
"""+str(results['documents'])+"""

"""

# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role":"system","content":system_prompt},
#         {"role":"user", "content":user_query},
#     ]
# )

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=user_query,
    config=types.GenerateContentConfig(
        system_instruction=system_prompt,
    ),
)


print("\n\n------------------------------------\n\n")

# print(response.choices[0].message.content)
print(response.text)