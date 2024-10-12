from openai import OpenAI
import time

import ollama
import google.generativeai as genai
import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

GEMINI_API_KEY=os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')


def chatgpt(query):
    query_session = [{"role":"user", "content": query}]
    resp = client.chat.completions.create(
        model='gpt-4-1106-preview',
        messages=query_session,
        temperature=0.1,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    ret = resp.choices[0].message.content
    return ret


def gpt4(query):
    query_session = [{"role":"user", "content": query}]
    resp = client.chat.completions.create(
        model='gpt-4o',
        messages=query_session,
        temperature=0.1,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    ret = resp.choices[0].message.content
    return ret


def phi3_5(query):
    query_session = [{"role": "user", "content": query}]
    resp = ollama.chat(
        model='phi3.5:3.8b',
        messages=query_session,
        # format='json'
    )
    return resp["message"]["content"]


def deepseek(query):
    query_session = [{"role": "user", "content": query}]
    resp = ollama.chat(
        model='deepseek-coder:33b-instruct-q8_0',
        messages=query_session,
        # format='json'
    )
    return resp["message"]["content"]


def mixtral(query):
    query_session = [{"role": "user", "content": query}]
    resp = ollama.chat(
        model='mixtral:8x7b-instruct-v0.1-q6_K',
        messages=query_session,
        # format='json'
    )
    return resp["message"]["content"]


def gemini(query):
    safety_settings = {
        'HATE': 'BLOCK_NONE',
        'HARASSMENT': 'BLOCK_NONE',
        'SEXUAL': 'BLOCK_NONE',
        'DANGEROUS': 'BLOCK_NONE'
    }
    # chat = gemini_model.start_chat()
    resp = gemini_model.generate_content(query, safety_settings=safety_settings)
    time.sleep(4) # free limit is 15 requests per minute
    return resp.text


if __name__ == '__main__':
    print((mixtral('How do I calculate the 100th prime number in Python?')))