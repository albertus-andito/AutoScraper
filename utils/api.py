from openai import OpenAI
import time

import ollama
import google.generativeai as genai
import os

from dotenv import load_dotenv

from openai import OpenAI
# from pydantic import BaseModel, Field
# import instructor

load_dotenv()

OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

vllm_client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

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

def get_json_schema(query):
    if "the value extracted from the HTML that match the instruction, if there is no data, keep it blank" in query:
        return {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string"
                },
                "value": {
                    "type": "string"
                },
                "xpath": {
                    "type": "string"
                }
            },
            "required": [
                "thought",
                "xpath"
            ]
        }
    elif "The element value" in query:
        return {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string"
                },
                "xpath": {
                    "type": "string"
                }
            },
            "required": [
                "thought",
                "xpath"
            ]
        }
    elif "judge whether the following HTML code contains all the expected value" or "judge whether the extracted value is consistent with the expected value" in query:
        return {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string"
                },
                "judgement": {
                    "type": "string"
                }
            },
            "required": [
                "thought",
                "judgement",
            ]
        }
    elif "the best action sequence choosen from the candidates" in query:
        return {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string"
                },
                "number": {
                    "type": "number"
                }
            },
            "required": [
                "thought",
                "number",
            ]
        }
    else:
        raise ValueError("No schema found for the query")


def qwen_coder(query):
    query_session = [{"role": "user", "content": query}]
    json_schema = get_json_schema(query)

    resp = vllm_client.chat.completions.create(
        model='qwen2.5-coder:32b-instruct-fp16',
        messages=query_session,
        temperature=0.0,
        extra_body={
            "guided_json": json_schema
        }
    )
    return resp.choices[0].message.content

    # resp = ollama.chat(
    #     model='qwen2.5-coder:32b-instruct-fp16',
    #     messages=query_session,
    #     # format='json'
    # )
    # return resp["message"]["content"]


# class Answer(BaseModel):
#     answer: str

def mixtral(query):
    query_session = [{"role": "user", "content": query}]
    options = ollama.Options(
        # f16_kv=False,
        temperature=0.0,
    )
    resp = ollama.chat(
        model='mixtral:8x7b-instruct-v0.1-fp16',
        messages=query_session,
        # format='json'
        options=options
    )
    # client = instructor.from_openai(
    #     OpenAI(
    #         base_url="http://localhost:11434/v1",
    #         api_key="ollama",  # required, but unused
    #     ),
    #     mode=instructor.Mode.JSON,
    # )
    # resp = client.chat.completions.create(
    #     model="llama3.2:latest",
    #     messages=query_session,
    #     response_model=Answer,
    # )
    # return resp
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