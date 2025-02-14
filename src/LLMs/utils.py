import os
from openai import OpenAI
from tenacity import retry, wait_fixed, stop_after_attempt
import tiktoken

# os.environ['http_proxy'] = 'http://127.0.0.1:15236'
# os.environ['https_proxy'] = 'http://127.0.0.1:15236'
client = OpenAI()
encoder = tiktoken.encoding_for_model("gpt-3.5-turbo-instruct")


# @retry(wait=wait_fixed(1), stop=stop_after_attempt(3))
def get_reply(prompt, return_logprobs=False, max_tokens=50, temperature=0):
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=max_tokens,
        logprobs=5,
        temperature=temperature
    )
    print(response)
    input()
    logprobs = response.choices[0].logprobs
    out = {
        'text_offset': logprobs.text_offset,
        'token_logprobs': logprobs.token_logprobs,
        'tokens': logprobs.tokens,
        'top_logprobs': logprobs.top_logprobs,
    }
    if return_logprobs:
        return response.choices[0].text, out
    return response.choices[0].text


def construct_length(text, length=3840):
    codes = encoder.encode(text)
    codes = codes[:length]
    return encoder.decode(codes)


def get_length(text):
    return len(encoder.encode(text))


print(get_reply('Introduce yourself'))
