import random
from retrying import retry
from openai import OpenAI
import requests

class OpenAI_Agent:
    def __init__(self, 
                model_name="gpt-4o-mini", 
                temperature=0.5,
                base_url = "https://api.ai-gaochao.cn/v1"):
        
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = base_url
        self.history = []
        self.key='sk-'

    def __post_process(self, response):
        return response.choices[0].message.content

    @retry(wait_fixed=300, stop_max_attempt_number=50)
    def __call__(self, message, systemPrompt=''):
        if message is None or message == "":
            return False, "Your input is empty."

        client = OpenAI(api_key=self.key, base_url=self.base_url)

        self.history.append({"role": "user", "content": message})
        messages = [{"role": "system", "content": systemPrompt}] + self.history

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            n=1,
        )

        model_reply = self.__post_process(response)
        self.history.append({"role": "assistant", "content": model_reply})

        return model_reply
    


# agent = OpenAI_Agent()

# system_prompt = "You are a helpful assistant."
# while True:
#     user_input = input("You: ").strip()
    
#     if user_input.lower() in ['quit', 'exit', 'bye']:
#         print("对话结束")
#         break
        
#     response = agent(user_input, systemPrompt=system_prompt)
#     print("Assistant:", response)