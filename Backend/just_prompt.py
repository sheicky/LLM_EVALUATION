from openai import OpenAI 
import google.generativeai as genai


class SimplePrompt : 

    def __init__(self) : 
        self.generation_config = {
              "temperature": 1,
              "top_p": 0.95,
              "top_k": 5,
              "max_output_tokens": 8192,
              "response_mime_type": "text/plain",
              }


    def gemini_model_response(self, model_name,api_key,prompt) : 
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=model_name,generation_config=self.generation_config)
        response = model.generate_content(prompt)
        return response.text
    
    def groq_model_response(self, model_name, api_key, prompt) : 
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

