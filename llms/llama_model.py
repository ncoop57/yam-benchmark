from llama_cpp import Llama
import json


class LLAMAModel:
    def __init__(self, path, from_pretrained):
        config = json.load(open("config.json"))
        self.hparams = config['hparams']
        self.hparams.update(config['llms']['llama_cpp'].get('hparams') or {})
        n_ctx = self.hparams.get("n_ctx", 1524)
        if from_pretrained:
            self.llm = Llama.from_pretrained(repo_id=path, filename="*Q8_0.gguf", n_ctx=n_ctx)
        else:
            self.llm = Llama(model_path=path, chat_format="llama-2", n_ctx=n_ctx)

    def make_request(self, conversation, add_image=None, logit_bias=None, max_tokens=None, skip_cache=False):
        conversation = [{"role": "user" if i%2 == 0 else "assistant", "content": content} for i,content in enumerate(conversation)]
        print("Start chat")
        out = self.llm.create_chat_completion(
          messages = conversation
        )
        print("End chat")
        return out['choices'][0]['message']['content']
