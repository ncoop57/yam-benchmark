from evaluator import *
import requests

DESCRIPTION = 'Test if the model can generate a summary of the given resume and checks its quality against another LLM'

TAGS = ['resume', 'summarize']

url = 'https://gist.githubusercontent.com/henry7720/429604fe4eb16bea0256a4f8f6330746/raw/b01607aedc62970a0f126a5dbf767bba89ed3469/the-full-bee-movie-script.txt'
movie_script = requests.get(url).text

question = f'{movie_script}\n\nWrite a summarize of the above movie script'

evaluation = f"""
{movie_script}
Is this an accurate summary of the above movie script? Output either yes or no: <A>
"""

TestResumeSum = (
    question \
        >> LLMRun() \
        >> LLMRun(evaluation, llm=EVAL_LLM) \
        >> (SubstringEvaluator("yes") | \
           SubstringEvaluator("Yes"))
)

if __name__ == "__main__":
    print(run_test(TestResumeSum))