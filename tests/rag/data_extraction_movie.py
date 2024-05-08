from evaluator import *
import requests

DESCRIPTION = 'Test if the model can generate a summary of the given resume and checks its quality against another LLM'

TAGS = ['movie', 'data_extraction']

url = 'https://gist.githubusercontent.com/henry7720/429604fe4eb16bea0256a4f8f6330746/raw/b01607aedc62970a0f126a5dbf767bba89ed3469/the-full-bee-movie-script.txt'
movie_script = requests.get(url).text[:8000]

question = f'{movie_script}\n\nWrite a summarize of the above movie script'

evaluation = f"""
{movie_script}
Is this an accurate summary of the above movie script? Output either yes or no: <A>
"""

def check_name():
    question = f'{movie_script}\n\nWhat is the name of the main character?'

    return (
        question \
            >> LLMRun() \
            >> (
                SubstringEvaluator("Barry B. Benson") | \
                SubstringEvaluator("Barry") | \
                SubstringEvaluator("barry b. benson") | \
                SubstringEvaluator("barry")
            )
    )

def check_creature():
    question = f'{movie_script}\n\nWhat kind of creature is the main character?'

    return (
        question \
            >> LLMRun() \
            >> (SubstringEvaluator("bee") | SubstringEvaluator("Bee"))
    )

def check_location():
    question = f'{movie_script}\n\nWhere does the movie take place?'

    return (
        question \
            >> LLMRun() \
            >> (SubstringEvaluator("New York City") | SubstringEvaluator("New York"))
    )

def check_love():
    question = f'{movie_script}\n\nWhat is the name of the main character\'s love interest?'

    return (
        question \
            >> LLMRun() \
            >> (SubstringEvaluator("Vanessa") | SubstringEvaluator("vanessa"))
    )

# TestResumeSum = (
#     question \
#         >> LLMRun() \
#         >> LLMRun(evaluation, llm=EVAL_LLM) \
#         >> (SubstringEvaluator("yes") | \
#            SubstringEvaluator("Yes"))
# )

if __name__ == "__main__":
    print(run_test(check_name()))
    print(run_test(check_creature()))
    print(run_test(check_location()))
    print(run_test(check_love()))