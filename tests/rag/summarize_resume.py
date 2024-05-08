from evaluator import *
from pypdf import PdfReader
import requests
import io

DESCRIPTION = 'Test if the model can generate a summary of the given resume and checks its quality against another LLM'

TAGS = ['resume', 'summarize']

# Download the PDF from the URL
response = requests.get('https://nathancooper.io/files/resume.pdf')

# Convert the response content into a file-like object
file = io.BytesIO(response.content)

# Read the PDF file
pdf = PdfReader(file)
# Initialize an empty string to store the extracted text
resume = ''
# Loop through each page in the PDF file
for page in pdf.pages:
    # Extract the text from the page and append it to the text string
    resume += page.extract_text()

question = f'{resume}\n\nWrite a summarize of the above resume'

evaluation = f"""
{resume}
Is this an accurate summary of the above resume? Output either yes or no: <A>
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

