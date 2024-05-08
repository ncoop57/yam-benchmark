from evaluator import *
from pypdf import PdfReader
import requests
import io

DESCRIPTION = 'Test if the model can generate a summary of the given quarterly report and checks its quality against another LLM'

TAGS = ['report', 'summarize']

url = 'https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q4-2023-Update.pdf'
# Download the PDF from the URL
file = requests.get(url)

# Convert the response content into a file-like object
file = io.BytesIO(file.content)

# Read the PDF file
pdf = PdfReader(file)
# Initialize an empty string to store the extracted text
report = ''
# Loop through each page in the PDF file
for page in pdf.pages:
    # Extract the text from the page and append it to the text string
    report += page.extract_text()

question = f'{report}\n\nWrite a summarize of the above quarterly report'

evaluation = f"""
{report}
Is this an accurate summary of the above quarterly report? Output either yes or no: <A>
"""

TestReportSum = (
    question \
        >> LLMRun() \
        >> LLMRun(evaluation, llm=EVAL_LLM) \
        >> (SubstringEvaluator("yes") | \
           SubstringEvaluator("Yes"))
)

if __name__ == "__main__":
    print(run_test(TestReportSum))