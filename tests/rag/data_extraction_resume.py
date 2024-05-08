from evaluator import *
from pypdf import PdfReader
import requests
import io

DESCRIPTION = 'Test if the model can properly extract important information from the resume'

TAGS = ['resume', 'data_extraction']

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

def check_name():
    question = f'{resume}\n\nWho is the person in the resume?'

    return (
        question \
            >> LLMRun() \
            >> (
                SubstringEvaluator("Nathan Cooper") | \
                SubstringEvaluator("nathan cooper")
            )
    )

def check_phd_date():
    question = f'{resume}\n\nWhen did this person start their PhD?'

    return (
        question \
            >> LLMRun() \
            >> SubstringEvaluator("2018")
    )

def check_phd_end_date():
    question = f'{resume}\n\nWhen did this person finish their PhD?'

    return (
        question \
            >> LLMRun() \
            >> SubstringEvaluator("2023")
    )

def check_journal_pubs():
    question = f'{resume}\n\nHow many journal publications does this person have? Just output the integer number'

    return (
        question \
            >> LLMRun() \
            >> ContainsIntEvaluator(3)
    )

if __name__ == "__main__":
    print(run_test(check_name()))
    print(run_test(check_phd_date()))
    print(run_test(check_phd_end_date()))
    print(run_test(check_journal_pubs()))

