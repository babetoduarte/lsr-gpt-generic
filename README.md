# LSR GPT Classification Framework  
by: Jorge A. Duarte - jduarte@ou.edu 

## About

This is a project skeleton for using OpenAI's ChatGPT API to classify Local Storm Report remarks. **This framework was created to be used with Flash Flood Severity Index textual definitions and flash flood LSRs, and will require adaptation for using other impact classification frameworks, and hazard types**.

## Dependencies

Dependencies for this project are listed under `/requirements.txt`, which you can easily install using `pip`, or `conda`. Make sure to install these python packages before trying to run this project.

## Data

Make sure to set up your GPT prompt which contains *textualized* versions of your impact definitions, as well as clear instruction's for ChatGPT on how to perform your classification, and presenting its results. A sample is provided in the file `/docs/ffsi_v1-original.txt`.

Make sure to set up your LSRs in a CSV file with clear column names. You may need to update/modify/reimplement functions for reading your specific LSR format in `/impacts_common.py`. A sample LSR CSV file is provided in `/data/test_flashflood_LSRs.csv`.

As a default, all outputs will be write out to `/results/`.

## Code

This framework uses an API key which should be pasted within the `/secrets/key.json` file as follows:

``` json
{
    "secret_key": "<YOUR SECRET API KEY GOES HERE WITHIN THE QUOTATIONS!>"
}
```

The file `/gpt_common.py` encapsulates various GPT API functionalities, and implements a waiting function for submitting successive API requests. Functions include performing a single API query, and performing said query for a given list of LSRs. Note that the `classify_lsr_remarks()` function does not implement any mitigation against interruptions or network errors (DO NOT USE THIS FUNCTION FOR PROCESSING A LARGE QUANTITY OF LSRs!)!

The file `/impacts_common.py` encapsulates various high-level functions like reading textual impact definitions from a text file (i.e. the 'prompt' to be sent for each classification), reading LSR remarks from a CSV containing complete Local Storm Reports (multiple functions implemented, choose accordingly), calculating the FFSI score, writing and reading intermediate results as JSON files, calculating unique filename-based identifiers (hashes), defining batches for batch-processing large quantities of LSRs, and matching previously-existing intermediate batch results for resuming classification upon interruptions during batch processing.

Lastly, the file `/gpt_classify.py` is the main file that should be executed from the command line as follows:

``` sh
$ python gpt_classify.py
```

Please not that within the first 45 lines of this file, you will find *constants* defined with  for the framework's execution including: wait times between API requests, location for the prompt text file to be used, location for the CSV file containing LSRs, batch size, and output folder. Make sure to change these accordingly.
