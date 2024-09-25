#!/usr/bin/env python3

import json
from sys import stdout
from time import sleep
from numpy import isnan

from openai import ChatCompletion

from impacts_common import ffsi_score


def read_api_key(json_file_path):
    ''' Read the OpenAI key stored as a dictionary in a txt file.

    Function that reads a JSON file containing an OpenAI API key, which is
    stored in a dictionary of the form:
        { "secret_key": "<OpenAI API Secret Key String>" }
    '''
    # Read the JSON file
    with open(json_file_path) as key_file:
        openai_key = json.load(key_file)

    # Access and return the value of the API key
    return openai_key["secret_key"]


def query_gpt(query, role="user", system_task={}, temperature=1, top_p=1):
    ''' Query the GPY API and return the first completion result.

    Function that queries the GPT API, and returns the first completion
    produced as a response to the query.

    OpenAI's default values for temperature and top_p are maintained here as
    default values. From the official documentation:

    - temperature:
        What sampling temperature to use, between 0 and 2. Higher values like
        0.8 will make the output more random, while lower values like 0.2 will
        make it more focused and deterministic.

        We generally recommend altering this or top_p but not both.

    - top_p:
        An alternative to sampling with temperature, called nucleus sampling,
        where the model considers the results of the tokens with top_p
        probability mass. So 0.1 means only the tokens comprising the top 10%
        probability mass are considered.

        We generally recommend altering this or temperature but not both.
    '''
    # Process valid non-empty, non-blank string queries
    if query and query != " ":
        # Message list for the API query
        message_list = []

        # If a system task is passed as parameter, make sure to first append it to
        # the message_list
        if system_task:
            message_list.append(system_task)


        # Append the query to the message_list
        message_list.append({"role": role, "content": query})

        # Generate and receive back a ChatGPT completion for the GPT API
        completion = ChatCompletion.create(
            # Use the GPT-3.5-turbo model
            model="gpt-3.5-turbo",
            messages=message_list,
            temperature=temperature,
            top_p=top_p
        )

        # Return the first completion produced by our query to the API
        result = completion.choices[0].message.content

    # Handle EMPTY LSR remarks, by returning a classification dictionary of
    # zero probabilities, and an EXTRA string reporting the missing remark
    # WARNING: THIS IS SPECIFIC TO FFSI CLASSES; YOU MAY NEED TO CHANGE THIS!
    else:
        result = """{"MINOR": 0,
                 "MODERATE": 0,
                 "SERIOUS": 0,
                 "SEVERE": 0,
                 "CATASTROPHIC": 0
                 }NO REMARK FOR THIS LSR!"""

    # Return the result
    return result


def wait_timeout(seconds=20):
    ''' Wait (sleep) for a determined number of seconds and show countdown.

    This function sleeps for a determined number of seconds, and logs to the
    console a countdown progress of how much time remains in the waiting period

    GPT API Request rate limits are defined in terms of both Requests Per
    Minute (RPM) and Tokens Per Minute (TPM):

    Free trial users: 3 RPM (20s timeout) / 150,000 TPM
    Pay-as-you-go users: 60 RPM (1s timeout) / 250,000 TPM
    '''
    print("Waiting before querying the API again...")
    for remaining in range(seconds, 0, -1):
        stdout.write("\r")
        stdout.write("{:2d} seconds remaining.".format(remaining))
        stdout.flush()
        sleep(1)
    print("\n")


def classify_lsr_remarks(lsr_remarks_list, impact_defs,
                         temperature=1, top_p=1, wait_time=20,
                         starting_idx=0, limit=0, verbose=False):
    ''' Classify a list of LSR remarks using an impact definition and ChatGPT.

    This function queries ChatGPT for a classification based on a specific
    impact definition (FFSI), for each LSR remark contained in the input list
    received.

    This function returns a dictionary containing the original remarks, their
    associated probabilistic classifications produced by ChatGPT using the
    impact definitions, and its associated FFSI score.
    '''

    # Empty dictionary which will hold the results of the processed LSRs
    processed_lsrs = {}

    # Initialize the ChatGPT system task based on the impact definitions
    initialization_task = {"role": "system", "content": impact_defs}

    # Print out limits, the system task and impact definitions if verbose
    if verbose:
        # If limit is provided, show a warning and wait a bit
        if limit:
            print("WARNING: only %i LSRs will be processed! \n" % limit)
            sleep(2.5)

        # Print out the system task
        print("SYSTEM TASK:\n", impact_defs, '\n')

    # Keep track of total LSRs that need to be processed
    total_lsrs = len(lsr_remarks_list)

    # If the starting_index is not 0, slice the lst list to start at the
    # requested initial index
    if starting_idx:
        # If verbose, print a warning notifying of new starting point
        if verbose:
            print("WARNING: starting at index %i, out of %i" % (starting_idx,
                                                                total_lsrs))
            sleep(2.5)

        # Slice the LSR remarks to start at the new index
        lsr_remarks_list = lsr_remarks_list[starting_idx:]
        # Update the number of total LSRs to be processed
        total_lsrs = len(lsr_remarks_list)


    # Query the ChatGPT API prepending the system task each time
    for index, remark in enumerate(lsr_remarks_list):

        if verbose:
            print(f"Remark sent: {remark}")

        # Construct and send the query, receiving the result
        result = query_gpt(remark,
                           role="user",
                           system_task=initialization_task,
                           temperature=temperature,
                           top_p=top_p)

        if verbose:
            print(f"Result received: {result}")

        # In case the GPT API response includes more text than the requested
        # JSON dictionary formatted answer, separate the dictionary portion
        # from the rest of the response, and add the remaining response as a
        # "extra" in the result's dictionary

        # Extract any extra output GPT may have generated
        response_extra = result.split('}')[-1].split("\n\n")[-1]

        # Extract the classification results
        response_classes = result.split('{')[-1].split('}')[0]

        # Handle non-classification outputs, which have only EXTRA GPT outputs
        if response_classes == response_extra:
            response_classes = """"MINOR": 0,
                               "MODERATE": 0,
                               "SERIOUS": 0,
                               "SEVERE": 0,
                               "CATASTROPHIC": 0"""
        response_json = '{' + response_classes + '}'

        if verbose:
            print(f"Response JSON: \n\t{response_json}")
            print(f"Response EXTRA: \n\t{response_extra}\n")

        # Read the resulting probabilities in JSON format as a dictionary
        probs_dict = json.loads(response_json)

        # Calculate the FFSI score for the current classification results
        score = ffsi_score(probs_dict)

        # If verbose, print out the original remark, the classification,
        # probabilities, and its corresponding score
        if verbose:
            print("INDEX: %s\n" % str(index),
                  "PROMPT: %s\n" % remark,
                  "PROBS: %s\n" % result,
                  "SCORE: %s\n" % str(score),
                  "EXTRA: %s\n" % response_extra)

        # Add the current results to the output LSRs dictionary
        processed_lsrs[index] = [remark, probs_dict, score, response_extra]

        # If a limit is provided, stop classifyig LSRs after x reports
        if limit:
            # if the current index is equal to the limit, stop the loop
            if index == limit - 1:
                break

        # If there are still remarks to process, wait some time to comply with
        # the API's request rate limits
        if index < total_lsrs - 1:
            # Wait wait_time seconds (X requests/min rate limit)
            wait_timeout(wait_time)

    # Return the processed LSRs
    return processed_lsrs
