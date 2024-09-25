#!/usr/bin/env python3

import json
import os
import sys

import openai
from openai.error import RateLimitError, ServiceUnavailableError, APIError

import gpt_common as gpt
import impacts_common as impacts

# Constant which will hold the path to the JSON file containing the OpenAI
# secret key, which enables the use of the ChatGPT API
KEY_FILE = './secrets/key.json'

# Timeout wait time
# FREE API KEY = 20s wait times
# PAID API KEY = 1s wait times
WAIT_TIME = 20
#WAIT_TIME = 1


# Constant which will hold the path to a text file contatining a narrative for
# flash flood impact category definitions, in a GPT-compatible prompt form
FFSI_DEFINITIONS = './docs/ffsi_v1-original.txt'

# Constant which will hold the path to a CSV text file containing Local Storm
# Reports
LSR_FILE = "./data/test_flashflood_LSRs.csv"

# Constant which will hold the batch size we want to use for splitting our LSR
# dataset, so we can process it one batch at a time
BATCH_SIZE = 10

# Constant which will hold the total number of batches that should be processed
# MAX_BATCHES = 0 means NO LIMIT!!
MAX_BATCHES = 0

# Constant which will hold the path to the desires results output location
RESULTS_OUTPUT = './results/'


# Main function, which will be the entry point that will be executed, when this
# program is run as a script from the command line
def main():
    '''Main function and point of entry for the execution of this script.
    '''
    # Import the secret OpenAI API key from the JSON file
    openai.api_key = gpt.read_api_key(KEY_FILE)

    # Read FFSI definition
    impact_defs = impacts.read_textual_definition(FFSI_DEFINITIONS)

    # Read whole LSRs from a standard CSV file
    #lsr_reports = impacts.read_standard_lsrs(LSR_FILE)
    lsr_reports = impacts.read_ibw_lsrs(LSR_FILE)

    # Define a Unique Identifyer for the LSR file that is being processed
    lsr_uuid = impacts.hash_filename(LSR_FILE)

    # Define batches for batch processing the LSRs, so that it is easier to
    # restart the process, in case of interruptions or failure.
    batches = impacts.define_batches(lsr_reports.shape[0], BATCH_SIZE)

    # Check for pre-existing batch result JSON files, and if found, update the
    # batches dictionary with "processed": True, for the matching batches
    batches = impacts.match_batch_results(file_uuid=lsr_uuid,
                                          batches=batches,
                                          results_path=RESULTS_OUTPUT)

    # Process the LSRs, using the ChatGPT API to classify them

    # Hold the number of total batches for future reference
    num_batches = len(batches)
    # if verbose:
    print(f"Processing LSR file {lsr_uuid}: {lsr_reports.shape[0]} reports / "
          f"{num_batches} batches")

    # Variables to keep track of the total number of processed reports, as well
    # as the number of processed and skipped batches
    total_processed = 0
    batches_skipped = 0
    batches_processed = 0

    # For each batch in batches
    for batch_id in batches:
        # if verbose:
        print(f"Batch ID: {batch_id}")
        # If the current batch has been processed
        if batches[batch_id]["processed"]:
            # Do nothing, and move to the next batch
            # Keep track of how many batches were skipped
            batches_skipped += 1
            # if verbose:
            print(f"WARNING: Skipping batch {batch_id + 1}/{num_batches}"
                  f"- already processed")
            processed_lsrs = None

        # If not, process the current batch
        else:
            # Get the current batch start and end indices for the LSR DataFrame
            start_idx, end_idx = batches[batch_id]["indices"]

            # if verbose:
            print(f"Indices: ({start_idx},{end_idx})")

            # Subset the LSR reports to only select the reports for this batch.
            # Notice that the end index is incremented by 1, since the top
            # index is always excluded by definition.
            current_lsrs = lsr_reports.iloc[start_idx : end_idx + 1]

            # Process the current batch of LSRs
            try:
                processed_lsrs = gpt.classify_lsr_remarks(current_lsrs['remark'],
                                                          impact_defs,
                                                          temperature=0,
                                                          wait_time=WAIT_TIME,
                                                          verbose=True)
            except (RateLimitError,
                    ServiceUnavailableError,
                    APIError,
                    OSError) as e:
                print(f"ERROR! - The following eception occurred:\n\t {e}")
                print("Waiting 10s and retrying once before breaking!")
                gpt.wait_timeout(10)
                processed_lsrs = gpt.classify_lsr_remarks(current_lsrs['remark'],
                                                          impact_defs,
                                                          temperature=0,
                                                          wait_time=WAIT_TIME,
                                                          verbose=True)

            # Write the current batch's result as a JSON file, identified by
            # the lsr_uuid string and the current batch's ID, in the desired
            # output results folder
            batch_filename = f"{lsr_uuid}_{batch_id}.json"
            batch_path = os.path.join(RESULTS_OUTPUT, batch_filename)
            impacts.write_results_json(processed_lsrs, batch_path)

            # if verbose:
            print(f"Wrote partial results file: {batch_path}")

            # Mark batch as processed:
            batches[batch_id]["processed"] = True

            # Keep track of how many batches were processed
            batches_processed += 1

            # if verbose:
            print(f"Processed {len(processed_lsrs)} for"
                  f"batch {batch_id + 1}/{num_batches}")

        # Break after MAX_BATCHES, if MAX_BATCHES > 0
        if MAX_BATCHES and batch_id == MAX_BATCHES:
            # if verbose:
            print(f"WARNING: MAX_BATCHES of {MAX_BATCHES} reached! HALTING!\n")
            break

    # Keep track of how many total LSRs were processed
    if processed_lsrs:
        total_processed += len(processed_lsrs)
    else:
        total_processed += 0

    # Notify the user processing is done, and provide some counts on results
    print(f"DONE!\n\t"
          f"LSR file: {LSR_FILE}\n\t"
          f"UUID: {lsr_uuid}\n\t"
          f"processed {batches_processed}/{num_batches} batches\n\t"
          f"skipped {batches_skipped}/{num_batches} batches \n\t"
          f"processed {total_processed} LSRs")

    # Consolidate processed JSON results into a single CSV file

    # Check if there are any JSON files in the current output path
    result_files = impacts.get_files_in_dir(RESULTS_OUTPUT, extension=".json")
    # Make sure that the JSON files found match the current UUID
    result_files = list(filter(lambda x: lsr_uuid in x, result_files))

    # If the list of result files in the output folder, with the requested uuid
    # is empty, notify that no batch results were found
    if not result_files:
        print("ERROR: No batch result files were found for current uuid!")

    # Else if there are result files in the output folder
    else:
        # Create a copy of the original LSR DataFrame
        output_lsrs = lsr_reports.copy()
        #output_lsrs.index = output_lsrs.remark
        # Create the new columns for the new results data in the DataFrame
        output_lsrs["MINOR"] = None
        output_lsrs["MODERATE"] = None
        output_lsrs["SERIOUS"] = None
        output_lsrs["SEVERE"] = None
        output_lsrs["CATASTROPHIC"] = None
        output_lsrs["FFSI"] = None
        output_lsrs["EXTRA"] = None
        # For each of these files
        for json_file in result_files:
            # Read the results JSON file
            batch_results = impacts.read_json_results(json_file)
            # Get the current batch ID from the batch results file
            batch_id = json_file.split("/")[-1].split('_')[-1].split('.')[0]
            # Get the LSR dataframe ID corresponding to the beginning of the
            # current batch
            start_index = int(batch_id) * BATCH_SIZE
            # Iterate over each result, and add the data to the corresponding
            # columns in the output LSR dataframe, for the corresponding LSR
            for batch_index in batch_results:
                idx = start_index + int(batch_index)
                p_min = batch_results[batch_index][1]["MINOR"]
                p_mod = batch_results[batch_index][1]["MODERATE"]
                p_ser = batch_results[batch_index][1]["SERIOUS"]
                p_sev = batch_results[batch_index][1]["SEVERE"]
                p_cat = batch_results[batch_index][1]["CATASTROPHIC"]
                score = batch_results[batch_index][2]
                extra = batch_results[batch_index][3]
                output_lsrs.loc[idx]["MINOR"] = p_min / 100
                output_lsrs.loc[idx]["MODERATE"] = p_mod / 100
                output_lsrs.loc[idx]["SERIOUS"] = p_ser / 100
                output_lsrs.loc[idx]["SEVERE"] = p_sev / 100
                output_lsrs.loc[idx]["CATASTROPHIC"] = p_cat / 100
                output_lsrs.loc[idx]["FFSI"] = score
                output_lsrs.loc[idx]["EXTRA"] = extra

        output_lsrs.reset_index(drop=True)
        output_lsrs.to_csv(f"./results/{lsr_uuid}_classified.csv", index=False)

        print("DONE!")


# Block of code which will be executed when this file is executed as a script
if __name__ == '__main__':
    # Run the main() function
    main()
    # Terminate with exit code 0!
    sys.exit(0)
