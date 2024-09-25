#!/usr/bin/env python3

import csv
import json
import os

from numpy import round, nan
from pandas import read_csv, to_datetime
from shortuuid import uuid


def read_textual_definition(text_file_path):
    ''' Read a text file containing an FFSI definition.

    This function reads an textfile into a multiline string object.
    '''
    with open(text_file_path) as text_file:
        ffsi_definition = text_file.read()

    return ffsi_definition


def read_lsr_remarks(lsr_file_path, column_name='REMARK'):
    ''' Read a single column from CSV file containing Local Storm Reports.

    This function reads a CSV file containing Local Storm Reports, and returns
    a list of 'remarks' contained in the 'REMARK' column.
    '''
    # Open the CSV file
    lsr_file = open(lsr_file_path)

    # Create a DictReader object
    lsrs = csv.DictReader(lsr_file)

    # List to hold the LSR remarks
    remarks = []

    # Iterate over each report, and append its remark to the remarks list
    for report in lsrs:
        remarks.append(report[column_name])

    # Return the LSR remarks
    return remarks


def read_standard_lsrs(lsr_file_path, no_index=True, no_category=True):
    """ Read a standard CSV file containing a collection of LSRs

    This function reads a "standard" CSV format containing LSR reports, as they
    are provided by the Iowa State Univeristy Local Storm Report Archive:
    https://mesonet.agron.iastate.edu/request/gis/lsrs.phtml
    """
    # Read the file, and define standard names for each columns, as well as
    # approrpiate data types
    if no_category:
        standard_lsrs = read_csv(lsr_file_path, delimiter=',', header=0,
                                names=["valid", "valid2", "lat", "lon",
                                        "mag", "wfo", "typecode", "typetext",
                                        "city", "county", "state", "source",
                                        "remark", "ugc", "ugcname"],
                                index_col=False,#"valid2",
                                dtype={"valid": str, "valid2": str,
                                        "lat": str, "lon": str,
                                        "mag": str, "wfo": str, "typecode": str,
                                        "typetext": str, "city": str,
                                        "county": str, "state": str, "source": str,
                                        "remark": str, "ugc": str, "ugcname": str})
        standard_lsrs["category"]=None
    else:
        standard_lsrs = read_csv(lsr_file_path, delimiter=',', header=0,
                                names=["valid", "valid2", "lat", "lon",
                                        "mag", "wfo", "typecode", "typetext",
                                        "city", "county", "state", "source",
                                        "remark", "category", "ugc", "ugcname"],
                                index_col=False,#"valid2",
                                dtype={"valid": str, "valid2": str,
                                        "lat": str, "lon": str,
                                        "mag": str, "wfo": str, "typecode": str,
                                        "typetext": str, "city": str,
                                        "county": str, "state": str, "source": str,
                                        "remark": str, "category": str, "ugc": str,
                                        "ugcname": str})

    if not no_index:
        standard_lsrs.set_index('valid2', inplace=True)

        # Make sure that dates and times are represented appropriately
        standard_lsrs.index = to_datetime(standard_lsrs.index)

        # Make sure that report sources, categories, and WFOs are represented
        # appropriately as categorical data
        standard_lsrs.source = standard_lsrs.source.astype('category')
        standard_lsrs.category = standard_lsrs.category.astype('category')
        standard_lsrs.wfo = standard_lsrs.wfo.astype('category')

    # Remove all NaNs from the remarks column (empty remarks) and replace them
    # with blank strings " "
    standard_lsrs.remark.replace(nan," ",regex=True, inplace=True)

    # Return the loaded data in a Pandas DataFrame
    return standard_lsrs


def read_ibw_lsrs(lsr_file_path, no_index=True):
    """ Read a CSV file containing a collection of Expertly-Classified LSRs

    The 'magnitude' field corresponds to IBW categories for each LSR
    """
    # Read the file, and define standard names for each columns, as well as
    # approrpiate data types
    standard_lsrs = read_csv(lsr_file_path, delimiter=',', header=0,
                             names=["time", "office", "local_time",
                                    "county", "location", "state",
                                    "event_type", "magnitude", "source",
                                    "lat", "lon", "remark"],
                             index_col=False,  # "valid2",
                             dtype={"time": str,
                                    "office": str,
                                    "local_time": str,
                                    "county": str,
                                    "location": str,
                                    "state": str,
                                    "event_type": str,
                                    "magnitude": str,
                                    "source": str,
                                    "lat": str,
                                    "lon": str,
                                    "remark": str})
    standard_lsrs["category"] = None

    if not no_index:
        standard_lsrs.set_index('time', inplace=True)

        # Make sure that dates and times are represented appropriately
        standard_lsrs.index = to_datetime(standard_lsrs.index)
        standard_lsrs.local_time = to_datetime(standard_lsrs.local_time)

        # Make sure that report sources, categories, and WFOs are represented
        # appropriately as categorical data
        standard_lsrs.source = standard_lsrs.source.astype('category')
        standard_lsrs.magnitude = standard_lsrs.source.astype('magnitude')
        standard_lsrs.category = standard_lsrs.category.astype('category')
        standard_lsrs.office = standard_lsrs.wfo.astype('office')

    # Remove all NaNs from the remarks column (empty remarks) and replace them
    # with blank strings " "
    standard_lsrs.remark.replace(nan, " ", regex=True, inplace=True)

    # Return the loaded data in a Pandas DataFrame
    return standard_lsrs


def ffsi_score(probs, normalize=False):
    ''' Calculate a single score from FFSI class probabilities.

    This function converts a dictionary or list of FFSI class probabilities (in
    percents), into a single score between 1 and 5, representing a continuum of
    values across the total number of classes. This score can also be normalized
    to values between 0 and 1.
    '''
    # If the input probabilities are passed in as a dictionary
    if type(probs) is dict:
        # Calculate the score, assume probabilities are in percent
        ffsi_score = (probs["MINOR"] / 100 * 1 +
                      probs["MODERATE"] / 100 * 2 +
                      probs["SERIOUS"] / 100 * 3 +
                      probs["SEVERE"] / 100 * 4 +
                      probs["CATASTROPHIC"] / 100 * 5)

    # Else if the input probabilities are passed as a list of numbers
    elif type(probs) is list:
        # Calculate the score, assume probabilities are in percent
        ffsi_score = (probs[0] / 100 * 1 +
                      probs[1] / 100 * 2 +
                      probs[2] / 100 * 3 +
                      probs[3] / 100 * 4 +
                      probs[4] / 100 * 5)

    # If varues are to be normalized
    if normalize:
        # Divide the score by the number of classes
        ffsi_score /= 5

    # Return the calculated FFSI score
    return ffsi_score


def write_results_json(results, fname="./results/test.json", indent=2):
    ''' Write a JSON file containing LSR remarks, FFSI classification and score.

    This function writes out a dictionary of Classified LSRs into a JSON file,
    including the remarks, the probability classes, and their FFSI scores.
    '''
    # Open a new file to be written
    with open(fname, "w") as outfile:
        # Dump the dictionary as JSON
        json.dump(results, outfile, indent=indent)


def define_batches(num_reports, batch_size=100):
    '''Define a dictionary of batches to batch process a dataframe of LSRs

    This function takes the total number of reports, and calculates a number of
    batches using a specific batch size. These batches are assigned an ID, and
    index ranges are associates to them, so that the number of records in each
    batch of the input DataFrame's total size corresponds to at most the
    defined batch size.

    Batches are defined as dictionary entries with the following structure:
        {
            <batch_id> : {
                            "indices": (<start_index>, <end_index>),
                            "processed": False
                         }
        }
    where <batch_id> is an integer (starting at 0), <start_index> is the
    first index of this batch, and <end_index> is the last index of the batch.
    Note that the last index in the last batch will be equal to num_reports-1
    since the indices start at 0, and not at 1! The 'processed' key will be
    used to keep track of whether this specific batch has been processed
    successfully or not.
    '''
    # Output dictionary of batches based on the input DataFrame
    batches = {}

    # If batch size is == 0, assume no batching is desired, and output a single
    # batch holding the entirety of the num_reports
    if batch_size == 0:
        batches[batch_size] = {"indices": (0, num_reports - 1),
                               "processed": False}
    # Else if the batch size is not zero, do the appropriate calculations, and
    # define the number of full and partial batches corresponding to the
    # requested parameters
    else:
        # Number of "full" batches
        full_batches = num_reports // batch_size

        # Remainder for "partial" batch
        partial_batch = num_reports % batch_size

        # Add the "full" batches to the dictionary
        for batch in range(full_batches):
            batches[batch] = {"indices": (batch_size * batch,
                                          batch_size * (batch + 1) - 1),
                              "processed": False}

        # If available, add the partial batch to the dictionary
        if partial_batch > 0:
            batches[full_batches] = {"indices": (batch_size * full_batches,
                                                 (batch_size * full_batches)
                                                 + partial_batch - 1),
                                     "processed": False}

    # Return the batches dictionary
    return batches


def hash_filename(file_path, verbose=False):
    ''' Strip the filename from a path, and generate a 22 digit UUID from it.

    This function receives a complete file path, remove any route paths out of
    it, and then generates a unique 'short' identifier (using shortuui) based
    on the stripped filename (including the file's extension).
    '''
    # Remove any paths from the file name
    stripped_filename = str(file_path).split('/')[-1]

    # If verbose, print the original path and the stripped file name
    if verbose:
        print(f"FILE_PATH: {file_path}\nFILE_NAME: {stripped_filename}")

    # Hash the file name and return a unique identifier
    return uuid(stripped_filename)


def get_files_in_dir(dir_path="./", extension=""):
    ''' Return a list of files with the same extension in a given directory.
    '''
    # Output list holding the complete file paths
    found_files = []

    # Iterate over the list of files in the directory
    for file in os.listdir(dir_path):
        # If the file's extension matches what we are searching for
        if file.endswith(extension):
            # Add the file to the output file list
            found_files.append(os.path.join(dir_path, file))

    # Return the list of found files
    return found_files


def match_batch_results(file_uuid, batches, results_path="./results/"):
    '''

    The expected format for the batch result files is:
        <results_path>/<file_uuid>_<batch_id>.json
    '''

    # Check if there are any JSON files in the specified path
    result_files = get_files_in_dir(results_path, extension=".json")

    # Make sure that the JSON files found match the specified UUID
    result_files = list(filter(lambda x: file_uuid in x, result_files))

    # If the list of files in the requested folder, with the requested uuid is
    # empty, notify that no previous batch results were found, and return a
    # None value
    if not result_files:
        print("WARNING: No previous batch results found for current uuid!")

    # Else if there are valid files that match the uuid
    else:
        # For each of these files
        for json_file in result_files:
            # Determine its batch ID
            batch_id = json_file.split("/")[-1].split("_")[-1].split(".")[0]
            # Set this batch's processed variable as True
            batches[int(batch_id)]["processed"] = True

    # Return the batches dictionary, with updated "processed" values for all
    # the JSON batch results files found
    return batches


def read_json_results(json_file_path):
    ''' Read a JSON file containing LSR remarks, FFSI classification and score.

    This function reads in a JSON file of Classified LSRs into a dictionary,
    including the remarks, the probability classes, and their FFSI scores.
    '''
    # Open the file to be read
    with open(json_file_path, 'r') as j:
        # Load the dictionary as JSON
        contents = json.loads(j.read())

    # Return the contents of the file
    return contents

