import requests
from utils.json_compare import json_compare

# General idea: The normal and test responses are sent in a request format.
# The key is to compare the two responses and check the differences

def responseCheck(baseline, testcase):
    '''
    :param baseline: the baseline response as a response object
    :param testcase: the testcase response as a response object
    :return:
    '''
    # Define types of returns
    results = {
        0: [0, "baseline_error","Baseline status code 500+. Check baseline request and response."],
        1: [1, "500_error", "Testcase response status 500+. Bug detected."],
        2: [2, "Server response differently", "API endpoint may be well protected" ],
        3: [3, "Server response normally", "Hidden State"]
    }
    similarity = 0
    # Check 1: HTTP status code
    baseline_status = baseline.status_code
    testcase_status = testcase.status_code

    # In normal cases, the baseline response should not be 500+ status code
    if baseline_status >= 500:
        result = results[0]
        return similarity, result

    if baseline_status == testcase_status:
        # status_code is the same
        status_code_same = True
    else:
        status_code_same = False

    # Check 2: Error Response
    # todo: add detailed oracle for 400+ handling
    if not status_code_same:
        if testcase_status >= 500: # type 1 error: server bug detected
            result = results[1]
            return similarity, result
        else:
            pass


    # Check 3: Reject Response
    if status_code_same:
        # need a nice way to identify reject response
        # Here we test with Jaccard distance
        similarity = json_compare(baseline.text, testcase.text)
        if similarity <= 0.5:
            result = results[2]
        else:
            result = results[3]
        return similarity, result









