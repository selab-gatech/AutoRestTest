import json
import time
import uuid
from datetime import datetime
import sys

import numpy as np
import requests

from model.operation_dependency_graph import OperationDependencyGraph
from validator.validator import validate
from validator.validator_payload import validate_payload
from .normal_executor import SequenceConverter
from .mutation_executor import mutationSequenceConverter
from .runtime_dictionary import RuntimeDictionary
from model.sequence import SequenceOrigin


class APIFuzzer:

    def __init__(self, apis=[], specification={}, odg: OperationDependencyGraph = OperationDependencyGraph(),
                 host_address="", pre_defined_headers={},
                 time_budget=300):
        self.runtime_dict = RuntimeDictionary()
        self.apis = apis
        self.server_address = host_address
        self.protocol = "HTTPS"
        self.odg = odg
        self.sequences = odg.generate_sequence()
        self.odg_sequences = set(self.sequences)
        self.specification = specification
        self.start_time = time.time()
        # for test case limit time
        self.time_limit = 2000
        self.end_time = self.start_time + time_budget
        self.seed_id = uuid.uuid4().__str__()
        self.sequence_id = uuid.uuid4().__str__()
        self.iteration_id = 0
        self.request_count = 0
        self.response_count = 0
        self.violations = set()
        self.error_sequence = []
        self.violations_sequence = []
        self.pending_add_sequence = set()
        self.pending_remove_sequence = set()
        self.pending_test_sequence_to_remove = set()
        self.status_code_status = {}
        self.time_budget = time_budget
        self.send_msg_count = 0
        self.receive_msg_count = 0
        self.success_sequence_count = 0
        self.success_apis = set()
        self.error_apis = set()
        self.total_apis = set()
        self.success_sequence = set()
        self.begin = time.time()
        self.success_sequence_output = []
        self.success_endpoint = set()
        self.error_endpoint = set()
        self.error_api_curve = []
        self.success_endpoint_api_curve = []
        self.error_endpoint_api_curve = []
        self.pre_defined_headers = pre_defined_headers
        self.total_apis_map = {}
        self.api_curve = [{
            "time": 0,
            'count': 0
        }]

    def run(self):
        # Collect All API
        for api in self.apis:
            for method in api.methods:
                self.total_apis.add(method)
                self.total_apis_map[method.method_signature] = method

        executor = SequenceConverter(self.runtime_dict)
        self.begin = time.time()
        begin = self.begin
        time_budget = self.time_budget
        previous_success_size = 0
        while time.time() - begin < time_budget:
            # Phase 1
            for sequence in self.sequences:
                try_times = 1 if len(sequence) > 1 else 20
                for _ in range(try_times):
                    res = executor.execute_sequence(self, sequence)
                    self.request_count += len(res)
                    self.send_msg_count += 1
                    self.process_response(res, sequence)
                    if time.time() - begin > time_budget:
                        break
            if time.time() - begin > time_budget:
                break

            self.iteration_id += 1

            # remove test sequence
            self.sequences = self.sequences.difference(self.pending_test_sequence_to_remove)
            self.pending_test_sequence_to_remove = set()

            # remove sequences
            self.sequences = self.sequences.difference(self.pending_remove_sequence)
            self.success_sequence = self.success_sequence.union(self.pending_remove_sequence)
            self.pending_remove_sequence = set()

            # add sequences
            self.sequences = self.sequences.union(self.pending_add_sequence)
            self.pending_add_sequence = set()

            # analyze dependency to add
            self.analyze_dependency_to_add()
            print('current seq')
            for seq in self.sequences:
                print(seq.to_str())
            # Phase 2
            # if previous_success_size != len(self.success_sequence) and np.random.random() > 0.95:
            #     previous_success_size = len(self.success_sequence)
            #     new_synthesis_sequence = self.synthesis_resource_sequence(executor)
            #     self.sequences = self.sequences.union(new_synthesis_sequence)

            if len(self.sequences) == 0:
                self.sequences = set(self.odg_sequences)
        self.write_result()
        # executor.draw_response_dict()

    def mutation_fuzz_run(self):
        print("Continue to mutation fuzzing \n-------------------- \n")
        time.sleep(3)
        # re_initialize the fuzzer
        self.runtime_dict = RuntimeDictionary()
        self.mutation_runtime_dict = RuntimeDictionary()
        # re-initialize error sequence
        self.error_sequence = []
        # create a pair of normal sequence executor and mutation sequence executor
        normalExecutor = SequenceConverter(self.runtime_dict)
        mutationExecutor = mutationSequenceConverter(self.mutation_runtime_dict)
        mutation_time_budget = self.time_budget  # make it a time budget parameter later
        mutation_fuzz_begin = time.time()
        while time.time() - mutation_fuzz_begin < mutation_time_budget:
            for sequence in self.success_sequence:
                # check pattern
                # TODO: update sequence checking
                path_to_check = ["post get"]
                req_methods_path = ''
                for i, method in enumerate(sequence.requests):
                    req_methods_path += method.method_type + ' '
                proceed = False
                for path in path_to_check:
                    if path in req_methods_path:
                        proceed = True
                if not proceed:
                    continue
                for mutation_index, method in enumerate(sequence.requests):
                    baseline_res = normalExecutor.execute_sequence(self, sequence)
                    res = mutationExecutor.execute_sequence(self, sequence,
                                                            mutation_index)  # i labels the position of mutation
                    self.request_count += len(res)
                    self.send_msg_count += 1
                    self.process_mutation_response(baseline_res, res, sequence, mutation_index)
        self.write_mutation_result()
        print(self.violations)

    def synthesis_resource_sequence(self, executor):
        print('synthesis resource')
        add_sequence_size = 100
        matched_pair = {}
        success_sequence = list(self.success_sequence)
        np.random.shuffle(success_sequence)
        success_sequence = success_sequence[:add_sequence_size]
        new_sequences = set()
        graph = executor.response_resource_graph.get("graph")
        # extract method pair
        for method_pair in graph.keys():
            first_method, second_method = method_pair
            first_method_set = matched_pair.get(first_method, set())
            second_method_set = matched_pair.get(second_method, set())
            matched_pair[first_method] = first_method_set
            matched_pair[second_method] = second_method_set
            first_method_set.add(method_pair)
            second_method_set.add(method_pair)
        # iterate each sequence
        for sequence in success_sequence:
            # omit single element sequence
            if len(sequence) == 1:
                continue
            # we do not want to add more sequences in runtime
            if np.random.random() < 0.99:
                continue
            for i in range(len(sequence)):
                current_method = sequence[i]
                # omit the last one
                if i == len(sequence) - 1:
                    continue
                # go through the
                if not (current_method.method_signature in matched_pair):
                    continue

                # by chance to omit current method
                # if np.random.random() < 0.5:
                #     continue

                # here we go through the pairs
                for method_pair in matched_pair[current_method.method_signature]:
                    pair_method = None
                    if method_pair[0] == current_method.method_signature:
                        pair_method = self.total_apis_map[method_pair[1]]
                    else:
                        pair_method = self.total_apis_map[method_pair[0]]
                    duplicate_sequence = sequence.duplicate()
                    # Change method
                    duplicate_sequence.set_method(i, pair_method)
                    # Check reference
                    matched_reference_pairs = graph[method_pair]
                    ref = duplicate_sequence.get_ref(i + 1).get_refs()
                    for refer_pair in matched_reference_pairs:
                        first_attribute, second_attribute = refer_pair
                        if first_attribute in ref:
                            ref[second_attribute] = ref[first_attribute]
                            ref.__delitem__(first_attribute)
                        if second_attribute in ref:
                            ref[first_attribute] = ref[second_attribute]
                            ref.__delitem__(second_attribute)
                    duplicate_sequence.origin = SequenceOrigin.RESOURCE_MERGING
                    sequence_requests = set()
                    is_valid = True
                    # sequence should not contains the same request
                    for seq_request in duplicate_sequence.requests:
                        if sequence_requests.__contains__(seq_request):
                            is_valid = False
                            break
                        else:
                            sequence_requests.add(seq_request)
                    if not is_valid:
                        continue
                    new_sequences.add(duplicate_sequence)
                    duplicate_sequence.sequence_trace.append([sequence.origin, str(sequence)])

        return new_sequences

    def write_brief_result(self, target_dir):
        orig_stdout = sys.stdout
        with open(target_dir + '/errors.json', 'w') as data:
            json.dump({"result": self.error_sequence}, data)

        with open(target_dir + '/violations_sequence.json', 'w') as data:
            json.dump({"result": self.violations_sequence}, data)

        f = open(target_dir + '/brief.text', 'w')
        sys.stdout = f
        self.overall_status()
        sys.stdout = orig_stdout
        f.close()

    def write_result(self):
        with open('errors.json', 'w') as data:
            json.dump({"result": self.error_sequence}, data)
        with open('success_resource_merging.json', 'w') as data:
            json.dump({"result": self.success_sequence_output}, data)
        with open('runtime.json', 'w') as data:
            json.dump({
                'keys': list(self.runtime_dict.signature_to_value.keys())
            }, data)

    def write_mutation_result(self):
        with open('violations_sequence.json', 'w') as data:
            json.dump({"result": self.violations_sequence}, data)

    def analyze_dependency_to_add(self):
        generated_sequences = set()
        if len(self.sequences) > len(self.apis):
            return
        if np.random.random() > 0.05:
            return
        success_sequences = list(self.success_sequence)
        np.random.shuffle(success_sequences)
        for sequence in self.sequences:
            # give chance to add longer sequence
            if len(sequence) > 1 and np.random.random() > 0.01:
                continue
            request_nominal_parameters = sequence.get_request_parameter_by_index(0)
            response_nominal_runtime_parameters = self.runtime_dict.signature_to_value.keys()
            matched_methods = {}
            request_parameter_to_methods = {}
            method_recorded_response = {}
            method_to_request = {}
            for request_parameter in request_nominal_parameters:
                chunked_token = request_parameter.lower().split('.')[-1]
                for response in response_nominal_runtime_parameters:
                    method_name, response_parameter = response.split(self.runtime_dict.signature_splitter)
                    chunked_response_token = response_parameter.lower().split('.')[-1]
                    if chunked_token != chunked_response_token:
                        continue
                    matched_parameters = matched_methods.get(method_name, set())
                    matched_parameters.add(response_parameter)
                    matched_methods[method_name] = matched_parameters
                    request_parameter_to_meth = request_parameter_to_methods.get(request_parameter, set())
                    request_parameter_to_meth.add(method_name)
                    request_parameter_to_methods[request_parameter] = request_parameter_to_meth
                    method_to_response = method_recorded_response.get(method_name, set())
                    method_to_response.add(response_parameter)
                    method_recorded_response[method_name] = method_to_response
                    method_to_req = method_to_request.get(method_name, set())
                    method_to_req.add((request_parameter, response_parameter))
                    method_to_request[method_name] = method_to_req
                    # for success_sequence in self.success_sequence:
                    #     if np.random.random() < 0.5:
                    #         break
                    #     if not success_sequence.has_method(method_name):
                    #         continue
                    #     new_seq = success_sequence.slice_by_method_name(method_name)
                    #     new_seq.add_def(len(new_seq) - 1, response_parameter)
                    #     new_seq.add_method(sequence.requests[0])
                    #     new_seq.add_ref(len(new_seq) - 1, request_parameter, response_parameter)
                    #     res.add(new_seq){'placeOrder', 'updateUser', 'createUser'}
                    #     print('add ref in sequence', sequence.requests[0].method_name, "to", method_name,
                    #           f'via {response_parameter} -> {request_parameter}', "new sequence",
                    #           len(new_seq))
            for method in method_recorded_response:
                matched_seq = None
                for success_seq in success_sequences:
                    if success_seq.has_method(method):
                        matched_seq = success_seq
                        break
                if matched_seq:
                    request_parameters = method_to_request[method]
                    response_parameters = method_recorded_response[method]
                    new_seq = success_seq.slice_by_method_name(method)
                    for response_parameter in response_parameters:
                        new_seq.add_def(len(new_seq) - 1, response_parameter)
                    new_seq.add_method(sequence.requests[0])
                    new_seq.origin = SequenceOrigin.DYNAMIC_FIX_ADD
                    for req, res in request_parameters:
                        new_seq.add_ref(len(new_seq) - 1, new_seq[len(new_seq) - 1], req, res)
                    new_seq.sequence_trace.append([success_seq.origin, str(success_sequences)])

                    sequence_requests = set()
                    is_valid = True
                    # sequence should not contains the same request
                    for seq_request in new_seq.requests:
                        if sequence_requests.__contains__(seq_request):
                            is_valid = False
                            break
                        else:
                            sequence_requests.add(seq_request)
                    if not is_valid:
                        break
                    generated_sequences.add(new_seq)
                    print('add ref in sequence', sequence.requests[0].method_name, "to", method,
                          f'via {request_parameters}', "new sequence",
                          len(new_seq))
                    break

        #     print('request_parameter_to_methods')
        #     print(request_parameter_to_methods)
        #     print('matched_methods')
        #     print(matched_methods)
        # time.sleep(20)
        self.sequences = self.sequences.union(generated_sequences)

    def process_response(self, resp, sequence):
        violations = validate(resp, self.apis)
        has_error = False
        # FIXME: temp: check 500
        for ind, response in enumerate(resp):
            status_code = str(response["statusCode"])
            self.runtime_dict.parse(sequence[ind], response)
            if int(status_code) > 499 and int(status_code) < 600 and not has_error:
                self.error_sequence.append({
                    "testcase": resp
                })
                has_error = True
            result_count = self.status_code_status.get(str(status_code), 0)
            self.status_code_status[status_code] = result_count + 1
        # analyze sequence
        self.analysis_sequence(resp, sequence)

        # traverse violation
        for violation in violations:
            signature = violation.signature()
            self.violations.add(signature)

        # try to get success and error api status
        is_all_success = True
        for response_ind, response in enumerate(resp):
            status_code = response["statusCode"]
            if status_code > 299:
                is_all_success = False
            if status_code > 199 and status_code < 300:
                if not self.success_apis.__contains__(response['apiName']):
                    self.api_curve.append({
                        "time": time.time() - self.begin,
                        'count': len(self.success_apis) + 1
                    })
                method_path = sequence[response_ind].method_path
                if not self.success_endpoint.__contains__(method_path):
                    self.success_endpoint_api_curve.append({
                        "time": time.time() - self.begin,
                        'count': len(self.success_endpoint) + 1
                    })
                print(method_path, 'method path')
                self.success_endpoint.add(method_path)
                self.success_apis.add(response["apiName"])
            if status_code > 499 and status_code < 600:
                if not self.error_apis.__contains__(response['apiName']):
                    self.error_api_curve.append({
                        "time": time.time() - self.begin,
                        'count': len(self.error_apis) + 1
                    })
                method_path = sequence[response_ind].method_path
                if not self.error_endpoint.__contains__(method_path):
                    self.error_endpoint_api_curve.append({
                        "time": time.time() - self.begin,
                        'count': len(self.error_endpoint) + 1
                    })
                self.error_endpoint.add(method_path)
                self.error_apis.add(response['apiName'])

        if is_all_success:
            self.success_sequence_count += 1
            self.success_sequence_output.append({
                "testcase": resp,
                "source": sequence.origin,
                "testcase_length": len(sequence),
                "sequence_trace": sequence.sequence_trace,
                "sequence": str(sequence)
            })

        self.receive_msg_count += 1
        self.response_count += len(resp)
        self.overall_status()

    def process_mutation_response(self, baseline_resp, resp, sequence, mutation_index):
        violations = validate_payload(baseline_resp, resp, mutation_index, self.apis)
        has_error = False
        # FIXME: temp: check 500
        for ind, response in enumerate(resp):
            status_code = str(response["statusCode"])
            self.runtime_dict.parse(sequence[ind], response)
            if int(status_code) > 499 and int(status_code) < 600 and not has_error:
                self.error_sequence.append({
                    "testcase": resp
                })
                has_error = True
            result_count = self.status_code_status.get(str(status_code), 0)
            self.status_code_status[status_code] = result_count + 1
        # analyze sequence
        self.analysis_sequence(resp, sequence)

        # traverse violation
        for violation in violations:
            signature = violation.signature()
            self.violations.add(signature)
        # store violation result
        if len(violations) != 0:
            current_violation_signatures = set()
            for violation in violations:
                signature = violation.signature()
                current_violation_signatures.add(signature)
            self.violations_sequence.append({
                "violation": str(current_violation_signatures),
                "testcase": resp,
                "baseline": baseline_resp
            })

        # try to get success and error api status
        is_all_success = True
        for response in resp:
            status_code = response["statusCode"]
            if status_code > 299:
                is_all_success = False
            if status_code > 199 and status_code < 300:
                if not self.success_apis.__contains__(response['apiName']):
                    self.api_curve.append({
                        "time": time.time() - self.begin,
                        'count': len(self.success_apis) + 1
                    })
                self.success_apis.add(response["apiName"])
            if status_code > 499 and status_code < 600:
                self.error_apis.add(response['apiName'])

        if is_all_success:
            self.success_sequence_count += 1
            self.success_sequence_output.append({
                "testcase": resp
            })

        self.receive_msg_count += 1
        self.response_count += len(resp)
        self.overall_status()  # verbose set to true/false to print status

    def has_success_api(self, responses):
        for resp in responses:
            status_code = resp["statusCode"]
            if status_code < 300 and status_code > 199:
                return True
        return False

    def chunk_responses(self, responses):
        success_sequence = []
        fail_sequence = []
        current_fail_sequence = []
        current_success_sequence = []

        # handle whole error status
        if not self.has_success_api(responses):
            for i in range(len(responses)):
                fail_sequence.append([i])
            return success_sequence, fail_sequence

        for index in range(len(responses)):
            status_code = responses[index]["statusCode"]
            if status_code < 300 and status_code > 199:
                current_success_sequence.append(index)
                if len(current_fail_sequence) > 0:
                    fail_sequence.append(current_fail_sequence)
                    current_fail_sequence = []
            else:
                current_fail_sequence.append(index)
                if len(current_success_sequence) > 0:
                    success_sequence.append(current_success_sequence)
                    current_success_sequence = []
        if len(current_success_sequence) > 0:
            success_sequence.append(current_success_sequence)
        else:
            fail_sequence.append(current_fail_sequence)
        return success_sequence, fail_sequence

    def analysis_sequence(self, responses, test_sequence):
        success_converted_sequences = []
        failed_converted_sequences = []
        success_sequences, fail_sequences = self.chunk_responses(responses)
        assert test_sequence.origin is not None
        # convert sequence
        for seq in success_sequences:
            new_seq = test_sequence.sub_sequence(seq)
            if len(seq) == len(test_sequence):
                new_seq.origin = test_sequence.origin
            else:
                new_seq.origin = SequenceOrigin.DYNAMIC_FIX_DELETE
            new_seq.sequence_trace.append([test_sequence.origin, str(test_sequence)])
            success_converted_sequences.append(new_seq)
        for seq in fail_sequences:
            new_seq = test_sequence.sub_sequence(seq)
            new_seq.origin = SequenceOrigin.DYNAMIC_FIX_DELETE
            new_seq.sequence_trace.append([test_sequence.origin, str(test_sequence)])
            failed_converted_sequences.append(new_seq)
        self.pending_add_sequence = self.pending_add_sequence.union(failed_converted_sequences)
        self.pending_remove_sequence = self.pending_remove_sequence.union(success_converted_sequences)
        self.pending_test_sequence_to_remove.add(test_sequence)

    def overall_status(self):
        status_stat = ""
        elapsed = time.time() - self.begin
        for status_code in self.status_code_status.keys():
            status_stat += f'{status_code}:{self.status_code_status[status_code]},' \
                           f'{float(self.status_code_status[status_code]) / self.response_count} '

        # collect the sequence source
        sequence_source = {}
        for seq in self.success_sequence:
            sequence_source[seq.origin] = sequence_source.get(seq.origin, 0) + 1
        print(
            f'{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}, Fuzzing Time: {elapsed}s, Send Msg Count: {self.send_msg_count}, '
            f'Receive Msg Count: {self.receive_msg_count}, Success Response Sequence Count: {self.success_sequence_count}, '
            f'Success Sequence Rate: {float(self.success_sequence_count) / self.receive_msg_count}, '
            f'Success API: {float(len(self.success_apis)) / len(self.total_apis)}'
            f' ({len(self.success_apis)}/{len(self.total_apis)}),'
            f'Error API: {float(len(self.error_apis)) / len(self.total_apis)}'
            f' ({len(self.error_apis)}/{len(self.total_apis)}),'
            f'Success Endpoint: {float(len(self.success_endpoint)) / len(self.apis)}'
            f' ({len(self.success_endpoint)}/{len(self.apis)}),'
            f'Error Endpoint: {float(len(self.error_endpoint)) / len(self.apis)}'
            f' ({len(self.error_endpoint)}/{len(self.apis)}),'
            f' Tested Error and Success Unique API: {float(len(set.union(self.success_apis, self.error_apis))) / len(self.total_apis)}'
            f' ({len(set.union(self.success_apis, self.error_apis))}/{len(self.total_apis)}),'
            f' Request Count: {self.request_count}, '
            f'Response Count: {self.response_count}, Response / Request: {float(self.response_count) / self.request_count}, '
            f'Request(QPS): {float(self.request_count) / elapsed}, Response(QPS):'
            f' {float(self.response_count) / elapsed}, Violation Count: {len(self.violations)}'
            f' Remain Sequence Count: {len(self.sequences)}'
            f' Success Sequence: {len(self.success_sequence)}')
        print('success')
        print(self.success_apis)
        print('errors')
        print(self.error_apis)
        print('==API Curve')
        print({
            "success_api": self.api_curve,
            "error_api": self.error_api_curve,
            "success_endpoint": self.success_endpoint_api_curve,
            "error_endpoint": self.error_endpoint_api_curve,
        })
        print(self.runtime_dict.signature_to_value.keys())
        print(status_stat)
        # print(sequence_source)
        print("sequence source", sequence_source)
