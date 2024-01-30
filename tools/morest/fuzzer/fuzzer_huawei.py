import json
import time
import uuid
from datetime import datetime

import numpy as np
import requests

from model.operation_dependency_graph import OperationDependencyGraph
from validator.validator_huawei import validate
from .huawei_converter import HuaWeiConverter
from .runtime_dictionary import RuntimeDictionary


class APIFuzzer:

    def __init__(self, apis=[], specification={}, odg: OperationDependencyGraph = OperationDependencyGraph(),
                 time_budget=10):
        self.runtime_dict = RuntimeDictionary()
        self.apis = apis
        self.base_path = specification.get("basePath", "")
        self.server_address = "https://10.162.187.212:32018"
        self.protocol = "HTTPS"
        self.sequences = odg.generate_sequence()
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
        self.pending_request = {}
        self.pending_sequence = {}
        self.pending_add_sequence = set()
        self.pending_remove_sequence = set()
        self.pending_test_sequence_to_remove = set()
        self.status_code_status = {}
        self.time_budget = 300
        self.send_msg_count = 0
        self.receive_msg_count = 0
        self.success_sequence_count = 0
        self.success_apis = set()
        self.error_apis = set()
        self.total_apis = set()
        self.success_sequence = set()
        self.begin = time.time()
        self.success_sequence_output = []
        self.api_curve = [{
            "time": 0,
            'count': 0
        }]

    def send_testcase(self, seq):
        request_api = "http://apitestgenexe.cd-cloud-test.gamma-y.dragon.tools.huawei.com/v1/request_flow"
        resp = requests.post(request_api, json=seq)
        # mandatory check for service
        assert resp.status_code == 200

    def run(self):
        init_listener()
        # Collect All API
        for api in self.apis:
            for method in api.methods:
                self.total_apis.add(method)

        huawei_converter = HuaWeiConverter(self.runtime_dict)
        self.begin = time.time()
        begin = self.begin
        time_budget = self.time_budget
        while time.time() - begin < time_budget:

            for sequence in self.sequences:
                seq = huawei_converter.convert_sequence(self, sequence)
                self.send_testcase(seq)
                self.request_count += len(seq["sceneApis"])
                sequence_id = seq["sequenceID"]
                self.pending_request[sequence_id] = seq
                self.pending_sequence[sequence_id] = sequence
                self.send_msg_count += 1
                # self.process_response()

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
            break
            print('current seq')
            for seq in self.sequences:
                print(seq.to_str())
            # print('\n runtime dictionary')
            # for signature in self.runtime_dict.signature_to_value.keys():
            #     print(signature)

        # FIXME: consuming current response
        while self.response_count != self.request_count:
            self.process_response()
            break
        self.write_result()
        terminate_listener()

    def write_result(self):
        with open('errors.json', 'w') as data:
            json.dump({"result": self.error_sequence}, data)
        with open('success_resource_merging.json', 'w') as data:
            json.dump({"result": self.success_sequence_output}, data)
        with open('runtime.json', 'w') as data:
            json.dump({
                'keys': list(self.runtime_dict.signature_to_value.keys())
            }, data)

    def analyze_dependency_to_add(self):
        res = set()
        if len(self.sequences) > len(self.apis):
            return
        success_sequences = list(self.success_sequence)
        np.random.shuffle(success_sequences)
        for sequence in self.sequences:
            if len(sequence) > 1:
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
                    method_name, response_parameter = response.split('-')
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
                    #     res.add(new_seq)
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
                    for req, res in request_parameters:
                        new_seq.add_ref(len(new_seq) - 1, req, res)
                    res.add(new_seq)
                    print('add ref in sequence', sequence.requests[0].method_name, "to", method,
                          f'via {request_parameters}', "new sequence",
                          len(new_seq))

        #     print('request_parameter_to_methods')
        #     print(request_parameter_to_methods)
        #     print('matched_methods')
        #     print(matched_methods)
        # time.sleep(20)
        self.sequences = self.sequences.union(res)

    def process_response(self):
        while len(ResponseReader.message_queue) > 0:
            ResponseReader.lock.acquire()
            resp = ResponseReader.message_queue.pop()
            response_sequence_id = resp["sequenceID"]
            if not self.pending_request.__contains__(response_sequence_id):
                print('mix request', response_sequence_id)
                continue
            violations = validate(resp, self.apis)
            has_error = False
            # FIXME: temp: check 500
            for response in resp["result"]:
                status_code = str(response["statusCode"])
                self.runtime_dict.parse(response)
                if int(status_code) > 499 and int(status_code) < 600 and not has_error:
                    self.error_sequence.append({
                        "sequence": self.pending_request[response_sequence_id],
                        "response": resp
                    })
                    has_error = True
                result_count = self.status_code_status.get(str(status_code), 0)
                self.status_code_status[status_code] = result_count + 1
            # analyze sequence
            self.analysis_sequence(resp['result'], self.pending_sequence[response_sequence_id])

            # traverse violation
            for violation in violations:
                signature = violation.signature()
                self.violations.add(signature)

            # try to get success and error api status
            is_all_success = True
            for response in resp["result"]:
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
                if status_code == 0:
                    print(resp)
                    time.sleep(10)
                if status_code > 499 and status_code < 600:
                    self.error_apis.add(response['apiName'])

            if is_all_success:
                self.success_sequence_count += 1
                self.success_sequence_output.append({
                    "sequence": self.pending_request[response_sequence_id],
                    "response": resp
                })

            self.receive_msg_count += 1
            self.response_count += len(resp["result"])
            del self.pending_request[response_sequence_id]
            del self.pending_sequence[response_sequence_id]
            self.overall_status()
            ResponseReader.lock.release()

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
        # convert sequence
        for seq in success_sequences:
            success_converted_sequences.append(test_sequence.sub_sequence(seq))
        for seq in fail_sequences:
            failed_converted_sequences.append(test_sequence.sub_sequence(seq))
        self.pending_add_sequence = self.pending_add_sequence.union(failed_converted_sequences)
        self.pending_remove_sequence = self.pending_remove_sequence.union(success_converted_sequences)
        self.pending_test_sequence_to_remove.add(test_sequence)

    def overall_status(self):
        status_stat = ""
        elapsed = time.time() - self.begin
        for status_code in self.status_code_status.keys():
            status_stat += f'{status_code}:{self.status_code_status[status_code]},' \
                           f'{float(self.status_code_status[status_code]) / self.response_count} '
        print(
            f'{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}, Fuzzing Time: {elapsed}s, Send Msg Count: {self.send_msg_count}, '
            f'Receive Msg Count: {self.receive_msg_count}, Success Response Sequence Count: {self.success_sequence_count}, '
            f'Success Sequence Rate: {float(self.success_sequence_count) / self.receive_msg_count}, '
            f'Success API: {float(len(self.success_apis)) / len(self.total_apis)}'
            f' ({len(self.success_apis)}/{len(self.total_apis)}),'
            f'Error API: {float(len(self.error_apis)) / len(self.total_apis)}'
            f' ({len(self.error_apis)}/{len(self.total_apis)}),'
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
        print(self.api_curve)
        print(self.runtime_dict.signature_to_value.keys())
        print(status_stat)
