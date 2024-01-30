def build_request(method, parameters):
    params = {}
    data = {}
    url = method.method_path
    headers = {}
    files = {}
    form_data = {}
    for parameter_pair in parameters:
        parameter, val = parameter_pair
        body = parameter.raw_body
        parameter_type = body["in"]
        if parameter_type == "header":
            headers[body["name"]] = val
        elif parameter_type == "query":
            params[body["name"]] = val
        elif parameter_type == "path":
            url = str(url)
            url = url.replace('{' + str(body["name"]) + '}', str(val))
        elif parameter_type == "formData":
            if body.__contains__('type') and body['type'] == 'file':
                files[body['name']] = val
                continue
            form_data[body["name"]] = val
        elif parameter_type == 'body':
            data = val
        else:
            raise Exception("Unrecognized type", parameter.raw_body)
    return url, params, data, headers, files,form_data
