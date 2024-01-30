# api-fuzzer-hw

1. Install dependencies in requirements.txt (pip3 install -r requirements.txt). The tool is tested on Ubuntu 18.04 and MacOS. There might be one or two python3 dependencies that are not listed in requirements.txt due to later development; when you encounter it, please just do the installation manually based on python3 errors.

2. To run the tool, use `python3 fuzzer.py`. 

3. To modify the configurations, in `main` function of `fuzzer.py`

   ```
   def main():
       parser = ResolvingParser(api-documentation, 
                                recursion_limit_handler=default_reclimit_handler)
       apis, odg = parse(parser.specification)
       headers = headers # if needed
       api_fuzzer = APIFuzzer(apis, parser.specification, odg, target_ip,
                              pre_defined_headers=headers)
       api_fuzzer.run()
   ```

   for openapi 2.0 documentations, please use: 
	```
    parser = ResolvingParser(api-documentation, 
                             recursion_limit_handler=default_reclimit_handler)
	```
	
	for openapi 3.0 documentations, please add an backend parameter:
	
	```
	 parser = ResolvingParser(api-documentation, backend = 'open-api-validator',
	                          recursion_limit_handler=default_reclimit_handler)
	```

4. The results are written in three files: success.json, error.json and runtime.json.
5. A common error is that API baseurl is not properly defined in documentation. For instance, the `url` for petstore is `https://petstore.swagger.io`, and its basepath is `/v2`, so the tool will send request to `https://petstore.swagger.io/v2`. We notice that many documentations add an extra slash at the end of url (`https://petstore.swagger.io/`), so after processing, it becomes https://petstore.swagger.io//v2, which causes an error in testing.  In this case, please edit the documentation manually. 
