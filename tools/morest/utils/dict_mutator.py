import copy
import requests

def dict_generator(indict, pre=None):
    pre = pre[:] if pre else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                for d in dict_generator(value, pre + [key]):
                    yield d
            elif isinstance(value, list) or isinstance(value, tuple):
                for v in value:
                    for d in dict_generator(v, pre + [key]):
                        yield d
            else:
                yield pre + [key, value]
    else:
        yield pre + [indict]

def change_value(jsonData,lst,new_value):
    toModifyJson = copy.deepcopy(jsonData)
    tempJson = toModifyJson

    for i in lst[:-2]:
        tempJson = tempJson[i]
    tempJson[lst[-2]] = new_value

    return toModifyJson

def tamper_all_parameter_values(extractedFlatLists,tamperedValueList):
    tamperedValueJsonList = []
    keyList = []
    tValueList = []
    for flatList in extractedFlatLists:
        #FlatList example format: flatList = ['auth', 'passwordCredentials', 'username', 'USER_NAME'] 

        oriKey = flatList[-2]

        #Value Tampering
        for tValue in tamperedValueList:
            tamperedValueJsonList.append(change_value(jsonData, flatList, tValue))
            keyList.append(oriKey)
            tValueList.append(tValue)

    return tamperedValueJsonList,keyList,tValueList

def change_key(jsonData,lst,new_key):
    toModifyJson = copy.deepcopy(jsonData)
    tempJson = toModifyJson

    for i in lst[:-2]:
        tempJson = tempJson[i]
    tempJson.pop(lst[-2])
    tempJson[new_key] = lst[-1]

    return toModifyJson

def tamper_all_parameter_keys(extractedFlatLists,tamperedKeyList):
    tamperedKeyJsonList = []
    tKeyList = []
    valueList = []
    for flatList in extractedFlatLists:
        #FlatList example format: flatList = ['auth', 'passwordCredentials', 'username', 'USER_NAME'] 
        oriValue = flatList[-1]

        #Key Tampering
        for tKey in tamperedKeyList:
                tamperedKeyJsonList.append(change_key(jsonData, flatList, tKey))
                tKeyList.append(tKey)
                valueList.append(oriValue)

    return tamperedKeyJsonList,tKeyList,valueList

def send_all_methods(http_methods,url,payload,key,value):
    for method in http_methods:
        response = requests.request(method,url,json=payload)
        #print(key,":",value,"     ",method,"    ",response.status_code,"    ",len(response.content))
        print("{:20.15}:{:20.15}{:10.8}{:15}{:10}".format(key, value, method, str(response.status_code), str(len(response.content))))

if __name__ == "__main__":

    #Read from textfile/input
    jsonData = {"auth":
        {"passwordCredentials":
            {"username": "USER_NAME",
              "password":"PASSWORD"}
        }
    }

    #Doesn't work for json nested using List
    #Only works for json nested using Dictionary
    #Eg:
    '''
    jsonData = {
   "username" : "my_username",
   "password" : "my_password",
   "validation-factors" : {
      "validationFactors" : [
         {
            "name" : "remote_address",
            "value" : "127.0.0.1"
         }
      ]
   }
}
'''

    HTTP_METHODS_RESTFUL = ['GET','POST','PUT','DELETE','PATCH']
    HTTP_METHODS_NON_RESTFUL = ['HEAD','OPTIONS','TRACE']
    HTTP_METHODS_POLLUTED = ['GVT','TEST','-1']
    HTTP_METHODS = HTTP_METHODS_RESTFUL + HTTP_METHODS_NON_RESTFUL + HTTP_METHODS_POLLUTED

    extractedFlatLists = list(dict_generator(jsonData))
    tamperedValueList = ['qwerty','<script> alert(\'xss\')</script>','<img src=1 onerror=alert("XSS")>'] #Read from textfile
    tamperedKeyList = ['qwerty','<script> alert(\'xss\')</script>','<img src=1 onerror=alert("XSS")>'] #Read from textfile

    tamperedValueJsonList,tKeyList,valueList = tamper_all_parameter_values(extractedFlatLists,tamperedValueList)
    #print("TamperedValueJSONList: ",tamperedValueJsonList)
    tamperedKeyJsonList,keyList,tValueList = tamper_all_parameter_keys(extractedFlatLists,tamperedKeyList)
    #print("TamperedKeyJSONList: ",tamperedKeyJsonList)
    combinedTamperedJsonList = tamperedValueJsonList + tamperedKeyJsonList
    combinedKeyList = tKeyList + keyList
    combinedValueList = valueList + tValueList
    print("Request Prepared. Sending requests..")
    print("--------------------------------------------------------------------------------------")


    #response = requests.request("CONNECT","http://127.0.0.1:3000/user",json=tamperedValueJsonList[0])
    #print(response)
    print("{:20}{:20}{:10}{:15}{:10}".format("Key","Value", "Method", "Response Code", "Response Length"))
    #print("Modified Parameter    Method      Response Code    Response Length")
    print("--------------------------------------------------------------------------------------")

    for i in range(0,len(combinedTamperedJsonList)):
        #May need to create custom request to modify headers to spoof as a real web request 
        #Modify the URL accordingly
        send_all_methods(HTTP_METHODS_RESTFUL,"http://127.0.0.1:3000/user",combinedTamperedJsonList[i], combinedKeyList[i], combinedValueList[i])
