from .classification_prompts import FEW_SHOT_CLASSIFICATON_PREFIX, CLASSIFICATION_SUFFIX
from bs4 import BeautifulSoup
from openai import OpenAI
import os

class ResponseHandler:
    def __init__(self):
        self.parser_type = "html.parser"
        self.language_model = ResponseLanguageModelHandler()
    def extract_response_text(self,response):
        if not response:
            raise ValueError()
        response_text = response.text
        result = ' '.join(BeautifulSoup(response_text, self.parser_type).stripped_strings)
        return result
    def classify_error(self, response):
        response_text = self.extract_response_text(response)
        return self.language_model.classify_response(response_text) 
    def handle_error(self, response):
        error_classification = self.classify_error(response)
        if error_classification == "PARAMETER CONSTRAINT":
            pass 
        elif error_classification == "FORMAT":
            pass
        elif error_classification == "PARAMETER DEPENDENCY":
            pass
        elif error_classification == "OPERATION DEPENDENCY":
            pass
        else:
            return None
class ResponseLanguageModelHandler:
    def __init__(self, language_model="OPENAI", **kwargs):
        if language_model == "OPENAI":
            env_var = os.getenv("OPENAI_API_KEY")
            if env_var is None or env_var.strip() == "":
                raise ValueError()
            self.client = OpenAI()
        else:
            raise Exception("Unsupported language model")        
    def language_model_query(self, response_text):
        #get openai chat completion 
        return self.client.chat.completions.create(
            engine="gpt-4-turbo-preview",
            messages=[
                {'role': 'user', 'content': FEW_SHOT_CLASSIFICATON_PREFIX + query + CLASSIFICATION_SUFFIX},
            ]
        ).choices[0].message['content']
    def extract_classification(self, response_text):
        classification = None
        if "PARAMETER CONSTRAINT" in response_text:
            classification = "PARAMETER CONSTRAINT"
        elif "FORMAT" in response_text:
            classification = "FORMAT"
        elif "PARAMETER DEPENDENCY" in response_text:
            classification = "PARAMETER DEPENDENCY"
        elif "OPERATION DEPENDENCY" in response_text:
            classification = "OPERATION DEPENDENCY"
        return classification
    def classify_response(self, response_text):
        return self.extract_classification(self.language_model_query(response_text))
class DummyResponse:
    def __init__(self, type="html"):
        if type == "html":
            self.text = '''
            <!doctype html><html lang="en"><head><script async src="https://www.googletagmanager.com/gtag/js?id=UA-17134933-4"></script><script src="https://kit.fontawesome.com/6be4547409.js" crossorigin="anonymous"></script><script>function gtag(){dataLayer.push(arguments)}window.dataLayer=window.dataLayer||[],gtag("js",new Date),gtag("config","UA-17134933-4")</script><meta charset="utf-8"/><link rel="icon" href="/favicon.ico"/><meta name="viewport" content="width=device-width,initial-scale=1"/><meta name="theme-color" content="#000000"/><link rel="manifest" href="/manifest.json"/><title>Genome Nexus</title><link href="/static/css/2.e87a3ba9.chunk.css" rel="stylesheet"><link href="/static/css/main.276d8240.chunk.css" rel="stylesheet"></head><body><noscript>You need to enable JavaScript to run this app.</noscript><div id="root"></div><script>!function(e){function r(r){for(var n,f,l=r[0],i=r[1],a=r[2],c=0,s=[];c<l.length;c++)f=l[c],Object.prototype.hasOwnProperty.call(o,f)&&o[f]&&s.push(o[f][0]),o[f]=0;for(n in i)Object.prototype.hasOwnProperty.call(i,n)&&(e[n]=i[n]);for(p&&p(r);s.length;)s.shift()();return u.push.apply(u,a||[]),t()}function t(){for(var e,r=0;r<u.length;r++){for(var t=u[r],n=!0,l=1;l<t.length;l++){var i=t[l];0!==o[i]&&(n=!1)}n&&(u.splice(r--,1),e=f(f.s=t[0]))}return e}var n={},o={1:0},u=[];function f(r){if(n[r])return n[r].exports;var t=n[r]={i:r,l:!1,exports:{}};return e[r].call(t.exports,t,t.exports,f),t.l=!0,t.exports}f.m=e,f.c=n,f.d=function(e,r,t){f.o(e,r)||Object.defineProperty(e,r,{enumerable:!0,get:t})},f.r=function(e){"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},f.t=function(e,r){if(1&r&&(e=f(e)),8&r)return e;if(4&r&&"object"==typeof e&&e&&e.__esModule)return e;var t=Object.create(null);if(f.r(t),Object.defineProperty(t,"default",{enumerable:!0,value:e}),2&r&&"string"!=typeof e)for(var n in e)f.d(t,n,function(r){return e[r]}.bind(null,n));return t},f.n=function(e){var r=e&&e.__esModule?function(){return e.default}:function(){return e};return f.d(r,"a",r),r},f.o=function(e,r){return Object.prototype.hasOwnProperty.call(e,r)},f.p="/";var l=this["webpackJsonpgenome-nexus-frontend"]=this["webpackJsonpgenome-nexus-frontend"]||[],i=l.push.bind(l);l.push=r,l=l.slice();for(var a=0;a<l.length;a++)r(l[a]);var p=i;t()}([])</script><script src="/static/js/2.3efbd946.chunk.js"></script><script src="/static/js/main.92aaf9d8.chunk.js"></script></body></html>
            '''.strip()
        if type == "json":
            self.text = '''
            {"variant":"cjgQ3mGPnuaXGKU","originalVariantQuery":"cjgQ3mGPnuaXGKU","successfully_annotated":false}
            '''.strip()
        if type == "plaintext":
            self.text = '''
            successful            
            '''.strip()

        
if __name__ == '__main__':
    response_handler = ResponseHandler() 
    print(response_handler.extract_response_text(DummyResponse(type="html")))
