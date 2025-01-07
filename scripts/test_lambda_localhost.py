import requests

url_predict = "http://localhost:8080/2015-03-31/functions/function/invocations"

data={ 'url':  'http://bit.ly/mlbookcamp-pants'}

result = requests.post(url_predict, json=data)
print(result.text)