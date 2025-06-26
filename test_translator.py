import requests, uuid, json
import os
from dotenv import load_dotenv

load_dotenv('.env')

# Add your key and endpoint
key = os.getenv("AZURE_TEXT_TRANSLATION_KEY")
endpoint = os.getenv("AZURE_TEXT_TRANSLATION_ENDPOINT")

# location, also known as region.
# required if you're using a multi-service or regional (not global) resource. It can be found in the Azure portal on the Keys and Endpoint page.
location = "eastus"

path = '/translate'
constructed_url = endpoint + path

params = {
    'api-version': '3.0',
    'from': 'th',
    'to': ['en']
}

headers = {
    'Ocp-Apim-Subscription-Key': key,
    # location required if you're using a multi-service or regional (not global) resource.
    'Ocp-Apim-Subscription-Region': location,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}

# You can pass more than one object in body.
body = [{
    'text': 'เทคโนโลยีในปัจจุบนี้ค่อนข้างเปลี่ยนไปอย่างรวดเร็ว การธนาคารจึงต้องปรับเปลี่ยน ปรับปรุงให้เท่ากันยุดสมัยอยู่เสมอ จึงเป็นเรื่องท้าทายสำหรับพนักงานในองค์กร ที่ต้องพัฒนาตัวเองให้ไว ให้ดี เท่ากันเทคโนโลยี เพื่อนำมาใช้ในงานให้เกิดประโยชน์สูงสุดแก่ธนาคารและลูกค้า ในฐานะพนักงานยอมรับว่ามีความกดดันและความเครียดค่อนข้างสูงแต่จะพยายามอย่างสุดความสามารถค่ะ'
}]

request = requests.post(constructed_url, params=params, headers=headers, json=body)
response = request.json()

print(json.dumps(response, sort_keys=True, ensure_ascii=False, indent=4, separators=(',', ': ')))