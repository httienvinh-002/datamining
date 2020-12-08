"""
Sử dụng gói (module) icrawler, là một framwork nhỏ dùng cho việc cào dữ liệu hình ảnh từ website Google Image
Trước tiên thay đổi class GoogleParser của module để phù hợp với sự thay đổi website Google Image
"""
import re
from icrawler import Parser
from bs4 import BeautifulSoup


class GoogleParser(Parser):
    def parse(self, response):
        soup = BeautifulSoup(
            response.content.decode('utf-8', 'ignore'), 'lxml')
        #image_divs = soup.find_all('script')
        image_divs = soup.find_all(name='script')
        for div in image_divs:
            #txt = div.text
            txt = str(div)
            #if not txt.startswith('AF_initDataCallback'):
            if 'AF_initDataCallback' not in txt:
                continue
            if 'ds:0' in txt or 'ds:1' not in txt:
                continue
            #txt = re.sub(r"^AF_initDataCallback\({.*key: 'ds:(\d)'.+data:function\(\){return (.+)}}\);?$",
            #             "\\2", txt, 0, re.DOTALL)
            #meta = json.loads(txt)
            #data = meta[31][0][12][2]
            #uris = [img[1][3][0] for img in data if img[0] == 1]

            uris = re.findall(r'http.*?\.(?:jpg|png|bmp)', txt)
            return [{'file_url': uri} for uri in uris]