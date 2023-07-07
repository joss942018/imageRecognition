from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import re
from dateutil.parser import parse
from spellchecker import SpellChecker
from datetime import datetime
app = Flask(__name__)

spell = SpellChecker(language='es')

dateFilterPatterns = [
  {
    "regex": r"\b\d{4}-\d{2}-\d{2}\b",
    "type": "date"
  },
  {
    "regex": r"\d{2}/\d{2}/\d{2}",
    "type": "dateUnformatedComplete"
  },
  {
    "regex": r"\d{2}/\d{2}/\d{1}",
    "type": "dateUnformatedSemi"
  },
   
]

invoiceFilterPatterns = [    

  {
    "regex": r'\d{5}-\d{8}',
    "type": "normal"
  },
  {
    "regex": r"P\.V\. N' \d{5}",
    "type": "separateCode",
    "secondariesFactors": [
        {
            "regex": r'Hora \d+',
            "filter": "Hora "
        }
        
    ],
    "filter": "P.V. N\'"
  },
  {
    "regex": r'PV\. N” \d{5}',
    "type": "separateCode",
    "secondariesFactors": [
        {
            "regex": r'Nro\. T\. \d{8}',
            "filter": "Nro. T. "
        }
        
    ],
    "filter": "PV. N”"
  } ,
  {
    "regex": r'PV\. We \d{3}',
    "type": "separateCode",
    "secondariesFactors": [
        {
            "regex": r'\d+ Nro\. T\. \d{8}',
            "filter": "Nro. T. "
        }
        
    ],
    "filter": "PV. We "
  }, 
  {
    "regex": r'PV\. We \d{3}',
    "type": "separateCode",
    "secondariesFactors": [
        {
            "regex": r'\d+ Nro\. T\. \d{8}',
            "filter": "Nro. T. "
        }
        
    ],
    "filter": "PV. We "
  },
  {
    "regex": r'\d{4}-\d{8}',
    "type": "normal"
  }
]

companyCodeFilterPatterns = [
  {
    "regex": r'\b\d{11}\b',
    "type": "company"
  },
  {
    "regex": r"\d{2}-\d{8}-\d",
    "type": "company"
  },
   
]

amountsFilterPatterns2 = [
    {
        "regex": r"TOTAL:? (\d+\.\d{2})\b",
        "type": "normal"
    },
    {
        "regex": r"(?<!,)-?\d+\.\d{2}",
        "type": "amount"
    },
    {
        "regex": r'\b\d{1,3}(?:,\d{3})*\.\d{2}\b',
        "type": "amountWithComaMillar"
    }
]


amountsFilterPatterns = [
  
   {
        "regex": r'(?<!\()\-?\d+(?:,|\.)\d{2}\b',
        "type": "amount"
    },
    {
        "regex": r'\b\d{1,}(?:,\d{1,})?\.\d{2}\b',
        "type": "amountWithComaMillar"
    }
   
]

@app.route('/processInvoice', methods=['POST'])
def process_invoice():
    if not request.json or 'token' not in request.json or 'base64InvoiceImage' not in request.json:
        abort(400)
    
    token = request.json['token']
    base64_invoice_image = request.json['base64InvoiceImage']

    try:
        # Convert base64 to image
        img_bytes = base64.b64decode(base64_invoice_image)

        # Convert bytes to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)

        # Load the image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert the image to grayscale
        if np.mean(image) < 120:
            image = increase_brightness(image)
        if is_low_quality(image):
            image = improve_quality(image)
        image = remove_shadows(image)
        image = deskew_image(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        binary = cv2.erode(binary, kernel, iterations=1)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        if variance < 10:  
            abort(400, 'The image is too blurry')
        
        extracted_text = pytesseract.image_to_string(image , lang='spa')
        extracted_text = normalize_dates(extracted_text)
        extracted_text = remove_noise(extracted_text)            
        codigo_factura = "N/D"
        local_factura = "N/D"
        nombre_cliente = "N/D"
        fecha = "N/D"

        fecha = filterTesseractOutput(extracted_text, dateFilterPatterns)
        codigoFactura = filterTesseractOutput(extracted_text, invoiceFilterPatterns)
        localFactura  = filterTesseractOutput(extracted_text, companyCodeFilterPatterns)
        totalFactura =  getAmount(extracted_text, amountsFilterPatterns)

        if fecha != "N/D" and codigoFactura != "N/D" and localFactura != "N/D" and totalFactura != "N/D":
            complete = True
        datos = {
            "cod_factura": codigoFactura,
            "local_factura": localFactura,
            "nombre_cliente": 'Consumidor Final',
            "fecha": fecha,
            "total" : totalFactura,
            "complete" : complete
        }

        # Here you can parse the extracted text to select the specific data you need
        return datos, 200
    except Exception as e:
        print("Exception occurred during image processing: ", str(e))
        abort(401, description='Exception occurred during image processing')

def filterTesseractOutput(text, patterns):
    encouteredText = "N/D"
    for pattern in patterns:
        
        if pattern['type'] == "date" or pattern['type'] == "dateUnformatedComplete" or pattern['type'] == "dateUnformatedSemi":              
            matches = re.findall(pattern['regex'], text) 
            if matches:
                if pattern['type'] == "dateUnformatedComplete":
                    dates = [datetime.strptime(match, "%d/%m/%y").strftime('%Y-%m-%d') for match in matches]
                    dates = [datetime.strptime(match, '%Y-%m-%d') for match in matches]
                elif pattern['type'] == "date":
                    dates = [datetime.strptime(match, '%Y-%m-%d') for match in matches]
                # get the current year
                current_year = datetime.now().year
                # filter dates between 2022 and current year
                filtered_dates = [date for date in dates if 2022 <= date.year <= current_year]
                if filtered_dates:
                    encouteredText = max(filtered_dates)
                    break                   
        elif pattern['type'] == "normal" or pattern['type'] == "company" : 
            match = re.search(pattern['regex'], text)            
            if match:
                encouteredText = match.group()
                if pattern['type'] == "company":
                    encouteredText = encouteredText.replace('-', '')
                break     
        elif pattern['type'] == "separateCode" : 
            text = text.replace('\n', ' ')
            text = text.replace('  ', ' ')
            match = re.search(pattern['regex'], text)                    
            if match:
                for secondaryFactors in pattern['secondariesFactors']:                    
                    secondaryMatch = re.search(secondaryFactors['regex'], text)
                    if secondaryMatch:   
                        encouteredText = match.group().replace(pattern['filter'] , '')
                        encouteredText += secondaryMatch.group().replace(secondaryFactors['filter'] , '-')
                        break
                   
    return encouteredText

    


def getAmount(text, patterns):
    maxList = []
    isNegative = False
    encouteredText = "N/D"
    whenIsNegativeList = []
    for pattern in patterns:
        if pattern['type'] == "amountWithComaMillar":
            matches = re.findall(pattern['regex'], text)
            if matches:
                matches_float = [float(match.replace(',', '')) for match in matches]
                positive_values = [value for value in matches_float if value >= 0]
                negative_values = [value for value in matches_float if value < 0]
                if not positive_values:
                    positive_values = [-1e-10]                 
                if negative_values:
                    isNegative = True
                    encouteredText = [format(float(i), '.2f') for i in positive_values]                    
                    whenIsNegativeList.extend(positive_values)
                else:
                    encouteredText = format(max(positive_values), '.2f')
                    maxList.append(encouteredText)
        elif pattern['type'] == "amount":
            matches = re.findall(pattern['regex'], text)      
            if matches:
                matches_float = [float(match.replace(',', '.')) for match in matches]
                positive_values = [value for value in matches_float if value > 0]
                negative_values = [value for value in matches_float if value < 0]
                if not positive_values:
                    positive_values = [-1e-10]              
                if negative_values:
                    isNegative = True                    
                    encouteredText = [format(float(i), '.2f') for i in positive_values]                    
                    whenIsNegativeList.extend(positive_values)
                else:                                         
                    encouteredText = format(max(positive_values), '.2f')
                    maxList.append(encouteredText) 
    if isNegative:
        encouteredText = max(["{:.2f}".format(float(item)) for item in whenIsNegativeList])
        encouteredText = sorted(set(whenIsNegativeList), reverse=True)

    elif len(maxList) > 0 :        
        encouteredText = max([float(item) for item in maxList])
        encouteredText = ["{:.2f}".format(encouteredText)]
        print(type(encouteredText))
        print(encouteredText)
    return encouteredText

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def increase_brightness(image, value=65):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image

def improve_quality(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return image

def remove_shadows(image):
    rgb_planes = cv2.split(image)

    result_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)

    result = cv2.merge(result_planes)
    return result

def is_low_quality(image, threshold=100):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()
    return variance < threshold   

def detect_skew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    return angle

def deskew_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

    #PostProcessin!
def clean_text(text):
    text = re.sub('[^A-Za-z0-9\s]+', '', text)
    text = text.lower()
    return text

def normalize_dates(text):
    dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b', text)    
    for date in dates:
        try:
            normalized_date = parse(date).strftime("%Y-%m-%d")
            text = text.replace(date, normalized_date)
        except ValueError:
            continue     
    return text

def remove_noise(text):
    noise = ["@", "#", "$", "%"]    
    for n in noise:
        text = text.replace(n, "")    
    return text

def spell_check(text):
    words = text.split()
    for word in words:
        correct_word = spell.correction(word)
        if correct_word is not None and correct_word != word: 
            text = re.sub(r'\b{}\b'.format(re.escape(word)), correct_word, text)  
    return text

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
