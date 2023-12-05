from django.http import HttpResponse
from django.shortcuts import render_to_response
import base64
import requests
from django_app.manager import ModelManager


def predict_by_image_filename(image_filename="1.png"):
    IMAGE_PATH = "/root/autodl-tmp/Swin-Transformer-Chinese-LaTex-OCR/swin-transformer-ocr/dataset/dataset1/resized_images/"+image_filename
    print(IMAGE_PATH)
    try:
        print(IMAGE_PATH)
        return ModelManager.instance().predict_png(IMAGE_PATH)
    except:
        return 'something wrong happend'


def search_form(request):
    # start page
    return render_to_response('search_form.html')


def search(request):
    # result page
    request.encoding = 'utf-8'
    print(request.GET)
    if len(request.GET) > 0:
        image_url = request.GET['info']  # “0.png”
        # a = predict_by_image_filename(image_url)
        # predict_string, gif_path = a[0], a[1]
        predict_string = predict_by_image_filename(image_url)
        print(predict_string)
        image_url = "http://localhost:8020/" + image_url.split("/")[-1].split("\\")[-1]
        print(image_url)
        gif_path = image_url
        # gif_path = "http://localhost:8030/" + gif_path.split("/")[-1].split("\\")[-1]
        return render_to_response('search_result.html', {
            'image_url': image_url,
            'gif_path': gif_path,
            'predict_string': predict_string
        })
    else:
        return render_to_response('search_result.html', {
            'image_url': 'incorrect',
            'gif_path': 'Please input right image URL',
            'predict_string': 'Please input right image URL'
        })
