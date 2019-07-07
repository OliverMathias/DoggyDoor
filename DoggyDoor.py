import os
import re
import json
import requests
from flask import *
from bs4 import BeautifulSoup
from werkzeug import secure_filename

app = Flask(__name__)

def getImages(dogname):
    dogname = dogname.replace(" ", "-")
    temp_list = []
    temp_list_features = []
    #Proxy api url
    url = "https://www.akc.org/dog-breeds/"+dogname
    #Gets the api text
    res = requests.get(url)
    html_page = res.content
    soup = BeautifulSoup(html_page, 'html.parser')

    images = soup.findAll('img',class_ = 'media-wrap__image lozad')

    for text in soup.findAll(attrs={'class' : 'attribute-list__description attribute-list__text attribute-list__text--lg mb4 bpm-mb5 pb0 d-block'}):
        Temperament = text.contents[0]

    for text in soup.findAll(attrs={'class' : 'attribute-list__description attribute-list__text'}):
        temp_list_features.append(text.contents[0])

    Height = temp_list_features[1]
    Weight = temp_list_features[2]
    Life = temp_list_features[3]
    Group = str(temp_list_features[4])
    Group_Link = re.findall('"([^"]*)"', Group)
    Group_Link = str(Group_Link[0])
    Group = re.search('/(.*)/', str(Group_Link))
    Group = Group.group(1)
    Group = Group[24:]
    print(Group_Link)

    for image in images:
    #print image source
        temp_list.append(image['data-src'])
    return temp_list, url, Temperament, Height, Weight, Life, Group, Group_Link



@app.route('/')
def upload():
    return render_template("upload.html")

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join('static/uploadedimages', secure_filename(f.filename)))
        path = os.path.join('./static/uploadedimages', secure_filename(f.filename))

        from model import run_app

        likely_breeds, likely_probs, dog_detected, human_detected=run_app(path)

        labels = likely_breeds

        values = likely_probs

        colors = [
            "#38B6FF", "#334DC6", "#6754E2", "#00BFFF",
             "#499DF5", "#CAE1FF", "#2E37FE","#E12F38"]

        #pass urls of pictures of the guess breed of dog
        breed_images, url, Temperament, Height, Weight, Life, Group, Group_Link =getImages(likely_breeds[0])

        return render_template("results.html", imgpath = path, guess = likely_breeds[0], pie_values = values, pie_labels = labels, pie_colors = colors, dog_detected=dog_detected, human_detected=human_detected, url=url, Temperament=Temperament, Height=Height, Weight=Weight, Life=Life, Group=Group, Group_Link=Group_Link, img1=breed_images[0],img2=breed_images[1],img3=breed_images[2],img4=breed_images[3],img5=breed_images[4])


if __name__ == '__main__':
    app.run(debug = True)
