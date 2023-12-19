from numpy import array, argpartition
from PIL import Image
from flask import Flask, request
from getHandwritingModel import getHandwritingModel

app = Flask(__name__)
model = getHandwritingModel()

@app.route("/", methods = ['GET', 'POST'])
def MainPage():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'there is no image in form!'

        formImage = request.files['image']
        with Image.open(formImage) as image:
            image = image.resize([28, 28]).convert('L')
            prediction = model.predict(array([array(image) / 255]).reshape(1, 28, 28, 1))
            print(prediction)
            return str(prediction.argsort()[0][-1])

    return '''
    <h1>Upload new File</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="image">
      <input type="submit">
    </form>
    '''

app.run(host = '0.0.0.0', port = 8080)
