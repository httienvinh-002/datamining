import os
import cv2
from flask import Flask,  render_template, json, request
from flask import flash, redirect, url_for
from werkzeug.utils import secure_filename

import utils


UPLOAD_FOLDER = 'C:/Users/Admin/PycharmProjects/data_mining/static'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        utils.reload()
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        image = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if image.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        utils.face_recog('static/'+filename)
        #utils.recog_LBPH('static/' + filename)
        return render_template("recognition.html", image_upload=filename)
    return render_template("recognition.html")


'''@app.route("/")
def main():
    return render_template('view.html')'''


@app.context_processor
def override_url_for():
    """
    Generate a new token on every request to prevent the browser from
    caching static files.
    """
    return dict(url_for=dated_url_for)


def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


if __name__ == "__main__":
    app.run(debug=True)