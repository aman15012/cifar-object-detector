import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory
import detector
import cv2
import numpy as np
from helper import pyramid,sliding_window 
import imutils
import json
import time

UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		# if user does not select file, browser also
		# submit a empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = "image." + file.filename.split(".")[1]
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			millis = int(round(time.time() * 1000))
			return redirect(url_for('results',
									filename=filename,millis=millis))
	return render_template('upload.html')



@app.route('/results/<filename>/<millis>')
def results(filename,millis):
	
	model = detector.cifar10vgg()

	# load the image and define the window width and height
	image = cv2.imread(filename)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
	W = image.shape[1]
	(winW, winH) = (np.max([W/10,32]), np.max([W/10,32]))

	dictionary = {}
	
	# initialize
	for i in range(10):
		dictionary[i] = []

	# loop over the image pyramid
	for resized in pyramid(image, scale=2):

		# loop over the sliding window for each layer of the pyramid
		for (x, y, window) in sliding_window(resized, stepSize=np.max([W/20,8]), windowSize=(winW, winH)):
		
			# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue

			# resizing the image to correct dimensions for the CNN
			clone = resized.copy()
			X = np.asarray(resized[y:y+winH,x:x+winW]).astype(float)
			X = np.asarray([imutils.resize(X, width=32)])

			if(X.shape[1:3]==(32,32)):
 
				predicted_x = model.predict(X)
				index = np.argmax(predicted_x,1)
				value = predicted_x[0][index]
				if(value>=0.99):
					# dictionayr[i] = [max_accuracy(i),sum_accuracies(i),i,(coordinates_of_max_accuracy(i))]
					if(len(dictionary[index[0]]) == 0):
						dictionary[index[0]] = [value,value,index[0],resized,x,y,x+winW,y+winH]
					elif(dictionary[index[0]][1]<value):
						temp  = dictionary[index[0]][0] + value
						dictionary[index[0]] = [temp,value,index[0],resized,x,y,x+winW,y+winH]
					else:
						temp  = dictionary[index[0]][0] + value
						temp_list = dictionary[index[0]]
						temp_list[0] = temp
						dictionary[index[0]] = temp_list

	values = sorted(dictionary.values())

	sum_values = 0.0
	for i in values:
		if(len(i)>0):
			sum_values += i[0]


	a = 0
	b = 0
	c = 0

	filename = [filename.split(".")[0]+"1."+filename.split(".")[1], filename.split(".")[0]+"2."+filename.split(".")[1], filename.split(".")[0]+"3."+filename.split(".")[1]]
	f = [os.path.join("static", filename[i]) for i in range(3)]
	

	dataPoints = []
	
	# top 3 predictions
	if(len(values)>0):
		if(len(values[-1])>0):
			clone1 = values[-1][3].copy()
			a = (detector.classes[values[-1][2]], values[-1][0]/sum_values, values[-1][1])
			cv2.rectangle(clone1, (values[-1][4], values[-1][5]), (values[-1][6], values[-1][7]), (255,0,0), 5)
			clone1 = imutils.resize(clone1, width=W)
			cv2.imwrite(os.path.join("static", filename[0]), cv2.cvtColor(clone1, cv2.COLOR_BGR2RGB))
			dataPoints.append({"label": a[0],"y": a[1][0]*100})


	if(len(values)>1):
		if(len(values[-2])>0):
			clone2 = values[-2][3].copy()
			b = (detector.classes[values[-2][2]], values[-2][0]/sum_values, values[-2][1])
			cv2.rectangle(clone2, (values[-2][4], values[-2][5]), (values[-2][6], values[-2][7]), (0,255,0), 5)
			clone2 = imutils.resize(clone2, width=W)   
			cv2.imwrite(os.path.join("static", filename[1]), cv2.cvtColor(clone2, cv2.COLOR_BGR2RGB))
			dataPoints.append({"label": b[0],"y": b[1][0]*100})

	if(len(values)>2):
		if(len(values[-3])>0):
			clone3 = values[-3][3].copy()
			c = (detector.classes[values[-3][2]], values[-3][0]/sum_values, values[-3][1])
			cv2.rectangle(clone3, (values[-3][4], values[-3][5]), (values[-3][6], values[-3][7]), (0,0,255), 5)   
			clone3 = imutils.resize(clone3, width=W)
			cv2.imwrite(os.path.join("static", filename[2]), cv2.cvtColor(clone3, cv2.COLOR_BGR2RGB))
			dataPoints.append({"label": c[0],"y": c[1][0]*100})
	
	dataPoints = json.dumps(dataPoints)
	print(dataPoints)

	print a
	print b
	print c

	cv2.waitKey(0) # wail till a key is pressed
	return render_template('results.html',**locals())

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response

if __name__ == "__main__":
	app.run(host='0.0.0.0')