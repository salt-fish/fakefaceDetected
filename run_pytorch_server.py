import io
import os
import json
import datetime
import time
import random
import flask
from flask_cors import CORS
import urllib.request
import cv2
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import face_recognition
import argparse
import model
from io import StringIO, BytesIO
from PIL import Image as pil_image
from glob import glob
# from util import get_config, Logger
# from data import imdb, dataset
from network.classifier import *
import dlib
import torchvision.models as models

UPLOAD_FOLDER = {
	'deepfake': 'upload/deepfake',
	'f2f': 'upload/f2f',
	'gan': 'upload/gan',
	'video_origin': 'static/video/origin',
	'video_result': 'static/video/origin',
	'person': 'upload/person',
}
app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'jpeg', 'JPEG'])

capsule = None
meso4_df = None
meso4_gan = None
ff = None
vgg_ext = None

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
opt = parser.parse_args()

def load_model():
	global capsule
	global SeparableNet
	global vgg_ext
	global detector

	model_capsule = os.path.join('models', 'capsule_99.pt')
	model_separableNet = os.path.join('models', 'SeparableNet.pkl')

	vgg_ext = model.VggExtractor()
	capsule = model.CapsuleNet(opt.gpu_id)
	capsule.load_state_dict(torch.load(model_capsule))
	capsule.eval()

	model_state_dict1 = torch.load(model_separableNet).state_dict()
	separableNet = SeparableNet()
	separableNet.load_state_dict(model_state_dict1)

	detector = dlib.get_frontal_face_detector()

	if opt.gpu_id >= 0:
		vgg_ext.cuda(opt.gpu_id)
		capsule.cuda(opt.gpu_id)
		separableNet.cuda()


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def create_uuid():  # 生成唯一的图片的名称字符串，防止图片显示时的重名问题
	nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 生成当前时间
	randomNum = random.randint(0, 100)  # 生成的随机整数n，其中0<=n<=100
	if randomNum <= 10:
		randomNum = str(0) + str(randomNum)
	uniqueNum = str(nowTime) + str(randomNum)
	return uniqueNum


@app.route('/')
@app.route('/index')
def index():
	return flask.render_template("index.html")


# @app.route('/upload', methods=['POST'], strict_slashes=False)
# def upload():
# 	file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER']['person'])
# 	if not os.path.exists(file_dir):
# 		os.makedirs(file_dir)
# 	f = flask.request.files['photo']
# 	if f and allowed_file(f.filename):
# 		# print(f.filename)
# 		ext = f.filename.rsplit('.', 1)[1]
# 		new_filename = create_uuid() + '.' + ext
# 		# print(os.path.join(file_dir, new_filename))
# 		f.save(os.path.join(file_dir, new_filename))
#
# 		result = recognition(new_filename)
# 		# print(result)
# 		return flask.jsonify(result)
# 		# return flask.jsonify({"success": True, "msg": "上传成功"})
# 	else:
# 		return flask.jsonify({"error": 1001, "msg": "上传失败"})


@app.route('/upload_gan', methods=['POST'], strict_slashes=False)
def upload_gan():

	file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER']['gan'])
	if not os.path.exists(file_dir):
		os.makedirs(file_dir)
	f = flask.request.files['photo']
	# print(f)
	if f and allowed_file(f.filename):
		print(f.filename)
		ext = f.filename.rsplit('.', 1)[1]
		new_filename = create_uuid() + '.' + ext
		f.save(os.path.join(file_dir, new_filename))

		result = predict_gan(new_filename)
		print(result)
		return flask.jsonify(result)
		# return flask.jsonify({"success": True, "msg": "上传成功"})
	else:
		return flask.jsonify({"success": False, "msg": "上传失败"})


@app.route('/upload_f2f', methods=['POST'], strict_slashes=False)
def upload_f2f():

	file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER']['f2f'])
	if not os.path.exists(file_dir):
		os.makedirs(file_dir)
	f = flask.request.files['photo']
	# print(f)
	if f and allowed_file(f.filename):
		print(f.filename)
		ext = f.filename.rsplit('.', 1)[1]
		new_filename = create_uuid() + '.' + ext
		f.save(os.path.join(file_dir, new_filename))

		result = predict_f2f(new_filename)
		print(result)
		return flask.jsonify(result)
		# return flask.jsonify({"success": True, "msg": "上传成功"})
	else:
		return flask.jsonify({"success": False, "msg": "上传失败"})


@app.route('/upload_video', methods=['POST'], strict_slashes=False)
def upload_video():

	file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER']['video_origin'])
	if not os.path.exists(file_dir):
		os.makedirs(file_dir)
	f = flask.request.files['video']
	# print(f)
	if f:
		print(f.filename)
		ext = f.filename.rsplit('.', 1)[1]
		new_filename = create_uuid() + '.' + ext
		f.save(os.path.join(file_dir, new_filename))

		# result = predict_video(new_filename)
		origin_path = os.path.join(file_dir, new_filename)
		result = {'succsee': True}
		# print(result)
		return flask.jsonify(result)
		# return flask.jsonify({"success": True, "msg": "上传成功"})
	else:
		return flask.jsonify({"success": False, "msg": "上传失败"})


@app.route('/detection', methods=['POST'], strict_slashes=False)
def detection():
	f = flask.request.files['photo']
	# print(f.read())
	if f and allowed_file(f.filename):
		# print(f.filename)
		img_data = np.asarray(pil_image.open(BytesIO(f.read())))
		# detector = dlib.get_frontal_face_detector()
		faces = detector(img_data, 1)
		print(len(faces))
		if len(faces) > 0:
			return flask.jsonify({"success": True, "msg": "检测到人脸"})
		else:
			return flask.jsonify({"success": False, "msg": "未检测到人脸"})
	else:
		return flask.jsonify({"success": False, "msg": "上传失败"})


@app.route('/identify_meso', methods=['POST'], strict_slashes=False)
def identify_meso():
	f = flask.request.files['photo']
	data = {"success": False}
	# print(f.read())
	stime = time.clock()
	if f and allowed_file(f.filename):
		# print(f.filename)
		img_data = pil_image.open(BytesIO(f.read()))
		# 人脸检测部分
		detect_data = np.asarray(img_data)
		# detector = dlib.get_frontal_face_detector()
		faces = detector(detect_data, 1)
		if len(faces) <= 0:
			return flask.jsonify({"success": False, "msg": "没有人脸"})

		# 人脸鉴别部分
		# img_data = pil_image.open(BytesIO(f.read()))
		transform_fwd = transforms.Compose([
			transforms.Resize((256, 256)),
			transforms.ToTensor(),
			transforms.Normalize([0.5] * 3, [0.5] * 3)
		])

		img_data = transform_fwd(img_data)
		img_data = img_data.unsqueeze(0).cuda()

		output, features = separableNet(img_data)
		# print(output)
		# output [real probability, fake probability]
		output = nn.Softmax(dim=1)(output)
		# tmp = np.zeros((8, 8))
		# for i in range(features.shape[1]):
		# 	tmp += features[0, i, :, :]
		# print(output)

		# 给出判断结果
		# 标签
		# _, prediction = torch.max(output, 1)
		# prediction = float(prediction.cpu().numpy())

		# 为真的概率
		pred = output.data.cpu().numpy()
		# print(pred)
		data['prediction'] = round(float(pred[0][0]), 2)

		# Indicate that the request was a success
		data["success"] = True
		# print(data)
		etime = time.clock()
		# data["feature"] = tmp
		data["time"] = etime - stime
		return flask.jsonify(data)
	else:
		return flask.jsonify({"success": False, "msg": "上传失败"})


@app.route('/identify_capsule', methods=['POST'], strict_slashes=False)
def identify_capsule():
	f = flask.request.files['photo']
	data = {"success": False}
	# print(f.read())
	stime = time.clock()
	if f and allowed_file(f.filename):
		# print(f.filename)
		# 人脸检测部分
		img_data = pil_image.open(BytesIO(f.read()))
		detect_data = np.asarray(img_data)
		# detector = dlib.get_frontal_face_detector()
		faces = detector(detect_data, 1)
		if len(faces) <= 0:
			return flask.jsonify({"success": False, "msg": "没有人脸"})

		# 人脸鉴别部分
		# img_data = pil_image.open(BytesIO(f.read()))
		transform_fwd = transforms.Compose([
			transforms.Resize((256, 256)),
			transforms.ToTensor(),
			transforms.Normalize([0.5] * 3, [0.5] * 3)
		])

		img_data = transform_fwd(img_data)
		img_data = img_data.unsqueeze(0)

		input_v = Variable(img_data)
		x = vgg_ext(input_v)
		classes, class_ = capsule(x, random=False)
		output = class_.data.cpu().numpy()
		data['prediction'] = round(float(output[0][0]), 2)

		# Indicate that the request was a success
		data["success"] = True
		etime = time.clock()
		data["time"] = etime - stime
		return flask.jsonify(data)
	else:
		return flask.jsonify({"success": False, "msg": "上传失败"})


@app.route('/upload_identify', methods=['POST'], strict_slashes=False)
def upload_identify():

	file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER']['deepfake'])
	if not os.path.exists(file_dir):
		os.makedirs(file_dir)
	if flask.request.form.get("dataType") == 'file':
		f = flask.request.files['photo']
		# print(f)
		if f and allowed_file(f.filename):
			# print(f.filename)
			ext = f.filename.rsplit('.', 1)[1]
			new_filename = create_uuid() + '.' + ext
			f.save(os.path.join(file_dir, new_filename))
	elif flask.request.form.get("dataType") == 'link':
		new_filename = create_uuid() + '.jpg'
		urllib.request.urlretrieve(flask.request.form.get("url"), os.path.join(file_dir, new_filename))
	else:
		return flask.jsonify({"success": False, "msg": "上传失败"})
	result = predict_df(new_filename)
	# result = predict_f2f(new_filename)
	# print(result)
	return flask.jsonify(result)



def recognition(filename):
	# Initialize the data dictionary that will be returned from the view
	data = {"success": False}
	upload_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER']['person'])
	db_dir = os.path.join(basedir, 'face_db')
	upload_path = os.path.join(upload_dir, filename)
	# if flask.request.method == 'GET':
	if filename is None:
		print(1)
		pass
	elif os.path.splitext(filename)[1].lower() not in ['.jpg', '.png', '.jpeg']:
		print(2)
		pass
	else:
		for people in os.listdir(db_dir):
			path = os.path.join(db_dir, people)
			known_image = face_recognition.load_image_file(path)
			unknown_image = face_recognition.load_image_file(upload_path)

			biden_encoding = face_recognition.face_encodings(known_image)[0]
			unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

			results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
			if results[0] == True:
				data['person'] = people
				data['success'] = True
				return data

		data['person'] = 'unknown'

		# Indicate that the request was a success
		data["success"] = True
	return data

def predict_df(filename):
	# Initialize the data dictionary that will be returned from the view
	data = {"success": False}
	file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER']['deepfake'])
	if filename is None:
		print(1)
		pass
	elif os.path.splitext(filename)[1].lower() not in ['.jpg', '.png', '.jpeg']:
		print(2)
		pass
	else:
		img_path = os.path.join(file_dir, filename)
		# img_data = cv2.imread(img_path)
		img_data = pil_image.open(img_path)

		# 人脸检测部分
		# detect_data = np.asarray(img_data)
		# faces = detector(detect_data, 1)
		# if len(faces) <= 0:
		# 	data = {"msg": "没有人脸"}
		# 	return data

		stime = time.clock()
		# 人脸鉴别部分
		# img_data = pil_image.open(BytesIO(f.read()))
		transform_fwd = transforms.Compose([
			transforms.Resize((256, 256)),
			transforms.ToTensor(),
			transforms.Normalize([0.5] * 3, [0.5] * 3)
		])

		img_data = transform_fwd(img_data)
		img_data = img_data.unsqueeze(0).cuda()

		result = 0
		for i in range(50):
			output, features = separableNet(img_data)
			# output [real probability, fake probability]
			output = nn.Softmax(dim=1)(output)
			# tmp = np.zeros((8, 8))
			# for i in range(features.shape[1]):
			# 	tmp += features[0, i, :, :]
			# print(output)

			# 给出判断结果
			# 标签
			# _, prediction = torch.max(output, 1)
			# prediction = float(prediction.cpu().numpy())

			# 为真的概率
			pred = output.data.cpu().numpy()
			pred = round(float(pred[0][0]), 2)
			# print("##################")
			# print(pred)
			# print(pred)
			pred = 0.1 if pred<0.1 else pred
			pred = 0.9 if pred>0.9 else pred

			result += pred

		# data['prediction'] = round(float(pred[0][0]), 2)
		data['prediction'] = round(result/50+0.1, 2)

		# Indicate that the request was a success
		data["success"] = True
		# print(data)
		etime = time.clock()
		# data["feature"] = tmp
		data["time"] = round(etime - stime, 2)
		return data
	data = {"msg": "检测失败"}
	return data


# @app.route("/predict/<string:filename>", methods=["GET"])
def predict_gan(filename):
	# Initialize the data dictionary that will be returned from the view
	data = {"success": False}
	file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER']['gan'])
	# if flask.request.method == 'GET':
	if filename is None:
		print(1)
		pass
	elif os.path.splitext(filename)[1].lower() not in ['.jpg', '.png', '.jpeg']:
		print(2)
		pass
	else:
		img_path = os.path.join(file_dir, filename)
		img_data = cv2.imread(img_path)

		transform_fwd = transforms.Compose([
			transforms.Resize(128),
			transforms.CenterCrop(128),
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		])

		img_data = transform_fwd(pil_image.fromarray(img_data))
		img_data = img_data.unsqueeze(0)
		if opt.gpu_id >= 0:
			img_data = img_data.cuda(opt.gpu_id)
		input_v = Variable(img_data)
		x = vgg_ext(input_v)
		classes, class_ = capsule(x, random=False)
		output = class_.data.cpu().numpy()
		# print(class_)
		# transform_fwd = transforms.Compose([
		# 	transforms.Resize((256, 256)),
		# 	transforms.ToTensor(),
		# 	transforms.Normalize([0.5] * 3, [0.5] * 3)
		# ])
		#
		# img_data = transform_fwd(pil_image.fromarray(img_data))
		# img_data = img_data.unsqueeze(0)
		# if opt.gpu_id >= 0:
		# 	img_data = img_data.cuda(opt.gpu_id)
		#
		# output = meso4_gan(img_data)
		# output = nn.Softmax(dim=1)(output)
		#
		# pred = output.data.cpu().numpy()
		# print(pred)
		# # 为假的概
		# data['prediction'] = float(pred[0][0])
		# 为假的概率
		data['prediction'] = round(float(output[0][0]), 2)

		# Indicate that the request was a success
		data["success"] = True

	return data


def predict_f2f(filename):
	# Initialize the data dictionary that will be returned from the view
	data = {"success": False}
	file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER']['deepfake'])
	# if flask.request.method == 'GET':
	if filename is None:
		print(1)
		pass
	elif os.path.splitext(filename)[1].lower() not in ['.jpg', '.png', '.jpeg']:
		print(2)
		pass
	else:
		img_path = os.path.join(file_dir, filename)
		img_data = cv2.imread(img_path)

		transform_fwd = transforms.Compose([
			transforms.Resize(128),
			transforms.CenterCrop(128),
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		])

		img_data = transform_fwd(pil_image.fromarray(img_data))
		img_data = img_data.unsqueeze(0)
		if opt.gpu_id >= 0:
			img_data = img_data.cuda(opt.gpu_id)
		input_v = Variable(img_data)
		x = vgg_ext(input_v)
		classes, class_ = capsule(x, random=False)
		output = class_.data.cpu().numpy()
		# print(class_)
		# 为假的概率
		data['prediction'] = round(float(output[0][0]), 2)

		# Indicate that the request was a success
		data["success"] = True

	return data


def predict_video(filename):
	# Initialize the data dictionary that will be returned from the view
	data = {"success": False}
	file_origin_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER']['video_origin'])
	file_result_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER']['video_result'])
	# if flask.request.method == 'GET':
	if filename is None:
		print(1)
		pass
	elif os.path.splitext(filename)[1].lower() not in ['.mp4']:
		print(2)
		pass
	else:
		############
		origin_video_path = os.path.join(file_origin_dir, filename)
		result_video_path = os.path.join(file_result_dir, filename)

		img_path = os.path.join(file_origin_dir, filename)
		img_data = cv2.imread(img_path)

		transform_fwd = transforms.Compose([
			transforms.Resize(128),
			transforms.CenterCrop(128),
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		])

		img_data = transform_fwd(pil_image.fromarray(img_data))
		img_data = img_data.unsqueeze(0)
		if opt.gpu_id >= 0:
			img_data = img_data.cuda(opt.gpu_id)
		input_v = Variable(img_data)
		x = vgg_ext(input_v)
		classes, class_ = capsule(x, random=False)
		output = class_.data.cpu().numpy()
		# print(class_)
		#
		data['origin'] = origin_video_path
		data['result'] = result_video_path

		# Indicate that the request was a success
		data["success"] = True

	return data


if __name__ == '__main__':
	print("Loading PyTorch model and Flask starting server...")
	print("Please wait until server has fully started")
	# init_config()
	load_model()
	# a=predict_gan('333333333333.jpg')
	# print(a)
	CORS(app, supports_credentials=True)
	app.run(host='10.10.3.8', port=5000)
