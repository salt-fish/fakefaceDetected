import requests
import argparse

PyTorch_REST_API_URL = 'http://127.0.0.1:5000/upload_photo'

def predict_result(image_path):
	image = open(image_path, 'rb').read()
	# print(image)
	payload = {'photo': (image,'image/png')}

	r = requests.post('http://127.0.0.1:5000/upload', files=payload)
	print(r)
	# if r['success']:
	# 	for (i, result) in enumerate(r['predictions']):
	# 		print('{}.{}:{:.4f'.format(i+1, result['label'], result['probability']))
	# else:
	# 	print('request failed')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Classification demo')
	parser.add_argument('--file', type=str, help='test image file')

	args = parser.parse_args()
	predict_result(args.file)