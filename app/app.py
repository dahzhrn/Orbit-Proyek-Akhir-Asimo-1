# =[Modules dan Packages]========================

from flask import Flask, render_template, request, jsonify
from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import glob, os, random, zipfile
import tensorflow as tf
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import array_to_img
from tensorflow.keras.layers import Dense , Flatten , GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.mobilenet import MobileNet
from PIL import Image
from fungsi import make_model

# =[Variabel Global]=============================

app = Flask(__name__, static_url_path='/static')

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS']  = ['.jpg','.JPG','.jpeg', '.JPEG', '.png', '.PNG']
app.config['UPLOAD_PATH']        = './static/images/uploads/'

model = None

NUM_CLASSES = 7
pakaianadat_classes = ["Pakaian Adat Bali", "Pakaian Adat Jawa Tengah", 
		       			"Pakaian Adat Kalimantan Barat", "Pakaian Adat Papua",
						"Pakaian Adat Riau", "Pakaian Adat Sulawesi Selatan",
						"Pakaian Adat Sumatera Utara"]

suku_classes = ["Suku Bali", "Suku Jawa", "Suku Dayak", "Suku Asmat",
				"Suku Melayu", "Suku Bugis", "Suku Batak"]

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]
@app.route("/")
def beranda():
	return render_template('index.html')

# [Routing untuk API]	
@app.route("/api/deteksi",methods=['POST'])
def apiDeteksi():
	# Set nilai default untuk hasil prediksi dan gambar yang diprediksi
	hasil_prediksi  = '(none)'
	gambar_prediksi = '(none)'
	hasil_suku = '(none)'

	# Get File Gambar yg telah diupload pengguna
	uploaded_file = request.files['file']
	filename      = secure_filename(uploaded_file.filename)
	
	# Periksa apakah ada file yg dipilih untuk diupload
	if filename != '':
	
		# Set/mendapatkan extension dan path dari file yg diupload
		file_ext        = os.path.splitext(filename)[1]
		gambar_prediksi = '/static/images/uploads/' + filename
		
		# Periksa apakah extension file yg diupload sesuai (jpg)
		if file_ext in app.config['UPLOAD_EXTENSIONS']:
			
			# Simpan Gambar
			uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
			
			# Memuat Gambar
			test_image         = Image.open('.' + gambar_prediksi)
			
			# Mengubah Ukuran Gambar
			test_image_resized = test_image.resize((224, 224))
			
			# Konversi Gambar ke Array
			image_array        = np.array(test_image_resized)
			test_image_x       = (image_array / 255) - 0.5
			test_image_x       = np.array([image_array])
			
			# Prediksi Gambar
			y_pred_test_single         = model.predict(test_image_x)
			y_pred_test_classes_single = np.argmax(y_pred_test_single, axis=1)
			
			hasil_prediksi = pakaianadat_classes[y_pred_test_classes_single[0]]
			hasil_suku = suku_classes[y_pred_test_classes_single[0]]
			
			# Return hasil prediksi dengan format JSON
			return jsonify({
				"prediksi": hasil_prediksi,
				"gambar_prediksi" : gambar_prediksi,
				"suku": hasil_suku
			})
		else:
			# Return hasil prediksi dengan format JSON
			gambar_prediksi = '(none)'
			return jsonify({
				"prediksi": hasil_prediksi,
				"gambar_prediksi" : gambar_prediksi,
				"suku": hasil_suku
			})

# =[Main]========================================		

if __name__ == '__main__':
	
	# Load model yang telah ditraining
	model = make_model()
	model.load_weights("model_pakaian_adat_indonesia.h5")

	# Run Flask di lokal
	# app.run(host="localhost", port=5000, debug=True)

	# Run Flask di goole colab
	run_with_ngrok(app)
	app.run()
	
