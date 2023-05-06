from flask import Flask, render_template, request
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Reshape, Dense, Dropout, UpSampling2D, Conv2D, Flatten, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
import numpy as np
from tqdm import tqdm

app=Flask(__name__)

def generator_model():
    latent_dim = 100
    generator = Sequential()
    generator.add(Input(shape=((latent_dim,)))),
    # 50x50
    generator.add(Dense(50*50*100)),
    generator.add(Reshape((50,50,100))),
    # 100x100
    generator.add(Conv2DTranspose(64,kernel_size=4, strides=2, padding="same")),
    generator.add(LeakyReLU(0.2)),
    #200x200
    generator.add(Conv2DTranspose(128,kernel_size=4, strides=2, padding="same")),
    generator.add(LeakyReLU(0.2)),
    # 400x400
    generator.add(Conv2DTranspose(64,kernel_size=4, strides=2, padding="same")),
    generator.add(LeakyReLU(0.2)),
    generator.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))
    # 800 x 800
    generator.add(Conv2DTranspose(64,kernel_size=4, strides=2, padding="same")),
    generator.add(LeakyReLU(0.2)),
    generator.add(Conv2D(3, kernel_size=(5, 5), padding='same', activation='tanh'))
    
    generator.compile(loss='binary_crossentropy', optimizer=opt)
    return generator       
        
    
def discriminator_model():
    discriminator = Sequential()
    # discriminator.add(Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', input_shape=(400,400,1)))
    discriminator.add(Input(shape=((800,800,3)))),
    discriminator.add(Conv2D(64, kernel_size=(5, 5), padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=opt)
    return discriminator


def generator_containing_discriminator(g, d):
    d.trainable = False
    ganInput = Input(shape=(100,))
    x = g(ganInput)
    ganOutput = d(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    gan.compile(loss='binary_crossentropy', optimizer=opt)
    return gan

def train(BATCH_SIZE, X_train):
    d = discriminator_model()
    print("#### discriminator ######")
    d.summary()
    g = generator_model()
    print("#### generator ######")
    g.summary()
    d_on_g = generator_containing_discriminator(g, d)
    d.trainable = True
    for epoch in tqdm(range(6)):
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            noise = np.random.uniform(0, 1, size=(BATCH_SIZE, 100))
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            # print("Image Batch",image_batch.shape)
            # print("Generated Image",generated_images.shape)

            X = np.concatenate((image_batch, generated_images))
            y = np.array([1] * BATCH_SIZE + [0] * BATCH_SIZE)
            d_loss = d.train_on_batch(X, y)
            noise = np.random.uniform(0, 1, (BATCH_SIZE, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, np.array([1] * BATCH_SIZE))
            d.trainable = True
        g.save_weights('assets/generator', True)
        d.save_weights('assets/discriminator', True)
    return d, g

def generate(BATCH_SIZE):
    g = generator_model()
    g.load_weights('/kaggle/input/assets/generator')
    noise = np.random.uniform(0, 1, (BATCH_SIZE, 100))
    generated_images = g.predict(noise)
    return generated_images
def sum_of_residual(y_true, y_pred):
    return tensorflow.reduce_sum(abs(y_true - y_pred))
def feature_extractor():
    d = discriminator_model()
    d.load_weights('/kaggle/input/assets/discriminator') 
    intermidiate_model = Model(inputs=d.layers[0].input, outputs=d.layers[-5].output)
    intermidiate_model.compile(loss='binary_crossentropy', optimizer=opt)
    return intermidiate_model
def anomaly_detector():
    g = generator_model()
    g.load_weights('/kaggle/input/assets/generator')
    g.trainable = False
    intermidiate_model = feature_extractor()
    intermidiate_model.trainable = False
    
    aInput = Input(shape=(100,))
    gInput = Dense((100))(aInput)
    G_out = g(gInput)
    D_out= intermidiate_model(G_out)    
    model = Model(inputs=aInput, outputs=[G_out, D_out])
    model.compile(loss=sum_of_residual, loss_weights= [0.9, 0.1], optimizer=opt)
    return model
def compute_anomaly_score(model, x):    
    z = np.random.uniform(0, 1, size=(1, 100))
    intermidiate_model = feature_extractor()
    d_x = intermidiate_model.predict(x)
    loss = model.fit(z, [x, d_x], epochs=500, verbose=0)
    similar_data, _ = model.predict(z)
    return loss.history['loss'][-1], similar_data

def mse(img1, img2):
    h, w = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    mse = err/(float(h*w))
    return mse, diff


# max_length = 32
# tokenizer = load(open("tokenizer.p","rb"))
# model = load_model('model_9.h5')
# xception_model = Xception(include_top=False, pooling="avg")

# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=True, help="Image Path")
# args = vars(ap.parse_args())
# img_path = args['image']

def extract_features(filename, model):
        try:
            image = Image.open(filename)
            
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


#path = 'Flicker8k_Dataset/111537222_07e56d5a30.jpg'


@app.route('/', methods=['GET'])
def about():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():

    imagefile=request.files["imagefile"]
    if imagefile.filename == '':
        result = 'Please select the Image'
        return render_template('index.html', prediction=result)
    imagepath='./images/'+imagefile.filename
    imagefile.save(imagepath)
    gene = './AnoGAN/Dataset/OK/008.png'
    if imagefile.name == '':
        result = 'Please select the Image'
        return render_template('index.html', prediction=result)
    original = cv2.imread(imagepath,0)
    gen = cv2.imread(gene,0)
    error, diff = mse(original, gen)
    if error>=23.13386:
    # photo = extract_features(imagepath, xception_model)
    # img = Image.open(imagepath)
        result = "I think Defective"
    # description = generate_desc(model, tokenizer, photo, max_length)
    # caption = ""
    else:
    # caption = description[5:len(description)-3]
        result = "I think Good"
    caption_generated='%s' % (result)

    return render_template('index.html', prediction=caption_generated)


if __name__ == '__main__':
    app.run(port=1234, debug=True)

    
# print("\n\n")
# print(description)
# plt.imshow(img)