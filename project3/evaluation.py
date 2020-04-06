import keras
import matplotlib.pyplot as plt
from keras.models import Sequential,Model,model_from_json
import numpy as np



def load_model():
    '''
    This function is used to load model, codes below are based on template.py.
    Please modify this function based on your own codes.
    '''
    with open("generator.json", "r") as json_file:
        md_json = json_file.read()
    t = model_from_json(md_json)
    t.load_weights(PATH+"generator.h5")
    return t

def generate_image(model):
    '''
    Take the model as input and generate one image, codes below are based on template.py.
    Please modify this function based on your own codes.
    '''
    # Set the dimensions of the noise
    noise = np.random.normal(0, 1, (5 * 5, 100))
    generated_images = model.predict(noise)
    
    r, c = 5, 5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
      for j in range(c):
        axs[i,j].imshow(generated_images[cnt, :,:,0], cmap='gray')
        axs[i,j].axis('off')
        cnt += 1
    plt.show()
    
    return generated_images

if __name__ == "__main__":
    model = load_model()
    image = generate_image(model)
