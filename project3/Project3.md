# Project3 Report：

### <center><b>Image Generation with GAN</b></center>

This project is to generate fake image using [Generative adversarial network (GAN)](https://en.wikipedia.org/wiki/Generative_adversarial_network) with classic [MNIST](http://yann.lecun.com/exdb/mnist/) data set in Python3 using [Keras](keras.io) in google [Colab](https://colab.research.google.com/notebooks/welcome.ipynb). 

In this zip folder contains:

* The report, this html file
* The final GAN model: "gan_generator.json" and "gan_generator.h5" 
* The code: evaluation.py file for evaluation, and gan.py for the model
* The model for Fashion MNIST dataset, 'GAN_fashion_generator.json' and 'GAN_fashion_generator.h5'
* The code evaluation_fashion.py for evaluate 

**Note:** this project is based on tensorflow_version 1.x, since Colab did not upgrade to tensorflow 2.X until the end of March.

## Experiments:

I tried several differen kind of GAN. Below I will introduce the structure of my GANs.

### GAN:

First, I tried the original [GAN](https://arxiv.org/abs/1406.2661) by Ian Goodfellow

#### Network Structure:

##### Generator

```python
def build_generator(self):
    model = Sequential()
    model.add(Dense(256, input_dim=self.latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(self.img_shape), activation='tanh'))
    model.add(Reshape(self.img_shape))
    model.summary()

    noise = Input(shape=(self.latent_dim,))
    img = model(noise)

    return Model(noise, img)
```

##### Discriminator

```python
def build_discriminator(self):
    model = Sequential()
    model.add(Flatten(input_shape=self.img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=self.img_shape)
    validity = model(img)

    return Model(img, validity)
```

#### Hyperameters

* Epoch = 50000

* Batch size = 64

* Activation Function = LeakyReLU(alpha = 0.2)
  * tanh for the output layer of generator
  * sigmoid for output layer of discriminator

* Loss Fuction = binary_crossentropy, since it is a binary classification (True or False)

* Optimizer = Adam

### DCGAN

Then I also tried the Deep Convolutional Generative Adversarial Network ([DCGAN](https://arxiv.org/abs/1511.06434)) by Alec Radford, Luke Metz, Soumith Chintala

#### Network Structure:

##### Generator

```python
def build_generator(self):
    model = Sequential()
    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    model.summary()

    noise = Input(shape=(self.latent_dim,))
    img = model(noise)

    return Model(noise, img)
```

##### Discriminator

```python
def build_discriminator(self):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=self.img_shape)
    validity = model(img)

    return Model(img, validity)
```

#### Hyperameters

* Epoch = 10000

* Batch size = 64

* Activation Function = relu
  * tanh for output layer of generator
  * sigmoid for output layer of discriminator

* Loss Fuction = binary_crossentropy, since it is a binary classification (True or False)

* Optimizer = Adam

### CGAN

Lastly, I tried the Conditional Generative Adversarial Network ([CGAN](https://arxiv.org/abs/1411.1784)) by Mehdi Mirza, Simon Osindero

#### Network Structure:

##### Generator

```python
def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)
```

##### Discriminator

```python
def build_discriminator(self):

    model = Sequential()
    model.add(Dense(512, input_dim=np.prod(self.img_shape)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=self.img_shape)
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
    flat_img = Flatten()(img)

    model_input = multiply([flat_img, label_embedding])

    validity = model(model_input)

    return Model([img, label], validity)
```

#### Hyperameters

* Epoch = 50000

* Batch size = 64

* Activation Function = LeakyReLU(alpha = 0.2)
  * tanh for output layer of generator
  * sigmoid for output layer of discriminator

* Loss Fuction = binary_crossentropy, since it is a binary classification (True or False)

* Optimizer = Adam

## Special Skills:

* Normalize the inputs
  * scaled the data into (-1, 1)
  * used tahn for last layer of the generator output
* Used LeakyReLU to avoid sparse gradients 
* Used soft input, when training the discriminator, 
* Tried DCGAN
* Used the ADAM Optimizer

## Visualization:

### GAN:

Loss for entire 50000 epoch:

![GAN_50000](https://github.com/XiaosongWen/DS504CS586-S20/blob/master/project3/pic/loss_gan.png)

However, it seem that after the model converge after around 400 epoch:

![GAN_1000](https://github.com/XiaosongWen/DS504CS586-S20/blob/master/project3/pic/loss_gan_1000.png)

Here is a gif which indecates the flow if the trainning

![GAN_GIF](https://github.com/XiaosongWen/DS504CS586-S20/blob/master/project3/pic/GAN.gif)

The final generated images:

![](https://github.com/XiaosongWen/DS504CS586-S20/blob/master/project3/pic/GAN_50000.png)

### DCGAN:

Loss for entire 10000 epoch:

![DCGAN_10000](https://github.com/XiaosongWen/DS504CS586-S20/blob/master/project3\pic\loss_dcgan.png)

Here is a gif which indecates the flow if the trainning:

![DCGAN_GIF](https://github.com/XiaosongWen/DS504CS586-S20/blob/master/project3\pic\DCGAN.gif)

The final generated images:

![DCGAN_10000](https://github.com/XiaosongWen/DS504CS586-S20/blob/master/project3\pic\DCGAN_10000.png)

### CGAN:

Loss for entire 50000 epoch:

![CGAN_50000](https://github.com/XiaosongWen/DS504CS586-S20/blob/master/project3\pic\loss_cgan.png)

However, it seem that after the model converge after around 1000 epoch:

![CGAN_1000](https://github.com/XiaosongWen/DS504CS586-S20/blob/master/project3\pic\loss_cgan_2000.png)

Here is a gif which indecates the flow if the trainning:

![CGAN_GIF](https://github.com/XiaosongWen/DS504CS586-S20/blob/master/project3\pic\CGAN.gif)

The final generated images:

![CGAN_50000](https://github.com/XiaosongWen/DS504CS586-S20/blob/master/project3\pic\CGAN_50000.png)

## Other Dataset:

I also tried the GAN model on the [Fashion MNIST](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/) dataset, which instead of number from 0 - 9, it has 10 classes of fashion stuff. It also keep the same structure, each data is a 28×28 grayscale image for a cloth, or shoes. 

Here are my result, 

![](https://github.com/XiaosongWen/DS504CS586-S20/blob/master/project3\pic\gan_fashion.gif)

![](https://github.com/XiaosongWen/DS504CS586-S20/blob/master/project3\pic\Fashion_GAN_50000.png)