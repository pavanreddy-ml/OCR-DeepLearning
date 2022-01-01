from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Rectangle
from kivy.graphics import Color
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.config import Config
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import load_model
import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#loading the model
OCR_Dataset1_DenseNet201 = load_model('Dataset1/OCR_Dataset1_DenseNet201.h5')
OCR_Dataset2_All_Simple = models.load_model('Dataset2/OCR_Dataset2_All_Simple.h5')
OCR_Dataset2_Upper_Simple = models.load_model('Dataset2/OCR_Dataset2_Upper_Simple.h5')
OCR_Dataset2_Lower_Simple = models.load_model('Dataset2/OCR_Dataset2_Lower_Simple.h5')
OCR_Dataset2_Digit_Simple = models.load_model('Dataset2/OCR_Dataset2_Digit_Simple.h5')

Window.size = (800, 520)
Window.clearcolor = (27/255, 36/255, 52/255, 1)
Config.set('graphics', 'resizable', '0')

dataset_list = {'Dataset1\nDenseNet201' : 1, 'Dataset2\nSimple Conv2D' : 2}
dataset = 1
character_type_list = {'All' : 1, 'Upper\nCase' : 2, 'Lower\nCase' : 3, 'Digits' : 4}
character_type = 1
disable_draw = False

path = None
images_dir = 'Final/'
dir_list = os.listdir(images_dir)
character_class_actual = None



class Boxes(Widget):

    pos_factor = 10
    grid_size = 50

    def __init__(self, **kwargs):
        super(Boxes, self).__init__(**kwargs)

        self.wid = Widget()
        self.second = BoxLayout()

        self.create_canvas()

        self.corrected_row = {}
        for i in range(self.grid_size):
            self.corrected_row[i] = (self.grid_size - 1) - i

        self.add_widget(self.second)
        self.add_widget(self.wid)


    def click(self, value):
        global dataset_list, dataset, character_type_list, character_type

        if value.state == 'normal':
            value.state = 'down'

        if value.text in dataset_list:
            dataset = dataset_list[value.text]
        elif value.text in character_type_list:
            character_type = character_type_list[value.text]


        if dataset == 1:
            self.ids.All.disabled = False
            self.ids.All.state = 'down'
            character_type = 1
            self.ids.Upper.state = 'normal'
            self.ids.Upper.disabled = True
            self.ids.Lower.state = 'normal'
            self.ids.Lower.disabled = True
            self.ids.Digits.state = 'normal'
            self.ids.Digits.disabled = True
        elif dataset == 2:
            self.ids.All.disabled = False
            self.ids.Upper.disabled = False
            self.ids.Lower.disabled = False
            self.ids.Digits.disabled = False

        print(dataset, character_type)


    def create_canvas(self):
        with self.wid.canvas:
            Color(1, 1, 1, 1, mode='rgba')
            Rectangle(pos=(5, 4), size=(512, 512))


    def on_touch_down(self, touch):
        global disable_draw

        super().on_touch_down(touch)
        if disable_draw == False:
            try:
                # in kivy grid co-ordinates follow the format (col,row)
                #also it helps in keeping the drawing parts within the canvas
                row, col = self.corrected_row[touch.pos[1] // self.pos_factor], int(touch.pos[0] // self.pos_factor) # it's basically corrected_row[COL//self.pos_factor] , int(ROW) => ROW , COL [normal]
                #print(row, col)

                if (row < 49) & (row > 0) & (col < 49) & (col > 0):
                    g_row = touch.pos[0] // self.pos_factor
                    g_col = touch.pos[1] // self.pos_factor
                    with self.wid.canvas:
                        Color(0, 0, 0, 1, mode='rgba')
                        Rectangle(pos=(g_row * self.pos_factor, g_col * self.pos_factor), size=(30, 30))
            except KeyError:
                pass


    def on_touch_move(self, touch):
        global disable_draw

        if disable_draw == False:
            try:
                row,col=self.corrected_row[touch.pos[1] // self.pos_factor],int(touch.pos[0] // self.pos_factor)
                #print(row, col)

                if (row < 49) & (row > 0) & (col < 49) & (col > 0):
                    g_row = touch.pos[0] // self.pos_factor
                    g_col = touch.pos[1] // self.pos_factor
                    with self.wid.canvas:
                        Color(0, 0, 0, 1, mode='rgba')
                        Rectangle(pos=(g_row * self.pos_factor, g_col * self.pos_factor), size=(30, 30))
            except:
                pass


    def clear_screen(self, instance):
        global disable_draw

        disable_draw = False
        self.wid.canvas.clear()
        self.create_canvas()


    def generate_random(self, value):
        global disable_draw, path, dir_list, character_class_actual

        disable_draw = True

        while True:
            if character_type == 1:
                rand_character = random.choice(dir_list)
                character_class_actual = rand_character
                break

            if character_type == 2:
                rand_character = random.choice(dir_list)
                if rand_character.isupper():
                    character_class_actual = rand_character
                    break

            if character_type == 3:
                rand_character = random.choice(dir_list)
                if rand_character.islower():
                    character_class_actual = rand_character
                    break

            if character_type == 4:
                rand_character = random.choice(dir_list)
                if rand_character.isnumeric():
                    character_class_actual = rand_character
                    break

        images_list = os.listdir(images_dir + rand_character + '/')

        rand_image = random.choice(images_list)

        path = images_dir + rand_character + '/' + rand_image

        print(character_class_actual)

        with self.wid.canvas:
            Image(source=path, allow_stretch=True, size=(512, 512), pos=(5, 4))


    def predict(self,instance):
        global path

        if disable_draw == False:
            Window.screenshot(name='sample.png')

            image = tf.io.read_file('sample0001.png')
            image = tf.io.decode_image(image)
            image = image[-516:-4, 5:517, 0:1]

            # temp_save = tf.cast(image, tf.uint8)
            # temp_save = tf.image.encode_jpeg(temp_save, quality=100, format='grayscale')
            # tf.io.write_file('temp_save.jpg', temp_save)

            print(image.shape)

            if os.path.exists('sample0001.png'):
                os.remove('sample0001.png')

        elif disable_draw == True:
            image = tf.io.read_file(path)
            image = tf.io.decode_image(image)
            print(image.shape)


        img_predict = tf.image.grayscale_to_rgb(image)
        img_predict = tf.image.resize(img_predict, [255, 255])
        img_predict = tf.expand_dims(img_predict, axis=0)
        print(img_predict.shape)


        if dataset == 1:
            pred_probs = OCR_Dataset1_DenseNet201.predict()

        if dataset == 2:
            if character_type == 1:
                pred_probs = OCR_Dataset2_All_Simple.predict()
            if character_type == 2:
                pred_probs = OCR_Dataset2_Upper_Simple.predict()
            if character_type == 3:
                pred_probs = OCR_Dataset2_Lower_Simple.predict()
            if character_type == 4:
                pred_probs = OCR_Dataset2_Digit_Simple.predict()

        print(pred_probs.argmax())


    #     prediction=model.predict(img_predict)
    #     # print(prediction)
    #     guessed_number=np.argmax(prediction)


    pass


class OpticalCharacterRecognition(App):

    def build(self):
        layout = Boxes()
        return layout



if __name__ == "__main__":
    OpticalCharacterRecognition().run()