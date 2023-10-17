import os
import time
from math import ceil
from PIL import Image, ImageFont, ImageDraw
import imageio
from os import listdir
from os.path import isfile, join
import re

from pathlib import Path
abs_path = Path(__file__).parent


# https://stackoverflow.com/questions/29760402/converting-a-txt-file-to-an-image-in-python
# https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
class GIF_generator:
    def __init__(self, experiment_name):
        self.PIL_GRAYSCALE = 'L'
        self.PIL_WIDTH_INDEX = 0
        self.PIL_HEIGHT_INDEX = 1
        self.COMMON_MONO_FONT_FILENAMES = [
            'DejaVuSansMono.ttf',  # Linux
            'Consolas Mono.ttf',  # MacOS, I think
            'Consola.ttf',  # Windows, I think
        ]
        self.experiment_name = experiment_name
        self.images_path = abs_path / ('../.tmp/images/')
        self.texts_path = abs_path / ('../.tmp/texts/')
        self.gif_name = abs_path / ('../gifs/' + self.experiment_name + '.gif')

    def textfile_to_image(self, textfile):
        with open(textfile) as f:
            lines = tuple(line.rstrip() for line in f.readlines())
            font = None
            large_font = 20
            for font_filename in self.COMMON_MONO_FONT_FILENAMES:
                try:
                    font = ImageFont.truetype(font_filename, size=large_font)
                    # print(f'Using font "{font_filename}".')
                    break
                except IOError:
                    print(f'Could not load font "{font_filename}".')
            if font is None:
                font = ImageFont.load_default()
                # print('Using default font.')

            # make a sufficiently sized background image based on the combination of font and lines
            font_points_to_pixels = lambda pt: round(pt * 96.0 / 72)
            margin_pixels = 20

            # height of the background image
            tallest_line = max(lines, key=lambda line: font.getsize(line)[self.PIL_HEIGHT_INDEX])
            max_line_height = font_points_to_pixels(font.getsize(tallest_line)[self.PIL_HEIGHT_INDEX])
            realistic_line_height = max_line_height * 0.8  # apparently it measures a lot of space above visible content
            image_height = int(ceil(realistic_line_height * len(lines) + 2 * margin_pixels))

            # width of the background image
            widest_line = max(lines, key=lambda s: font.getsize(s)[self.PIL_WIDTH_INDEX])
            max_line_width = font_points_to_pixels(font.getsize(widest_line)[self.PIL_WIDTH_INDEX])
            image_width = int(ceil(max_line_width + (2 * margin_pixels)))

            # draw the background
            background_color = 255  # white
            image = Image.new(self.PIL_GRAYSCALE, (image_width, image_height), color=background_color)
            draw = ImageDraw.Draw(image)

            # draw each line of text
            font_color = 0  # black
            horizontal_position = margin_pixels
            for i, line in enumerate(lines):
                vertical_position = int(round(margin_pixels + (i * realistic_line_height)))
                draw.text((horizontal_position, vertical_position), line, fill=font_color, font=font)

            return image

    def convert_all_texts_to_images(self):
        onlyfiles = [f for f in listdir(self.texts_path) if isfile(join(self.texts_path, f)) and self.experiment_name in f and '.txt' in f]
        for textfile in onlyfiles:
            name = textfile.split(".")[0]
            name = name + '.png'
            image = self.textfile_to_image(os.path.join(self.texts_path, textfile))
            # image.show()
            image.save(os.path.join(self.images_path, name))

    def get_gif_from_images(self):
        self.convert_all_texts_to_images()
        onlyfiles = [f for f in listdir(self.images_path) if isfile(join(self.images_path, f)) and self.experiment_name in f and '.png' in f]
        onlyfiles.sort(key=natural_keys)
        images = []
        for img in onlyfiles:
            images.append(imageio.v2.imread(os.path.join(self.images_path, img)))
        imageio.mimsave(self.gif_name, images, fps=4)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def create_directory(parent_path, dir_name):
    dirs = os.listdir(parent_path)
    if dir_name not in dirs:
        path = os.path.join(parent_path, dir_name)
        os.mkdir(path)
        os.mkdir(os.path.join(path, 'texts'))
        os.mkdir(os.path.join(path, 'images'))
    return


def get_timestamp():
    ts = time.gmtime()
    return str(time.strftime("%d_%m_%Y__%H_%M_%S", ts))
