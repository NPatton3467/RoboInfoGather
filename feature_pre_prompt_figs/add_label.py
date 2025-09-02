import imageio as iio
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import numpy as np

im_file = './fridge_not_near_microwave.png'
save_img_file = './labeled/fridge_not_near_microwave.png'
img_raw = iio.imread(im_file)
img = Image.fromarray(img_raw).convert('RGB')
draw = ImageDraw.Draw(img)

coord = np.array([270,370])
cr = 18

draw.ellipse(
    (
        coord[0] - cr,
        coord[1] - cr,
        coord[0] + cr,
        coord[1] + cr,
    ),
    fill=(200, 200, 200, 255),
    outline=(0, 0, 0, 255),
    width=3,
)

draw.text(
    tuple(coord.astype(int).tolist()),
    'A',
    #font=fnt,
    fill=(0, 0, 0, 255),
    anchor="mm",
    font_size=15,
)

plt.imshow(img)
plt.axis('off')
dpi = 500
figure = plt.gcf()
figure.set_size_inches(img_raw.shape[1]/dpi, img_raw.shape[0]/dpi)
plt.savefig(save_img_file, bbox_inches='tight', pad_inches=0, dpi=dpi)
