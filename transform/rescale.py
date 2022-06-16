from skimage.transform import rescale, resize
from skimage.color import gray2rgb
from torchvision import transforms

class Rescale(object):

    def __init__(self, output_size=512):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.resize = transforms.Resize(output_size)

    def __call__(self, sample):
        source_image = sample["image"]
        return {"image": self.resize(source_image)}
        # new_source_image = self.__resize(source_image)
        # if new_source_image.shape == (self.output_size, self.output_size, 3):
        #     return {"image": new_source_image}
        # else:
        #     return {"image": self.__resize(source_image, True)}

    def __resize(self, image, force_resize=False):
        if len(image.shape) < 3:
            image = gray2rgb(image)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        if force_resize:
            if isinstance(self.output_size, int):
                return resize(image, (self.output_size, self.output_size))
            else:
                new_h, new_w = self.output_size
                return resize(image, (int(new_h), int(new_w)))
        else:
            h, w = image.shape[:2]
            if isinstance(self.output_size, int):
                if h == w:
                    try:
                        return rescale(image, (self.output_size / h, self.output_size / w, 1))
                    except Exception as e:
                        print(e)
                else:
                    if h > w:
                        new_h, new_w = self.output_size * h / w, self.output_size
                    else:
                        new_h, new_w = self.output_size, self.output_size * w / h
                    return resize(image, (int(new_h), int(new_w)))
            else:
                new_h, new_w = self.output_size
                h_ratio = h / new_h if h / new_h > 1 else 1 / (h / new_h)
                w_ratio = w / new_w if w / new_w > 1 else 1 / (w / new_w)
                if h_ratio.is_integer() and w_ratio.is_integer():
                    return rescale(image, (h / new_h, w / new_w))
                return resize(image, (int(new_h), int(new_w)))