from torchvision import transforms

class Normalize(object):

    def __init__(self):
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __call__(self, sample):
        image = sample["image"]
        image = self.normalize(image)
        return {"image": image}
