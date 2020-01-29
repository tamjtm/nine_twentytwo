from io import BytesIO

from django.contrib.auth.models import User
from django.core.files import File
from django.db import models

# Create your models here.
from django.utils.deconstruct import deconstructible

from verification.detection import *


class Case(models.Model):
    id = models.CharField(primary_key=True, max_length=20)
    owner = models.ForeignKey(User, related_name='own', on_delete=models.SET_NULL, null=True)
    upload_user = models.ForeignKey(User, related_name='upload', on_delete=models.SET_NULL, null=True)
    upload_time = models.DateTimeField(auto_now_add=True)
    confirm_user = models.ForeignKey(User, related_name='confirm', on_delete=models.SET_NULL, null=True, blank=True)
    confirm_time = models.DateTimeField(null=True, blank=True)
    confirm_status = models.BooleanField(null=True, blank=True)

    class Meta:
        ordering = ('id', 'upload_time', )

    def __str__(self):
        return self.id

    @property
    def get_status(self):
        if self.confirm_status == 1:
            return 'accepted'
        elif self.confirm_status == 0:
            return 'rejected'
        else:
            return 'waiting'

    @property
    def get_result(self):
        all_count = pos_count = neg_count = 0
        for img in Image.objects.filter(case=self).select_related():
            if img.result == 1:
                pos_count += 1
            elif img.result == 0:
                neg_count += 1
            all_count += 1
        if pos_count > 0:
            return 'Positive %.2f%%' % (pos_count/all_count*100)
        elif neg_count > 0:
            return 'Negative %.2f%%' % (neg_count/all_count*100)
        else:
            return 'cannot detect'


@deconstructible
class ImagePath(object):

    def __init__(self, img_name):
        self.img_name = img_name

    def __call__(self, instance, filename):
        return '/'.join([instance.case.id, instance.id, self.img_name + ".jpg"])


def set_images_path(image_name):
    return ImagePath(image_name)


def save_image(img, ch):
    temp = BytesIO()
    img.save(temp, 'JPEG')
    ch.save('temp.jpg', File(temp), save=False)


class Image(models.Model):
    id = models.CharField(primary_key=True, max_length=30, default='none')
    original_image = models.ImageField(upload_to=set_images_path("original"))
    case = models.ForeignKey(Case, on_delete=models.CASCADE)
    upload_user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    result = models.BooleanField(null=True, blank=True)
    result_image = models.ImageField(upload_to=set_images_path("result"), null=True, blank=True)
    chromosome_9a = models.ImageField(upload_to=set_images_path("9a"), null=True, blank=True)
    chromosome_9b = models.ImageField(upload_to=set_images_path("9b"), null=True, blank=True)
    chromosome_9c = models.ImageField(upload_to=set_images_path("9c"), null=True, blank=True)
    chromosome_9d = models.ImageField(upload_to=set_images_path("9d"), null=True, blank=True)
    chromosome_22a = models.ImageField(upload_to=set_images_path("22a"), null=True, blank=True)
    chromosome_22b = models.ImageField(upload_to=set_images_path("22b"), null=True, blank=True)
    chromosome_22c = models.ImageField(upload_to=set_images_path("22c"), null=True, blank=True)
    chromosome_22d = models.ImageField(upload_to=set_images_path("22d"), null=True, blank=True)

    class Meta:
        ordering = ('case', 'id', )

    def predict(self):
        ch_img, contours, meta_img = import_meta(self.original_image)

        model_9n = load_922_model('models/9N')
        model_9p = load_922_model('models/9P')
        img_9, result_9, framed = predict_922(ch_img[0], model_9n, model_9p, 9, contours, meta_img)

        model_22n = load_922_model('models/22N')
        model_22p = load_922_model('models/22P')
        img_22, result_22, framed = predict_922(ch_img[0], model_22n, model_22p, 22, contours, framed)
        framed = array_to_img(framed)

        if len(img_9) >= 2 and len(img_22) >= 2:
            save_image(img_9[0], self.chromosome_9a)
            save_image(img_9[1], self.chromosome_9b)
            save_image(img_22[0], self.chromosome_22a)
            save_image(img_22[1], self.chromosome_22b)
            save_image(framed, self.result_image)

            if len(img_9) == 3:
                save_image(img_9[2], self.chromosome_9c)
            if len(img_9) == 4:
                save_image(img_9[2], self.chromosome_9c)
                save_image(img_9[3], self.chromosome_9d)

            if len(img_22) == 3:
                save_image(img_22[2], self.chromosome_22c)
            if len(img_22) == 4:
                save_image(img_22[2], self.chromosome_22c)
                save_image(img_22[3], self.chromosome_22d)

            if result_9 + result_22 == 0:
                print(">>> Negative")
                return 0
            elif result_9 + result_22 == 2:
                print(">>> Positive")
                return 1
        print(">>> Cannot detect")
        return None

    def save(self, flag=True, *args, **kwargs):
        if flag:
            name = "%02d" % (Image.objects.filter(case=self.case).count()+1)
            self.id = self.case.id + "_" + name
            self.result = self.predict()
            self.save(flag=False, *args, **kwargs)
        return super(Image, self).save(*args, **kwargs)

    def __str__(self):
        return str(self.id)

    @property
    def get_result(self):
        if self.result == 1:
            return 'Positive'
        elif self.result == 0:
            return 'Negative'
        else:
            return 'cannot detect'
