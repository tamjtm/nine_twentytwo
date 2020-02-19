import sys
from io import BytesIO

from django.contrib.auth.models import User
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.db import models

# Create your models here.
from django.db.models import Q
from django.utils.deconstruct import deconstructible

from verification.detection import *


def to_imagefield(img):
    img_io = BytesIO()
    img.save(img_io, 'JPEG')
    ch_img = InMemoryUploadedFile(img_io, 'ImageField', str(img), 'image/jpeg', sys.getsizeof(img_io), None)
    return ch_img


class Case(models.Model):
    id = models.CharField(primary_key=True, max_length=20)
    owner = models.ForeignKey(User, verbose_name="Physician", related_name='own', on_delete=models.SET_NULL, null=True)
    diff_diagnosis = models.CharField(max_length=100, verbose_name="Differential Diagnosis", null=True, blank=True)
    result = models.BooleanField(null=True, blank=True)
    upload_user = models.ForeignKey(User, related_name='upload', on_delete=models.SET_NULL, null=True)
    upload_time = models.DateTimeField(auto_now_add=True)
    confirm_user = models.ForeignKey(User, related_name='confirm', on_delete=models.SET_NULL, null=True, blank=True)
    confirm_time = models.DateTimeField(null=True, blank=True)
    confirm_status = models.BooleanField(null=True, blank=True)
    reject_message = models.CharField(max_length=200, null=True, blank=True)

    class Meta:
        ordering = ('id', 'upload_time',)

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

    # @property
    # def get_result(self):
    #     all_count = pos_count = neg_count = 0
    #     for img in MetaphaseImage.objects.filter(case=self).select_related():
    #         if img.result == 1:
    #             pos_count += 1
    #         elif img.result == 0:
    #             neg_count += 1
    #         all_count += 1
    #     if pos_count > 0:
    #         return True
    #     elif neg_count > 0:
    #         return False
    #     else:
    #         return None

    @property
    def get_metaphases(self):
        return MetaphaseImage.objects.filter(case=self).order_by('id')

    @property
    def get_new_metaphases(self):
        return MetaphaseImage.objects.filter(
            Q(case=self) & Q(result=None)
        ).order_by('id')

    def predict(self):
        meta_filenames = []
        for meta_img in self.get_new_metaphases:
            meta_filenames.append(meta_img.original_image)
        
        output = nine_22(meta_filenames)
        
        img_9, prob_9, pred_9, result_9 = [x for x in output[0:4]]     
        img_22, prob_22, pred_22, result_22 = [x for x in output[4:8]]
        #img_9 = [[img1-1,img1-2], [img2-1,img2-2,img2-3], [img3-1,img3-2]]
        framed  = output[8]     # [img1,img2,img3]

        # temp = BytesIO()
        # framed.save(temp, 'JPEG')
        # self.result_image.save('result.jpg', File(temp), save=False)
        for i, meta_img in enumerate(self.get_new_metaphases):
            meta_img.result_image = to_imagefield(framed[i])
            meta_img.save(flag=False)

            for j in range(len(img_9[i])):
                chromosome = ChromosomeImage(metaphase=meta_img,
                                             type=9,
                                             prediction=pred_9[i][j],
                                             image=to_imagefield(img_9[i][j]),
                                             prob=prob_9[i][j] * 100)
                chromosome.save()

            for j in range(len(img_22[i])):
                chromosome = ChromosomeImage(metaphase=meta_img,
                                             type=22,
                                             prediction=pred_22[i][j],
                                             image=to_imagefield(img_22[i][j]),
                                             prob=prob_22[i][j] * 100)
                chromosome.save()

            if result_9 is not None and result_22 is not None:
                if result_9 + result_22 == 0:
                    print(">>> Negative")
                    meta_img.result = 0
                    meta_img.save(flag=False)
                else:
                    print(">>> Positive")
                    meta_img.result = 1
                    meta_img.save(flag=False)
            else:
                print(">>> Cannot detect")
                meta_img.result = None
                meta_img.save(flag=False)

        all_count = pos_count = neg_count = 0
        for meta_img in self.get_metaphases:
            if meta_img.result == 1:
                pos_count += 1
            elif meta_img.result == 0:
                neg_count += 1
            all_count += 1
        print(all_count, pos_count, neg_count)
        if pos_count > 0:
            return True
        elif neg_count > 0:
            return False
        else:
            return None

    def save(self, flag=True, *args, **kwargs):
        if self.get_new_metaphases is not None and flag:
            self.result = self.predict()
            self.save(flag=False, *args, **kwargs)
        return super(Case, self).save(*args, **kwargs)


@deconstructible
class ImagePath(object):

    def __init__(self, img_name):
        self.img_name = img_name

    def __call__(self, instance, filename):
        if isinstance(instance, MetaphaseImage):
            return '/'.join([instance.case.id, instance.id, self.img_name + ".jpg"])
        else:
            return '/'.join([instance.metaphase.case.id, instance.metaphase.id, instance.name + ".jpg"])


def set_images_path(image_name):
    return ImagePath(image_name)


class MetaphaseImage(models.Model):
    id = models.CharField(primary_key=True, max_length=30, default='none')
    original_image = models.ImageField(upload_to=set_images_path("original"))
    case = models.ForeignKey(Case, on_delete=models.CASCADE)
    upload_user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    result = models.BooleanField(null=True, blank=True)
    result_image = models.ImageField(upload_to=set_images_path("result"), null=True, blank=True)

    class Meta:
        ordering = ('case', 'id',)

    # def predict(self):
    #     ch_img, contours, meta_img = import_meta(self.original_image)
    #
    #     model_9n = load_922_model('models/9N')
    #     model_9p = load_922_model('models/9P')
    #     img_9, prob_9, pred_9, result_9, framed, temp_index9 = predict_9(ch_img[0], model_9n, model_9p,
    #                                                                      contours, meta_img)
    #
    #     model_22f = load_922_model('models/22Find')
    #     model_22c = load_922_model('models/22Classify')
    #     img_22, prob_22, pred_22, result_22, framed, temp_index22 = predict_22(ch_img[0], model_22f, model_22c,
    #                                                                            contours, framed)
    #     framed = array_to_img(framed)
    #     framed = temp_index_function(framed, temp_index9, 9)
    #     framed = temp_index_function(framed, temp_index22, 22)
    #
    #     # if len(img_9) > 0 and len(img_22) > 0:
    #     temp = BytesIO()
    #     framed.save(temp, 'JPEG')
    #     self.result_image.save('result.jpg', File(temp), save=False)
    #
    #     for i in range(len(img_9)):
    #         chromosome = ChromosomeImage(metaphase=self,
    #                                      type=9,
    #                                      prediction=pred_9[i],
    #                                      image=to_imagefield(img_9[i]),
    #                                      prob=prob_9[i] * 100)
    #         chromosome.save()
    #
    #     for i in range(len(img_22)):
    #         chromosome = ChromosomeImage(metaphase=self,
    #                                      type=22,
    #                                      prediction=pred_22[i],
    #                                      image=to_imagefield(img_22[i]),
    #                                      prob=prob_22[i] * 100)
    #         chromosome.save()
    #
    #     if result_9 is not None and result_22 is not None:
    #         if result_9 + result_22 == 0:
    #             print(">>> Negative")
    #             return 0
    #         else:
    #             print(">>> Positive")
    #             return 1
    #     print(">>> Cannot detect")
    #     return None

    def save(self, flag=True, *args, **kwargs):
        if flag:
            name = "%02d" % (MetaphaseImage.objects.filter(case=self.case).count() + 1)
            self.id = self.case.id + "_" + name
            self.save(flag=False, *args, **kwargs)
            # self.result = self.predict()
        return super(MetaphaseImage, self).save(*args, **kwargs)

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

    @property
    def get_chromosome9(self):
        return ChromosomeImage.objects.filter(
            Q(metaphase=self) & Q(type=9)
        ).order_by('name')

    @property
    def get_chromosome22(self):
        return ChromosomeImage.objects.filter(
            Q(metaphase=self) & Q(type=22)
        ).order_by('name')


class ChromosomeImage(models.Model):
    id = models.CharField(primary_key=True, max_length=30, default='none')
    name = models.CharField(max_length=2, default='none')
    metaphase = models.ForeignKey(MetaphaseImage, on_delete=models.CASCADE)
    type = models.IntegerField(choices=((9, 'Chromosome 9'), (22, 'Chromosome 22')))
    prediction = models.BooleanField(null=True, blank=True)
    image = models.ImageField(upload_to=set_images_path("."), null=True, blank=True)
    prob = models.IntegerField(null=True, blank=True)

    def save(self, flag=True, *args, **kwargs):
        if flag:
            number = ChromosomeImage.objects.filter(
                Q(metaphase=self.metaphase) & Q(type=self.type)
            ).count()
            self.name = str(self.type) + chr(65 + number)
            self.id = self.metaphase.id + "_" + self.name
            self.save(flag=False, *args, **kwargs)
        return super(ChromosomeImage, self).save(*args, **kwargs)

    @property
    def get_prediction(self):
        if self.prediction == 0:
            return "NM"
        elif self.prediction == 1:
            return "PH"
        return "None"
