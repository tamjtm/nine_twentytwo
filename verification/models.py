import sys
from io import BytesIO

from django.contrib.auth.models import User
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.db import models
# Create your models here.
from django.db.models import Q
from django.utils.deconstruct import deconstructible

from verification.detection import nine_22


def to_imagefield(img):
    img_io = BytesIO()
    img.save(img_io, 'JPEG')
    ch_img = InMemoryUploadedFile(img_io, 'ImageField', str(img), 'image/jpeg', sys.getsizeof(img_io), None)
    return ch_img


class Case(models.Model):
    id = models.CharField(primary_key=True, verbose_name="Case ID", max_length=20)
    owner = models.ForeignKey(User, verbose_name="Physician", related_name='own', on_delete=models.SET_NULL, null=True)
    diff_diagnosis = models.CharField(max_length=100, verbose_name="Differential Diagnosis", null=True, blank=True)
    result = models.BooleanField(null=True, blank=True)
    percentage = models.IntegerField(null=True, blank=True)
    upload_user = models.ForeignKey(User, related_name='upload', on_delete=models.SET_NULL, null=True)
    upload_time = models.DateTimeField(auto_now_add=True)
    confirm_user = models.ForeignKey(User, related_name='confirm', on_delete=models.SET_NULL, null=True, blank=True)
    confirm_time = models.DateTimeField(null=True, blank=True)
    confirm_status = models.BooleanField(null=True, blank=True)
    reject_message = models.CharField(max_length=200, null=True, blank=True)
    recheck_message = models.CharField(max_length=200, null=True, blank=True)

    class Meta:
        ordering = ('id', 'upload_time',)

    def __str__(self):
        return self.id

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

        print('detection started...')
        prediction = nine_22(meta_filenames)
        print('...interpreting...')

        for i, meta_img in enumerate(self.get_new_metaphases, 1):
            meta_img.result_image = to_imagefield(prediction[i]['framed'])
            meta_img.save(flag=False)

            for j in range(len(prediction[i]['img_9'])):
                chromosome = ChromosomeImage(metaphase=meta_img,
                                             type=9,
                                             prediction=prediction[i]['pred_9'][j],
                                             image=to_imagefield(prediction[i]['img_9'][j]),
                                             prob=prediction[i]['prob_9'][j] * 100)
                chromosome.save()

            for j in range(len(prediction[i]['img_22'])):
                chromosome = ChromosomeImage(metaphase=meta_img,
                                             type=22,
                                             prediction=prediction[i]['pred_22'][j],
                                             image=to_imagefield(prediction[i]['img_22'][j]),
                                             prob=prediction[i]['prob_22'][j] * 100)
                chromosome.save()

            meta_img.result = prediction[i]['result']
            meta_img.save(flag=False)

        meta_result = []
        for meta_img in self.get_metaphases:
            meta_result.append(meta_img.result)

        print('...detection finished')
        print('---------------------------')
        print("ph detection result:", meta_result)

        self.percentage = sum(result for result in meta_result) / len(meta_result) * 100

        if 1 in meta_result:
            return True
        elif 0 in meta_result:
            return False
        else:
            return None

    def save(self, flag=False, *args, **kwargs):
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

    def save(self, flag=True, *args, **kwargs):
        if flag:
            name = "%02d" % (MetaphaseImage.objects.filter(case=self.case).count() + 1)
            self.id = self.case.id + "_" + name
            self.save(flag=False, *args, **kwargs)
        return super(MetaphaseImage, self).save(*args, **kwargs)

    def __str__(self):
        return str(self.id)

    @property
    def get_result(self):
        if self.result == 1:
            return 'found'
        elif self.result == 0:
            return 'not found'
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
        elif self.prediction == 1 and self.type == 9:
            return "DER"
        elif self.prediction == 1 and self.type == 22:
            return "PH"
        return "None"
