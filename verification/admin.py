from django.contrib import admin

# Register your models here.
from verification.models import *


class CaseAdmin(admin.ModelAdmin):
    list_display = ('id', 'upload_user', 'confirm_user', 'owner')


class MetaphaseImageAdmin(admin.ModelAdmin):
    list_display = ('id', 'case', 'original_image', 'result')


class ChromosomeImageAdmin(admin.ModelAdmin):
    list_display = ('id', 'metaphase', 'name', 'image', 'prob')


admin.site.register(Case, CaseAdmin)
admin.site.register(MetaphaseImage, MetaphaseImageAdmin)
admin.site.register(ChromosomeImage, ChromosomeImageAdmin)
