from django.contrib import admin

# Register your models here.
from verification.models import Case, Image


class CaseAdmin(admin.ModelAdmin):
    list_display = ('id', 'upload_user', 'confirm_user', 'owner')


class ImageAdmin(admin.ModelAdmin):
    list_display = ('id', 'case', 'original_image', 'result')


admin.site.register(Case, CaseAdmin)
admin.site.register(Image, ImageAdmin)
