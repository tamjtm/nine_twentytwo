# Generated by Django 3.0.2 on 2020-02-11 09:14

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import verification.models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Case',
            fields=[
                ('id', models.CharField(max_length=20, primary_key=True, serialize=False)),
                ('diff_diagnosis', models.CharField(blank=True, max_length=100, null=True, verbose_name='Differential Diagnosis')),
                ('upload_time', models.DateTimeField(auto_now_add=True)),
                ('confirm_time', models.DateTimeField(blank=True, null=True)),
                ('confirm_status', models.BooleanField(blank=True, null=True)),
                ('reject_message', models.CharField(blank=True, max_length=200, null=True)),
                ('confirm_user', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='confirm', to=settings.AUTH_USER_MODEL)),
                ('owner', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='own', to=settings.AUTH_USER_MODEL, verbose_name='Physician')),
                ('upload_user', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='upload', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ('id', 'upload_time'),
            },
        ),
        migrations.CreateModel(
            name='MetaphaseImage',
            fields=[
                ('id', models.CharField(default='none', max_length=30, primary_key=True, serialize=False)),
                ('original_image', models.ImageField(upload_to=verification.models.ImagePath('original'))),
                ('result', models.BooleanField(blank=True, null=True)),
                ('result_image', models.ImageField(blank=True, null=True, upload_to=verification.models.ImagePath('result'))),
                ('case', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='verification.Case')),
                ('upload_user', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ('case', 'id'),
            },
        ),
        migrations.CreateModel(
            name='ChromosomeImage',
            fields=[
                ('id', models.CharField(default='none', max_length=30, primary_key=True, serialize=False)),
                ('name', models.CharField(default='none', max_length=2)),
                ('type', models.IntegerField(choices=[(9, 'Chromosome 9'), (22, 'Chromosome 22')])),
                ('prediction', models.BooleanField(blank=True, null=True)),
                ('image', models.ImageField(blank=True, null=True, upload_to=verification.models.ImagePath('.'))),
                ('prob', models.FloatField(blank=True, null=True)),
                ('metaphase', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='verification.MetaphaseImage')),
            ],
        ),
    ]
