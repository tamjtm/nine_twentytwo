import time
from random import randint

from django.contrib.auth.mixins import LoginRequiredMixin, PermissionRequiredMixin
from django.contrib.auth.models import User
from django.db.models import Q
from django.forms import TextInput
from django.shortcuts import redirect
from django.utils import timezone
from django.views.generic import ListView, DetailView, CreateView

from verification.models import Case, MetaphaseImage, ChromosomeImage


class CaseListView(LoginRequiredMixin, ListView):
    model = Case

    def get_queryset(self):
        query = self.request.GET.get('search')
        if query:
            object_list = Case.objects.filter(
                Q(id__icontains=query) | Q(upload_user__username__icontains=query)
                | Q(confirm_user__username__icontains=query) | Q(owner__username__icontains=query)
            ).order_by('confirm_status', 'upload_time')
        else:
            object_list = Case.objects.all().order_by('confirm_status', 'upload_time')
        return object_list

    def get_context_data(self, **kwargs):
        context = super(CaseListView, self).get_context_data(**kwargs)
        context['keyword'] = self.request.GET.get('search')
        return context


class CaseUserListView(PermissionRequiredMixin, ListView):
    model = Case
    permission_required = 'verification.view_case'

    def get_queryset(self):
        if self.request.user.has_perm('verification.change_case'):
            object_list = Case.objects.filter(
                Q(upload_user=self.request.user) | Q(confirm_user=self.request.user)
            ).order_by('upload_time')
        else:
            object_list = Case.objects.filter(owner=self.request.user).order_by('upload_time')
        return object_list


class CaseDetailView(LoginRequiredMixin, DetailView):
    model = Case
    fields = ['reject_message']

    # confirmation
    def post(self, request, *args, **kwargs):
        instance = Case.objects.get(id=request.POST.get('id'))
        if request.POST.get('result') == "accept":
            instance.confirm_status = True
            instance.reject_message = None
            instance.recheck_message = None
            instance.confirm_time = timezone.now()
            instance.confirm_user = request.user
        elif request.POST.get('result') == "reject":
            instance.confirm_status = False
            instance.recheck_message = None
            instance.reject_message = request.POST.get('message')
            instance.confirm_time = timezone.now()
            instance.confirm_user = request.user
        elif request.POST.get('result') == "recheck":
            instance.confirm_status = None
            instance.recheck_message = request.POST.get('message')
            instance.confirm_user = None
            instance.confirm_time = None
        instance.save()
        return redirect('index')


class UploadView(PermissionRequiredMixin, CreateView):
    model = Case
    permission_required = 'verification.add_metaphaseimage'
    fields = ['id', 'diff_diagnosis']
    widgets = {
        'text': TextInput(attrs={
            'required': True,
        }),
    }

    def post(self, request, *args, **kwargs):
        user = request.user

        count = Case.objects.filter(id=request.POST.get('id')).count()

        if count > 0:
            case = Case.objects.get(id=request.POST.get('id'))
        else:
            owner_list = User.objects.filter(groups__name='Doctor')
            count = owner_list.count()
            if count > 0:
                random_index = randint(0, count-1)
                owner = owner_list[random_index]
            else:
                owner = request.user
            case = Case(id=request.POST.get('id'), owner=owner,
                        diff_diagnosis=request.POST.get('diff_diagnosis'), upload_user=user)
            case.save()

        start = time.time()
        images_list = request.FILES.getlist('images')

        for i, file in enumerate(images_list, 1):
            image = MetaphaseImage(case=case, original_image=file, upload_user=user)
            image.save()

        case.confirm_status = None
        case.save(flag=True)
        end = time.time()
        timer = int(end-start)
        print(len(images_list), "imgs =>", timer, "s.")

        return redirect('case-detail', pk=case.id)


class MetaphaseListView(PermissionRequiredMixin, ListView):
    model = MetaphaseImage
    permission_required = 'verification.add_metaphaseimage'

    def get_context_data(self, **kwargs):
        context = super(MetaphaseListView, self).get_context_data(**kwargs)
        context['case_list'] = Case.objects.all()
        return context


class MetaphaseDetailView(PermissionRequiredMixin, DetailView):
    model = MetaphaseImage
    permission_required = 'verification.add_metaphaseimage'

    def get_context_data(self, **kwargs):
        context = super(MetaphaseDetailView, self).get_context_data(**kwargs)
        context['chromosomes'] = ChromosomeImage.objects.filter(
            Q(name__icontains='ch')
        )
        return context
