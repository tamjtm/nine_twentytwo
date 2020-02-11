from random import randint

from django.contrib.auth.mixins import LoginRequiredMixin, PermissionRequiredMixin
from django.contrib.auth.models import User
from django.db.models import Q
from django.shortcuts import redirect
from django.urls import reverse
from django.utils import timezone
from django.views.generic import ListView, DetailView, CreateView
from verification.models import Case, MetaphaseImage


class CaseListView(LoginRequiredMixin, ListView):
    model = Case

    def get_queryset(self):
        query = self.request.GET.get('search')
        if query:
            object_list = Case.objects.filter(
                Q(id__icontains=query) | Q(upload_user__username__icontains=query)
                | Q(confirm_user__username__icontains=query) | Q(owner__username__icontains=query)
                | Q(confirm_status__icontains=query)

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

    def get_context_data(self, **kwargs):
        context = super(CaseDetailView, self).get_context_data(**kwargs)
        context['images'] = MetaphaseImage.objects.filter(case=self.object).order_by('id')
        return context

    # confirmation
    def post(self, request, *args, **kwargs):
        instance = Case.objects.get(id=request.POST.get('id'))
        if request.POST.get('result') == "accept":
            instance.confirm_status = True
            instance.reject_message = None
        elif request.POST.get('result') == "reject":
            instance.confirm_status = False
            instance.reject_message = request.POST.get('message')
        instance.confirm_time = timezone.now()
        instance.confirm_user = request.user
        instance.save()
        return redirect('index')


def add_images(images_list, case_id, user_id):
    case = Case.objects.get(id=case_id)
    user = User.objects.get(id=user_id)

    for file in images_list:
        image = MetaphaseImage(case=case, original_image=file, upload_user=user)
        image.save()
        case.confirm_status = None
        case.save()


class UploadView(PermissionRequiredMixin, CreateView):
    model = Case
    permission_required = 'verification.add_metaphaseimage'
    fields = ['diff_diagnosis']

    def post(self, request, *args, **kwargs):
        try:
            case = Case.objects.get(id=request.POST.get('case'))
        except Case.DoesNotExist:
            user = request.user
            # perm_doctor = Permission.objects.get(codename='view_case')
            # perm_meditech = Permission.objects.get(codename='change_case')
            # owner_list = User.objects.filter(
            #     Q(user_permissions=perm_doctor) & ~Q(user_permissions=perm_meditech)
            # )
            owner_list = User.objects.filter(groups__name='Doctor')
            count = owner_list.count()
            if count > 0:
                random_index = randint(0, count-1)
                owner = owner_list[random_index]
            else:
                owner = request.user
            case = Case(id=request.POST.get('case'), owner=owner,
                        diff_diagnosis=request.POST.get('diff_diagnosis'), upload_user=user)
            case.save()

        add_images(request.FILES.getlist('images'), case.id, request.user.id)
        return redirect('case-detail', pk=case.id)
