from random import randint
import time
from django.contrib.auth.mixins import LoginRequiredMixin, PermissionRequiredMixin
from django.contrib.auth.models import User
from django.db.models import Q
from django.forms import TextInput
from django.shortcuts import redirect
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
    fields = ['reject_message']

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


# def add_images(images_list, case_id, user_id):
#     case = Case.objects.get(id=case_id)
#     user = User.objects.get(id=user_id)
#     import time
#     start = time.time()
#     for i, file in enumerate(images_list):
#         image = MetaphaseImage(case=case, original_image=file, upload_user=user)
#         image.save()
#         return render(request, 'userhomepage.html', result)
#     case.confirm_status = None
#     case.save()
#     end = time.time()
#     timer = int(end-start)
#     print(len(images_list), "imgs =>", timer, "s.")


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
        try:
            case = Case.objects.get(id=request.POST.get('id'))
        except Case.DoesNotExist:
            owner_list = User.objects.filter(groups__name='Doctor')
            count = owner_list.count()
            if count > 0:
                random_index = randint(0, count-1)
                owner = owner_list[random_index]
            else:
                owner = request.user
            case = Case(id=request.POST.get('id'), owner=owner,
                        diff_diagnosis=request.POST.get('diff_diagnosis'), upload_user=user)
            case.save(flag=False)

        # add_images(request.FILES.getlist('images'), case.id, request.user.id)
        images_list = request.FILES.getlist('images')
        start = time.time()
        rendered_str = []
        for i, file in enumerate(images_list, 1):
            image = MetaphaseImage(case=case, original_image=file, upload_user=user)
            image.save()
            result = {'current': i, 'total': len(images_list)}
            # render(request, 'verification/case_form.html', result)
        case.confirm_status = None
        case.save()
        end = time.time()
        timer = int(end-start)
        print(len(images_list), "imgs =>", timer, "s.")

        return redirect('case-detail', pk=case.id)
