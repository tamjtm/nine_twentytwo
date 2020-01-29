from django.contrib.auth.mixins import LoginRequiredMixin, PermissionRequiredMixin
from django.db.models import Q
from django.shortcuts import redirect

# Create your views here.
from django.utils import timezone
from django.views.generic import ListView, DetailView, CreateView

from verification.models import Case, Image


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
        context['images'] = Image.objects.filter(case=self.object).order_by('id')
        return context

    def post(self, request, *args, **kwargs):
        # if request.method == 'POST':
        instance = Case.objects.get(id=request.POST.get('id'))
        if request.POST.get('result') == "accept":
            instance.confirm_status = True
        elif request.POST.get('result') == "reject":
            instance.confirm_status = False
        instance.confirm_time = timezone.now()
        instance.confirm_user = request.user
        instance.save()
        return redirect('index')
        # case = get_object_or_404(Case, pk=pk)
        # images = Image.objects.filter(case=case).order_by('id')
        # context = {'case': case, 'images': images}
        # self.object = self.get_object()
        # context = super(CaseDetailView, self).get_context_data(**kwargs)
        # return self.render_to_response(context=context)


class UploadView(PermissionRequiredMixin, CreateView):
    model = Image
    permission_required = 'verification.add_image'
    fields = []

    def post(self, request, *args, **kwargs):
        try:
            case = Case.objects.get(id=request.POST.get('case'))
        except Case.DoesNotExist:
            user = request.user
            case = Case(id=request.POST.get('case'), owner=user, upload_user=user)
            case.save()

        for file in request.FILES.getlist('images'):
            image = Image(case=case, original_image=file, upload_user=request.user)
            image.save()
            case.confirm_status = None
            case.save()
        return redirect('index')
