from django.shortcuts import redirect, render
from django.contrib.messages.views import SuccessMessageMixin
from django.urls import reverse_lazy
from django.views import generic
from bootstrap_modal_forms.mixins import PassRequestMixin
from .recommend.transformer import RecTransformer
from .recommend.embeddings import get_book_embedding
from .models import User, Book, Chat, DeleteRequest, Feedback, UserBookInteraction
from django.contrib import messages
from django.db.models import Sum
from django.views.generic import CreateView, DetailView, DeleteView, UpdateView, ListView
from .forms import ChatForm, BookForm, UserForm, ISBNForm
from .utils import fetch_book_info_from_isbn
from . import models
import operator
import itertools
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.contrib.auth import authenticate, logout
from django.contrib import auth, messages
from django.contrib.auth.hashers import make_password
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Book
import cv2
import numpy as np
import torch
from paddleocr import PaddleOCR
from PIL import Image
import requests
import tempfile

def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image

def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    return cv2.bitwise_not(image)

def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    return cv2.bitwise_not(image)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh



# Shared Views
def login_form(request):
    return render(request, 'bookstore/login.html')


def logoutView(request):
    logout(request)
    return redirect('home')


def loginView(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None and user.is_active:
            auth.login(request, user)
            if user.is_admin or user.is_superuser:
                return redirect('dashboard')
            elif user.is_librarian:
                return redirect('librarian')
            else:
                return redirect('publisher')
        else:
            messages.info(request, "Invalid username or password")
            return redirect('home')


def register_form(request):
    return render(request, 'bookstore/register.html')


def registerView(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        password = make_password(password)

        a = User(username=username, email=email, password=password)
        a.save()
        messages.success(request, 'Account was created successfully')
        return redirect('home')
    else:
        messages.error(request, 'Registration fail, try again later')
        return redirect('regform')



















            


# Publisher views
@login_required
def publisher(request):
    return render(request, 'publisher/home.html')


@login_required
def uabook_form(request):
    return render(request, 'publisher/add_book.html')


@login_required
def request_form(request):
    return render(request, 'publisher/delete_request.html')


@login_required
def feedback_form(request):
    return render(request, 'publisher/send_feedback.html')

@login_required
def about(request):
    return render(request, 'publisher/about.html')	


@login_required
def usearch(request):
    query = request.GET['query']
    print(type(query))


    #data = query.split()
    data = query
    print(len(data))
    if( len(data) == 0):
        return redirect('publisher')
    else:
                a = data

                # Searching for It
                qs5 =models.Book.objects.filter(id__iexact=a).distinct()
                qs6 =models.Book.objects.filter(id__exact=a).distinct()

                qs7 =models.Book.objects.all().filter(id__contains=a)
                qs8 =models.Book.objects.select_related().filter(id__contains=a).distinct()
                qs9 =models.Book.objects.filter(id__startswith=a).distinct()
                qs10 =models.Book.objects.filter(id__endswith=a).distinct()
                qs11 =models.Book.objects.filter(id__istartswith=a).distinct()
                qs12 =models.Book.objects.all().filter(id__icontains=a)
                qs13 =models.Book.objects.filter(id__iendswith=a).distinct()




                files = itertools.chain(qs5, qs6, qs7, qs8, qs9, qs10, qs11, qs12, qs13)

                res = []
                for i in files:
                    if i not in res:
                        res.append(i)


                # word variable will be shown in html when user click on search button
                word="Searched Result :"
                print("Result")

                print(res)
                files = res




                page = request.GET.get('page', 1)
                paginator = Paginator(files, 10)
                try:
                    files = paginator.page(page)
                except PageNotAnInteger:
                    files = paginator.page(1)
                except EmptyPage:
                    files = paginator.page(paginator.num_pages)
   


                if files:
                    return render(request,'publisher/result.html',{'files':files,'word':word})
                return render(request,'publisher/result.html',{'files':files,'word':word})



@login_required
def delete_request(request):
    if request.method == 'POST':
        book_id = request.POST['delete_request']
        current_user = request.user
        user_id = current_user.id
        username = current_user.username
        user_request = username + "  want book with id  " + book_id + " to be deleted"

        a = DeleteRequest(delete_request=user_request)
        a.save()
        messages.success(request, 'Request was sent')
        return redirect('request_form')
    else:
        messages.error(request, 'Request was not sent')
        return redirect('request_form')



@login_required
def send_feedback(request):
    if request.method == 'POST':
        feedback = request.POST['feedback']
        current_user = request.user
        user_id = current_user.id
        username = current_user.username
        feedback = username + " " + " says " + feedback

        a = Feedback(feedback=feedback)
        a.save()
        messages.success(request, 'Feedback was sent')
        return redirect('feedback_form')
    else:
        messages.error(request, 'Feedback was not sent')
        return redirect('feedback_form')


























class UBookListView(LoginRequiredMixin,ListView):
    model = Book
    template_name = 'publisher/book_list.html'
    context_object_name = 'books'
    paginate_by = 100

    def get_queryset(self):
        return Book.objects.order_by('-id')

@login_required
def uabook(request):
    if request.method == 'POST':
        title = request.POST['title']
        author = request.POST['author']
        year = request.POST['year']
        publisher = request.POST['publisher']
        desc = request.POST['desc']
        cover = request.FILES['cover']
        pdf = request.FILES['pdf']
        current_user = request.user
        user_id = current_user.id
        username = current_user.username

        a = Book(title=title, author=author, year=year, publisher=publisher, 
            desc=desc, cover=cover, pdf=pdf, uploaded_by=username, user_id=user_id)
        a.save()
        messages.success(request, 'Book was uploaded successfully')
        return redirect('publisher')
    else:
        messages.error(request, 'Book was not uploaded successfully')
        return redirect('uabook_form')	



class UCreateChat(LoginRequiredMixin, CreateView):
    form_class = ChatForm
    model = Chat
    template_name = 'publisher/chat_form.html'
    success_url = reverse_lazy('ulchat')


    def form_valid(self, form):
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        self.object.save()
        return super().form_valid(form)


class UListChat(LoginRequiredMixin, ListView):
    model = Chat
    template_name = 'publisher/chat_list.html'

    def get_queryset(self):
        return Chat.objects.filter(posted_at__lt=timezone.now()).order_by('posted_at')































# Librarian views
def librarian(request):
    book = Book.objects.all().count()
    user = User.objects.all().count()

    context = {'book':book, 'user':user}

    return render(request, 'librarian/home.html', context)


@login_required
def labook_form(request):
    return render(request, 'librarian/add_book.html')


@login_required
def labook(request):
    if request.method == 'POST':
        title = request.POST['title']
        author = request.POST['author']
        year = request.POST['year']
        publisher = request.POST['publisher']
        desc = request.POST['desc']
        cover = request.FILES['cover']
        pdf = request.FILES['pdf']
        current_user = request.user
        user_id = current_user.id
        username = current_user.username

        a = Book(title=title, author=author, year=year, publisher=publisher, 
            desc=desc, cover=cover, pdf=pdf, uploaded_by=username, user_id=user_id)
        a.save()
        messages.success(request, 'Book was uploaded successfully')
        return redirect('llbook')
    else:
        messages.error(request, 'Book was not uploaded successfully')
        return redirect('llbook')
    
@login_required
def labookisbn_form(request):
    return render(request, 'librarian/add_book_isbn.html')

@login_required
def labookisbn(request):
    if request.method == 'POST':
        form = ISBNForm(request.POST)
        if form.is_valid():
            isbn = form.cleaned_data['isbn']
            book_data = fetch_book_info_from_isbn(isbn)
            if book_data:
                book = Book.objects.create(
                    isbn=isbn,
                    title=book_data['title'],
                    author=book_data['author'],
                    publisher=book_data['publisher'],
                    year=book_data['year'],
                    desc=book_data['desc'],
                    uploaded_by=request.user.username,
                    user_id=request.user.id
                )
                # return redirect('book_detail', pk=book.pk)
                return redirect('llbook')
                
            else:
                form.add_error('isbn', 'Không tìm thấy sách với ISBN này.')
            return redirect('llbook')
    else:
        form = ISBNForm()
        # return redirect('llbook')
    return render(request, 'librarian/add_book_isbn.html', {'form': form})

@login_required
def labookocr_form(request):
    return render(request, 'librarian/add_book_cover.html')

@login_required
def labook_ocr(request):
    print("🔍 Đang vào view ladd_book_ocr với path:", request.path)

    if request.method != 'POST':
        messages.warning(request, "Phương thức không hợp lệ. Vui lòng dùng form để gửi ảnh bìa.")
        return redirect('ladd_book_ocr')

    if 'cover' not in request.FILES:
        messages.error(request, 'Không có ảnh bìa được tải lên.')
        return redirect('ladd_book_ocr')

    file = request.FILES['cover']

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            for chunk in file.chunks():
                temp_file.write(chunk)
            temp_path = temp_file.name

        img = cv2.imread(temp_path)
        img = noise_removal(cv2.bitwise_not(img))
        processed = preprocess_image(thick_font(img))

        ocr = PaddleOCR(
            lang="vi",
            det_db_box_thresh=0.3,
            use_angle_cls=True,
            rec_algorithm="SVTR_LCNet",
            rec_image_shape="3, 32, 320",
            rec_batch_num=6,
            max_text_length=50
        )
        results = ocr.ocr(processed, cls=True)
        ocr_text = "\n".join([word_info[1][0] for line in results for word_info in line])

    except Exception as e:
        print("❌ Lỗi xử lý ảnh OCR:", e)
        messages.error(request, f"Lỗi xử lý ảnh OCR: {e}")
        return redirect('ladd_book_ocr')

    try:
        response = requests.post(
            "https://zep.hcmute.fit/7889/extract_book_info",
            json={"ocr_text": ocr_text},
            timeout=20  # để tránh treo
        )
        response.raise_for_status()
        data = response.json()
        title = data.get("title", "Không rõ")
        author = data.get("author", "Không rõ")

    except Exception as e:
        print("❌ Lỗi khi gọi LLM server:", e)
        messages.error(request, f"Lỗi khi gọi server LLM: {e}")
        return redirect("ladd_book_ocr")

    return render(request, "librarian/confirm_book_info.html", {
        "title": title,
        "author": author,
        "ocr_text": ocr_text,
        "image_path": file,
    })

class LBookListView(LoginRequiredMixin,ListView):
    model = Book
    template_name = 'librarian/book_list.html'
    context_object_name = 'books'
    paginate_by = 3

    def get_queryset(self):
        return Book.objects.order_by('-id')


class LManageBook(LoginRequiredMixin,ListView):
    model = Book
    template_name = 'librarian/manage_books.html'
    context_object_name = 'books'
    paginate_by = 3

    def get_queryset(self):
        return Book.objects.order_by('-id')


class LDeleteRequest(LoginRequiredMixin,ListView):
    model = DeleteRequest
    template_name = 'librarian/delete_request.html'
    context_object_name = 'feedbacks'
    paginate_by = 3

    def get_queryset(self):
        return DeleteRequest.objects.order_by('-id')


class LViewBook(LoginRequiredMixin,DetailView):
    model = Book
    template_name = 'librarian/book_detail.html'

    
class LEditView(LoginRequiredMixin,UpdateView):
    model = Book
    form_class = BookForm
    template_name = 'librarian/edit_book.html'
    success_url = reverse_lazy('lmbook')
    success_message = 'Data was updated successfully'


class LDeleteView(LoginRequiredMixin,DeleteView):
    model = Book
    template_name = 'librarian/confirm_delete.html'
    success_url = reverse_lazy('lmbook')
    success_message = 'Data was deleted successfully'


class LDeleteBook(LoginRequiredMixin,DeleteView):
    model = Book
    template_name = 'librarian/confirm_delete2.html'
    success_url = reverse_lazy('librarian')
    success_message = 'Data was dele successfully'



@login_required
def lsearch(request):
    query = request.GET['query']
    print(type(query))


    #data = query.split()
    data = query
    print(len(data))
    if( len(data) == 0):
        return redirect('publisher')
    else:
                a = data

                # Searching for It
                qs5 =models.Book.objects.filter(id__iexact=a).distinct()
                qs6 =models.Book.objects.filter(id__exact=a).distinct()

                qs7 =models.Book.objects.all().filter(id__contains=a)
                qs8 =models.Book.objects.select_related().filter(id__contains=a).distinct()
                qs9 =models.Book.objects.filter(id__startswith=a).distinct()
                qs10 =models.Book.objects.filter(id__endswith=a).distinct()
                qs11 =models.Book.objects.filter(id__istartswith=a).distinct()
                qs12 =models.Book.objects.all().filter(id__icontains=a)
                qs13 =models.Book.objects.filter(id__iendswith=a).distinct()




                files = itertools.chain(qs5, qs6, qs7, qs8, qs9, qs10, qs11, qs12, qs13)

                res = []
                for i in files:
                    if i not in res:
                        res.append(i)


                # word variable will be shown in html when user click on search button
                word="Searched Result :"
                print("Result")

                print(res)
                files = res




                page = request.GET.get('page', 1)
                paginator = Paginator(files, 10)
                try:
                    files = paginator.page(page)
                except PageNotAnInteger:
                    files = paginator.page(1)
                except EmptyPage:
                    files = paginator.page(paginator.num_pages)
   


                if files:
                    return render(request,'librarian/result.html',{'files':files,'word':word})
                return render(request,'librarian/result.html',{'files':files,'word':word})


class LCreateChat(LoginRequiredMixin, CreateView):
    form_class = ChatForm
    model = Chat
    template_name = 'librarian/chat_form.html'
    success_url = reverse_lazy('llchat')


    def form_valid(self, form):
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        self.object.save()
        return super().form_valid(form)




class LListChat(LoginRequiredMixin, ListView):
    model = Chat
    template_name = 'librarian/chat_list.html'

    def get_queryset(self):
        return Chat.objects.filter(posted_at__lt=timezone.now()).order_by('posted_at')














# Admin views

def dashboard(request):
    book = Book.objects.all().count()
    user = User.objects.all().count()

    context = {'book':book, 'user':user}

    return render(request, 'dashboard/home.html', context)

def create_user_form(request):
    choice = ['1', '0', 'Publisher', 'Admin', 'Librarian']
    choice = {'choice': choice}

    return render(request, 'dashboard/add_user.html', choice)


class ADeleteUser(SuccessMessageMixin, DeleteView):
    model = User
    template_name='dashboard/confirm_delete3.html'
    success_url = reverse_lazy('aluser')
    success_message = "Data successfully deleted"


class AEditUser(SuccessMessageMixin, UpdateView): 
    model = User
    form_class = UserForm
    template_name = 'dashboard/edit_user.html'
    success_url = reverse_lazy('aluser')
    success_message = "Data successfully updated"

class ListUserView(generic.ListView):
    model = User
    template_name = 'dashboard/list_users.html'
    context_object_name = 'users'
    paginate_by = 4

    def get_queryset(self):
        return User.objects.order_by('-id')

def create_user(request):
    choice = ['1', '0', 'Publisher', 'Admin', 'Librarian', 'Student']
    choice = {'choice': choice}
    if request.method == 'POST':
            first_name=request.POST['first_name']
            last_name=request.POST['last_name']
            username=request.POST['username']
            userType=request.POST['userType']
            email=request.POST['email']
            password=request.POST['password']
            password = make_password(password)
            print("User Type")
            print(userType)
            if userType == "Publisher":
                a = User(first_name=first_name, last_name=last_name, username=username, email=email, password=password, is_publisher=True)
                a.save()
                messages.success(request, 'Member was created successfully!')
                return redirect('aluser')
            elif userType == "Admin":
                a = User(first_name=first_name, last_name=last_name, username=username, email=email, password=password, is_admin=True)
                a.save()
                messages.success(request, 'Member was created successfully!')
                return redirect('aluser')
            elif userType == "Librarian":
                a = User(first_name=first_name, last_name=last_name, username=username, email=email, password=password, is_librarian=True)
                a.save()
                messages.success(request, 'Member was created successfully!')
                return redirect('aluser') 
            elif userType == "Student":
                a = User(first_name=first_name, last_name=last_name, username=username, email=email, password=password, is_student=True)
                a.save()
                messages.success(request, 'Member was created successfully!')
                return redirect('aluser')   
            else:
                messages.success(request, 'Member was not created')
                return redirect('create_user_form')
    else:
        return redirect('create_user_form')


class ALViewUser(DetailView):
    model = User
    template_name='dashboard/user_detail.html'



class ACreateChat(LoginRequiredMixin, CreateView):
    form_class = ChatForm
    model = Chat
    template_name = 'dashboard/chat_form.html'
    success_url = reverse_lazy('alchat')


    def form_valid(self, form):
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        self.object.save()
        return super().form_valid(form)




class AListChat(LoginRequiredMixin, ListView):
    model = Chat
    template_name = 'dashboard/chat_list.html'

    def get_queryset(self):
        return Chat.objects.filter(posted_at__lt=timezone.now()).order_by('posted_at')


@login_required
def aabook_form(request):
    return render(request, 'dashboard/add_book.html')


@login_required
def aabook(request):
    if request.method == 'POST':
        title = request.POST['title']
        author = request.POST['author']
        year = request.POST['year']
        publisher = request.POST['publisher']
        desc = request.POST['desc']
        cover = request.FILES['cover']
        pdf = request.FILES['pdf']
        current_user = request.user
        user_id = current_user.id
        username = current_user.username

        a = Book(title=title, author=author, year=year, publisher=publisher, 
            desc=desc, cover=cover, pdf=pdf, uploaded_by=username, user_id=user_id)
        a.save()
        messages.success(request, 'Book was uploaded successfully')
        return redirect('albook')
    else:
        messages.error(request, 'Book was not uploaded successfully')
        return redirect('aabook_form')


class ABookListView(LoginRequiredMixin,ListView):
    model = Book
    template_name = 'dashboard/book_list.html'
    context_object_name = 'books'
    paginate_by = 3

    def get_queryset(self):
        return Book.objects.order_by('-id')




class AManageBook(LoginRequiredMixin,ListView):
    model = Book
    template_name = 'dashboard/manage_books.html'
    context_object_name = 'books'
    paginate_by = 3

    def get_queryset(self):
        return Book.objects.order_by('-id')




class ADeleteBook(LoginRequiredMixin,DeleteView):
    model = Book
    template_name = 'dashboard/confirm_delete2.html'
    success_url = reverse_lazy('ambook')
    success_message = 'Data was dele successfully'


class ADeleteBookk(LoginRequiredMixin,DeleteView):
    model = Book
    template_name = 'dashboard/confirm_delete.html'
    success_url = reverse_lazy('dashboard')
    success_message = 'Data was dele successfully'


class AViewBook(LoginRequiredMixin,DetailView):
    model = Book
    template_name = 'dashboard/book_detail.html'




class AEditView(LoginRequiredMixin,UpdateView):
    model = Book
    form_class = BookForm
    template_name = 'dashboard/edit_book.html'
    success_url = reverse_lazy('ambook')
    success_message = 'Data was updated successfully'




class ADeleteRequest(LoginRequiredMixin,ListView):
    model = DeleteRequest
    template_name = 'dashboard/delete_request.html'
    context_object_name = 'feedbacks'
    paginate_by = 3

    def get_queryset(self):
        return DeleteRequest.objects.order_by('-id')



class AFeedback(LoginRequiredMixin,ListView):
    model = Feedback
    template_name = 'dashboard/feedback.html'
    context_object_name = 'feedbacks'
    paginate_by = 3

    def get_queryset(self):
        return Feedback.objects.order_by('-id')



@login_required
def asearch(request):
    query = request.GET['query']
    print(type(query))


    #data = query.split()
    data = query
    print(len(data))
    if( len(data) == 0):
        return redirect('dashborad')
    else:
                a = data

                # Searching for It
                qs5 =models.Book.objects.filter(id__iexact=a).distinct()
                qs6 =models.Book.objects.filter(id__exact=a).distinct()

                qs7 =models.Book.objects.all().filter(id__contains=a)
                qs8 =models.Book.objects.select_related().filter(id__contains=a).distinct()
                qs9 =models.Book.objects.filter(id__startswith=a).distinct()
                qs10 =models.Book.objects.filter(id__endswith=a).distinct()
                qs11 =models.Book.objects.filter(id__istartswith=a).distinct()
                qs12 =models.Book.objects.all().filter(id__icontains=a)
                qs13 =models.Book.objects.filter(id__iendswith=a).distinct()




                files = itertools.chain(qs5, qs6, qs7, qs8, qs9, qs10, qs11, qs12, qs13)

                res = []
                for i in files:
                    if i not in res:
                        res.append(i)


                # word variable will be shown in html when user click on search button
                word="Searched Result :"
                print("Result")

                print(res)
                files = res




                page = request.GET.get('page', 1)
                paginator = Paginator(files, 10)
                try:
                    files = paginator.page(page)
                except PageNotAnInteger:
                    files = paginator.page(1)
                except EmptyPage:
                    files = paginator.page(paginator.num_pages)
   


                if files:
                    return render(request,'dashboard/result.html',{'files':files,'word':word})
                return render(request,'dashboard/result.html',{'files':files,'word':word})

@login_required
def recommend_books(request):
    user = request.user
    interactions = UserBookInteraction.objects.filter(user=user).order_by('-timestamp')[:10]
    if not interactions:
        return render(request, 'recommendation.html', {'books': []})

    book_embeddings = [get_book_embedding(inter.book) for inter in reversed(interactions)]
    input_tensor = torch.tensor(np.stack(book_embeddings)).unsqueeze(1)

    model = RecTransformer(n_books=Book.objects.count())
    model.load_state_dict(torch.load('bookstore/recommend/model.pt'))
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)
        top_indices = torch.topk(logits[0], 5).indices.tolist()

    all_books = list(Book.objects.all())
    recommended_books = [all_books[i] for i in top_indices]

    return render(request, 'recommendation.html', {'books': recommended_books})

def book_grid_view(request):
    # Lấy tất cả sách hoặc lọc theo thể loại nếu có
    category = request.GET.get('category')
    if category:
        books = Book.objects.filter(category=category)
    else:
        books = Book.objects.all()
    # Lấy danh sách các thể loại (giả sử Book có trường category)
    categories = Book.objects.values_list('category', flat=True).distinct()
    return render(request, 'bookstore/book_grid.html', {
        'books': books,
        'categories': categories,
        'selected_category': category,
    })
















