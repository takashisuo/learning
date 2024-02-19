from django.views.generic import ListView
from blog.models import Post

class PostList(ListView):
    model = Post
    