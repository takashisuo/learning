from django.views.generic import ListView, DetailView
from django.core.exceptions import PermissionDenied
from blog.models import Post

class PostList(ListView):
    model = Post
    context_object_name = "posts"

class PostDetail(DetailView):
    model = Post
    context_object_name = "post"

    def get_object(self):
        post = super().get_object()
        if post.is_published or self.request.user.is_authenticated:
            return post
        else:
            raise PermissionDenied
