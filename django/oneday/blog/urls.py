from django.urls import path
from blog.views import PostList

urlpatterns = [
    path("", PostList.as_view(), name="post_list"),
]