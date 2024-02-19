from django.urls import path
from blog.views import PostList, PostDetail

urlpatterns = [
    path("", PostList.as_view(), name="post_list"),
    path("post/<int:pk>/", PostDetail.as_view(), name="post_detail")
]