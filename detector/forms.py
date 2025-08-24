from django import forms

class VideoUploadForm(forms.Form):
    video = forms.FileField(
        required=True,
        label="Upload a video",
        widget=forms.ClearableFileInput(attrs={"accept": "video/*"})
    )