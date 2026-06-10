from django.contrib import admin
from .models import ImageAnalysisRequest, AnalysisResult

@admin.register(ImageAnalysisRequest)
class ImageAnalysisRequestAdmin(admin.ModelAdmin):
    list_display = ("id", "status", "original_filename", "created_at", "completed_at")
    list_filter = ("status", "created_at")
    search_fields = ("id", "original_filename")
    readonly_fields = ("id", "created_at", "updated_at", "width", "height", "file_size")

@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = ("request", "is_deepfake", "processing_time_ms", "created_at")
    list_filter = ("is_deepfake", "created_at")
    search_fields = ("request__id", "error_message")
    readonly_fields = ("created_at",)
