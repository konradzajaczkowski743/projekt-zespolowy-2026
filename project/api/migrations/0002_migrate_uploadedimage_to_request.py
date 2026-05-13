from __future__ import annotations

import django.db.models.deletion
from django.db import migrations, models


def forwards(apps, schema_editor):
    ImageAnalysisRequest = apps.get_model("api", "ImageAnalysisRequest")
    UploadedImage = apps.get_model("api", "UploadedImage")
    AnalysisResult = apps.get_model("api", "AnalysisResult")

    db_alias = schema_editor.connection.alias

    # Copy uploaded image metadata into the parent ImageAnalysisRequest
    for ui in UploadedImage.objects.using(db_alias).all():
        try:
            req = ImageAnalysisRequest.objects.using(db_alias).get(pk=ui.request_id)
        except ImageAnalysisRequest.DoesNotExist:
            continue

        # Copy simple fields; allow overwriting only when empty on the request
        if not getattr(req, "original_filename", None):
            req.original_filename = ui.original_filename
        if getattr(req, "file_size", None) is None:
            req.file_size = ui.file_size
        if getattr(req, "width", None) is None:
            req.width = ui.width
        if getattr(req, "height", None) is None:
            req.height = ui.height
        if getattr(req, "latitude", None) is None:
            req.latitude = ui.latitude
        if getattr(req, "longitude", None) is None:
            req.longitude = ui.longitude
        # JSON fields: prefer existing request value if present
        if not getattr(req, "exif_data", None):
            req.exif_data = ui.exif_data or {}

        # Copy file path (ImageField stores a string path in the DB)
        if not getattr(req, "file", None):
            req.file = ui.file

        req.save()

    # Backfill AnalysisResult.request from the related UploadedImage -> request
    for ar in AnalysisResult.objects.using(db_alias).all():
        image_id = getattr(ar, "image_id", None)
        if not image_id:
            continue
        try:
            ui = UploadedImage.objects.using(db_alias).get(pk=image_id)
        except UploadedImage.DoesNotExist:
            continue

        if getattr(ui, "request_id", None):
            ar.request_id = ui.request_id
            ar.save(update_fields=["request"])


def reverse(apps, schema_editor):
    # Reverse is intentionally left as a no-op because restoring the previous
    # UploadedImage rows from ImageAnalysisRequest would be lossy in the general case.
    return


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0001_initial"),
    ]

    operations = [
        # Add new fields to ImageAnalysisRequest (nullable/defaulted so migration is safe)
        migrations.AddField(
            model_name="imageanalysisrequest",
            name="original_filename",
            field=models.CharField(blank=True, default="", help_text="The original filename for reference.", max_length=255),
        ),
        migrations.AddField(
            model_name="imageanalysisrequest",
            name="file_size",
            field=models.PositiveIntegerField(blank=True, help_text="File size in bytes.", null=True),
        ),
        migrations.AddField(
            model_name="imageanalysisrequest",
            name="width",
            field=models.IntegerField(blank=True, help_text="Automated pixel width.", null=True),
        ),
        migrations.AddField(
            model_name="imageanalysisrequest",
            name="height",
            field=models.IntegerField(blank=True, help_text="Automated pixel height.", null=True),
        ),
        migrations.AddField(
            model_name="imageanalysisrequest",
            name="latitude",
            field=models.FloatField(blank=True, help_text="Geospatial latitude coordinate.", null=True),
        ),
        migrations.AddField(
            model_name="imageanalysisrequest",
            name="longitude",
            field=models.FloatField(blank=True, help_text="Geospatial longitude coordinate.", null=True),
        ),
        migrations.AddField(
            model_name="imageanalysisrequest",
            name="exif_data",
            field=models.JSONField(blank=True, default=dict, help_text="Extracted EXIF metadata in JSON format."),
        ),
        migrations.AddField(
            model_name="imageanalysisrequest",
            name="file",
            field=models.ImageField(blank=True, help_text="The physical image file stored on the local filesystem.", height_field="height", upload_to="analysis_images/%Y/%m/%d/", width_field="width"),
        ),

        # Add nullable request field to AnalysisResult to allow backfill
        migrations.AddField(
            model_name="analysisresult",
            name="request",
            field=models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name="result", to="api.ImageAnalysisRequest"),
        ),

        # Run backfill to populate the new fields from UploadedImage
        migrations.RunPython(forwards, reverse),

        # Make the request relation required (non-nullable)
        migrations.AlterField(
            model_name="analysisresult",
            name="request",
            field=models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name="result", to="api.ImageAnalysisRequest"),
        ),

        # Remove the old image relation and drop the UploadedImage model
        migrations.RemoveField(
            model_name="analysisresult",
            name="image",
        ),
        migrations.DeleteModel(
            name="UploadedImage",
        ),
    ]
