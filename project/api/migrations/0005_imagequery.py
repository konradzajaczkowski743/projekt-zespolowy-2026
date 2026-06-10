# Generated migration for ImageQuery model

from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0004_analysisresult_distance_estimation'),
    ]

    operations = [
        migrations.CreateModel(
            name='ImageQuery',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, help_text='Unique identifier for this query.', primary_key=True, serialize=False)),
                ('user_query', models.TextField(help_text="The user's question about the image.")),
                ('ai_response', models.TextField(blank=True, default='', help_text='The AI-generated response to the user\'s question.')),
                ('status', models.CharField(choices=[('PENDING', 'Pending'), ('COMPLETED', 'Completed'), ('FAILED', 'Failed')], db_index=True, default='PENDING', help_text='Current state of the query processing.', max_length=20)),
                ('error_message', models.TextField(blank=True, default='', help_text='Error details if the query processing failed.')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('result', models.ForeignKey(help_text='The analysis result this query refers to.', on_delete=django.db.models.deletion.CASCADE, related_name='queries', to='api.analysisresult')),
            ],
            options={
                'verbose_name': 'Image Query',
                'verbose_name_plural': 'Image Queries',
                'ordering': ['-created_at'],
            },
        ),
    ]
