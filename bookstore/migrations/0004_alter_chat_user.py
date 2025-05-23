# Generated by Django 5.0.14 on 2025-05-16 09:08

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("bookstore", "0003_alter_book_title"),
    ]

    operations = [
        migrations.AlterField(
            model_name="chat",
            name="user",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="bookstore_chats",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
    ]
