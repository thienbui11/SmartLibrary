import random
from datetime import timedelta
from django.utils import timezone
from django.core.management.base import BaseCommand
from bookstore.models import User, Book, UserBookInteraction

class Command(BaseCommand):
    help = 'Generate 100,000 random user-book interactions'

    def handle(self, *args, **options):
        users = list(User.objects.all().values_list('id', flat=True))
        books = list(Book.objects.all().values_list('id', flat=True))
        actions = ['view', 'ask', 'download']
        now = timezone.now()

        if not users or not books:
            self.stdout.write(self.style.ERROR("❌ Không có user hoặc book trong database."))
            return

        interactions = []
        for _ in range(100_000):
            user_id = random.choice(users)
            book_id = random.choice(books)
            action = random.choice(actions)
            # Chọn thời điểm trong 2 năm qua
            random_days = random.randint(0, 730)
            random_seconds = random.randint(0, 86400)
            timestamp = now - timedelta(days=random_days, seconds=random_seconds)

            interaction = UserBookInteraction(
                user_id=user_id,
                book_id=book_id,
                action=action,
                timestamp=timestamp
            )
            interactions.append(interaction)

            # Bulk insert mỗi 5000 dòng để tránh overload RAM
            if len(interactions) >= 5000:
                UserBookInteraction.objects.bulk_create(interactions)
                interactions.clear()
                self.stdout.write(self.style.SUCCESS("✔️ Đã thêm 5000 interactions..."))

        # Thêm các dòng còn lại
        if interactions:
            UserBookInteraction.objects.bulk_create(interactions)

        self.stdout.write(self.style.SUCCESS("✅ Đã tạo xong 100,000 tương tác!"))
