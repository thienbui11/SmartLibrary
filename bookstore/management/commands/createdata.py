import os
import glob
from django.core.files import File
from django.core.management.base import BaseCommand
from bookstore.models import Book, Department

class Command(BaseCommand):
    help = 'Create book data from PDF and cover files'

    def handle(self, *args, **options):
        PDF_FOLDER = 'data/pdfs/'
        COVER_FOLDER = 'data/covers/'

        pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]

        for pdf_file in pdf_files:
            # Chuyển 'Cong_nghe_so.pdf' -> 'Cong nghe so'
            title_raw = pdf_file[:-4]
            title = title_raw.replace('_', ' ').strip()

            pdf_path = os.path.join(PDF_FOLDER, pdf_file)

            # Tìm ảnh bìa có tên giống file pdf (bỏ đuôi)
            cover_pattern = os.path.join(COVER_FOLDER, f"{title_raw}.*")
            cover_matches = glob.glob(cover_pattern)

            if not cover_matches:
                self.stdout.write(self.style.ERROR(f"❌ Không tìm thấy ảnh bìa cho: {title}"))
                continue

            cover_path = cover_matches[0]

            # Khởi tạo chỉ với title
            book = Book(title=title)

            # Gán file PDF và ảnh bìa
            try:
                with open(pdf_path, 'rb') as f_pdf:
                    book.pdf.save(os.path.basename(pdf_path), File(f_pdf), save=False)
            except FileNotFoundError:
                print(f"❌ File not found: {pdf_path}")
                continue

            try:
                with open(cover_path, 'rb') as f_img:
                    book.cover.save(os.path.basename(cover_path), File(f_img), save=False)
            except FileNotFoundError:
                self.stdout.write(self.style.ERROR(f"❌ Cover file not found: {cover_path}"))
                continue

            book.save()
            self.stdout.write(self.style.SUCCESS(f"✅ Đã thêm sách: {title}")) 