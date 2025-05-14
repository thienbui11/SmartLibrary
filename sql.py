from datetime import datetime

password_hash = "pbkdf2_sha256$260000$vx0Zp3nQ3gKrFWnAlRT8UQ$AwljHUp5OJfXyN3SVR8lsn7qYH9AYMiRPLvqOZ1SshI="
now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print("INSERT INTO auth_user_extended (id, password, last_login, is_superuser, username, first_name, last_name, email, is_staff, is_active, date_joined, is_admin, is_publisher, is_librarian, is_student, face_embedding, qr_code) VALUES")

for i in range(1, 201):
    uid = f"user{i:03}"
    email = f"{uid}@example.com"
    is_admin = 1 if i <= 20 else 0
    is_superuser = is_admin
    is_staff = is_admin
    is_student = 0 if is_admin else 1

    values = (
        i,
        f"'{password_hash}'",
        'NULL',
        is_superuser,
        f"'{uid}'",
        f"'User{i}'",
        "'Test'",
        f"'{email}'",
        is_staff,
        1,  # is_active
        f"'{now}'",
        is_admin,
        0,  # is_publisher
        0,  # is_librarian
        is_student,
        'NULL',
        'NULL'
    )

    end = ',' if i < 200 else ';'
    print(f"({', '.join(map(str, values))}){end}")
