import requests

def fetch_book_info_from_isbn(isbn):
    url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}&key=AIzaSyD8x9l7CQEpc0OTyBfX_9nNnkAbiaEX2gk"
    response = requests.get(url)
    data = response.json()
    if "items" not in data or len(data["items"]) == 0:
        return None

    volume_info = data["items"][0]["volumeInfo"]

    title = volume_info.get("title")
    authors = volume_info.get("authors", []),
    publishers = volume_info.get("publisher"),
    year = volume_info.get("publishedDate"),

    return {
        'title': title,
        'author': authors,
        'publisher': publishers,
        'year': year,
        'desc': volume_info.get("description"),
    }
