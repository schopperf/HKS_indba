import nltk
import bs4
import requests

url = "https://www.heise.de/newsticker/meldung/Jugendliche-lieben-Netflix-und-WhatsApp-keiner-mag-Facebook-4234532.html"
page = requests.get(url)
soup = bs4.BeautifulSoup(page.content, 'html.parser')

## do this to see the sites html.
## print(soup.prettify())

a = soup.findAll('p')
for b in a:
    print(b.getText())