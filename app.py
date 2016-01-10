import markovify
import requests
import nltk
import random
import re
import os
from flask import Flask
from bs4 import BeautifulSoup

#Constants
port = int(os.environ.get('PORT', 5000))
api = "http://shesahomewrecker.com/api/infinity-scroll/?query=home&page="
PAGES_TO_TRAIN = 2
TOTAL_PAGES = 900

#Install Natural Language data
print("Downloading setup...")
nltk.download('averaged_perceptron_tagger')
print("Done.")

class POSifiedText(markovify.Text):
	def word_split(self, sentence):
		words = re.split(self.word_split_pattern, sentence)
		words = [ "::".join(tag) for tag in nltk.pos_tag(words) ]
		return words

	def word_join(self, words):
		sentence = " ".join(word.split("::")[0] for word in words)
		return sentence


def getQueryUrl(page):
	return api + str(page)

#Returns an array of paragraphs
def getPageText(page):
	html = requests.get(getQueryUrl(page)).text
	soup = BeautifulSoup(html, 'html.parser')
	tags = soup.find_all(class_='post-text')
	return [el.text.strip() for el in tags]

print("Starting setup...")

# Setup
pages = []
for i in range(0, PAGES_TO_TRAIN):
	page_num = random.randrange(0, TOTAL_PAGES)
	print("	Downloading page " + str(page_num))
	pages.extend(getPageText(page_num))
pages = [page for page in pages if page is not None and len(page) > 0]
text = " ".join(pages)
print("	Training...")
text_model = POSifiedText(text)
print("	Done. Starting app.")

# Web stuff
app = Flask(__name__)
@app.route("/")
def hello():
	texts = [text_model.make_sentence() for i in range(5)]
	texts = [sentence for sentence in texts if sentence is not None and len(sentence) > 0]
	return " ".join(texts)

#Run if debugging
if __name__ == "__main__":
	app.run(debug=True)
	print("Listening on port " + str(port))