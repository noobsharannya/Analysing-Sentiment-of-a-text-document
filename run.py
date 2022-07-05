from flask import Flask, render_template, flash, request, redirect, url_for
import os
from matplotlib import pyplot as plt
import docx2txt
import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pathlib import Path
import aspose.words as aw
import joblib
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


def extract_text_by_page(pdf_path):

    with open(pdf_path, 'rb') as fh:

        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):

            resource_manager = PDFResourceManager()
            fake_file_handle = io.StringIO()

            converter = TextConverter(resource_manager,
                                      fake_file_handle)

            page_interpreter = PDFPageInterpreter(resource_manager,
                                                  converter)

            page_interpreter.process_page(page)
            text = fake_file_handle.getvalue()

            yield text

            # close open handles
            converter.close()
            fake_file_handle.close()


def extract_text(pdf_path):
    text = ""
    for page in extract_text_by_page(pdf_path):
        # print(page)
        text += page
        text += '\n'
    return text


def convertToText(filepath):
    if filepath[-4:] == '.doc':
        text = docx2txt.process(filepath)
        with open("Output.txt", "w") as text_file:
            print(text, file=text_file)
    elif filepath[-4:] == '.docx':
        text = docx2txt.process(filepath)
        with open("Output.txt", "w") as text_file:
            print(text, file=text_file)
    elif filepath[-4:] == ".pdf":
        text = extract_text(filepath)
        with Path('Output.txt').open(mode="w", encoding="utf-8") as output_file:
            output_file.write(text)
    elif filepath[-4:] == ".rtf":
        doc = aw.Document(filepath)
        doc.save("Output.txt")
    elif filepath[-4:] == ".txt":
        with open(filepath) as f:
            lines = f.readlines()
        text = ""
        for line in lines:
            text += line
            text += " "
        with Path('Output.txt').open(mode="w", encoding="utf-8") as output_file:
            output_file.write(text)


stop_words = set(stopwords.words('english'))


def preprocess(s):
    s2 = []
    for i in str(s):
        if not i.isdigit():
            s2.append(i)
    s = "".join(s2)
    s2 = []
    s = s.lower()
    s = BeautifulSoup(s).get_text(strip=True)
    s = re.sub(r"http\S+", "", s)
    for x in s:
        if((x >= 'A') and (x <= 'Z')) or ((x >= 'a') and (x <= 'z')) or (x == ' '):
            s2.append(x)
    s = "".join(s2)
    s = re.sub(" \s+", " ", s)
    s2 = []
    negation = ['nor', 'no', 'not']
    important = ['but', 'against', 'between',
                 'too', 'just', 'more', 'most', 'than']
    opposite = False
    for x in s.split():
        if (x not in stop_words):
            if opposite:
                syns = wordnet.synsets(x)
                flag = False
                for syn in wordnet.synsets("good"):
                    for l in syn.lemmas():
                        if l.antonyms():
                            s2.append(l.antonyms()[0].name())
                            flag = True
                            break
                    if(flag):
                        break
                opposite = False
            else:
                s2.append(x)
        elif (x in negation):
            opposite = True
        elif (x in important):
            s2.append(x)
    s = " ".join(s2)
    s2 = []
    lemma = WordNetLemmatizer()
    for x in s.split():
        s2.append(lemma.lemmatize(x))
    s = " ".join(s2)
    return s


def analyse():
    with open('Output.txt') as f:
        lines = f.readlines()
        text = ""
        for line in lines:
            text += line
            text += " "
        positive = 0
        negative = 0
        neutral = 0
        overall = 0.0
        num = 0
        sentences = text.split('.')
        sentence_preprocessed = []
        vectorizer = joblib.load('vectorizer.pkl')
        sentiment = joblib.load('model.pkl')
        for s in sentences:
            num = num + 1
            sentence_preprocessed.append(preprocess(s))
        features = vectorizer.transform(sentence_preprocessed)
        prediction = sentiment.predict(features)
        for p in prediction:
            if(p == 1):
                positive = positive+1
            elif(p == -1):
                negative = negative+1
            else:
                neutral = neutral+1
            overall = overall+p
        overall = overall/num
        plt.style.use("fivethirtyeight")
        slices = [positive, neutral, negative]
        labels = ['positive',
                  'neutral', 'negative']
        explode = [0.05, 0, 0.1]
        colors = ['#9ACD32', '#1E90FF', '#FFA500']
        plt.figure(0)
        plt.pie(slices, labels=labels, colors=colors, explode=explode, shadow=True, startangle=90,
                autopct='%1.1f%%', wedgeprops={'edgecolor': 'black'})
        plt.title("Sentiment-Distribution")
    plt.tight_layout()
    dir = os.getcwd()
    file1 = dir+'/static/images/pie.png'
    plt.savefig(file1)
    plt.close()
    sent_x = [-1, 0, 1]
    num_y = [negative, neutral, positive, ]
    plt.figure(1)
    plt.bar(sent_x, num_y, color='maroon', width=0.4)
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Number of sentences')
    plt.tight_layout()
    file2 = dir+'/static/images/bar.png'
    plt.savefig(file2)
    plt.close()
    if overall > 0.33:
        return 'The document contains positive text'
    else:
        if overall < -0.33:
            return 'The document contains negative text'
        else:
            return 'The document contains neutral text'


app = Flask(__name__)
app.config['SECRET_KEY'] = '638f47d377ec9b6a8884d0246a4e9031'
app.config['FILE_UPLOADS'] = os.getcwd()

picfolder = os.path.join('static', 'images')
app.config['UPLOAD_FOLDER'] = picfolder


@app.route("/", methods=['GET', 'POST'])
@app.route("/upload-files", methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        if request.files:
            document = request.files['document']
            document.save(os.path.join(
                app.config['FILE_UPLOADS'], document.filename))
            convertToText(document.filename)
            message = analyse()
            flash(message, 'success')
            return redirect(url_for('analysis'))
    return render_template('newform.html')


@app.route("/analysis")
def analysis():
    bar = os.path.join(app.config['UPLOAD_FOLDER'], 'bar.png')
    pie = os.path.join(app.config['UPLOAD_FOLDER'], 'pie.png')
    return render_template('results.html', bar_chart=bar, pie_chart=pie)


if __name__ == '__main__':
    app.run(debug=True)
