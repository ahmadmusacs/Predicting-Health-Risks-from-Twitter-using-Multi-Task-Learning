FROM python:3

ADD main.py /

ADD data /data/

ADD result /result/

RUN pip install nltk

RUN pip install scikit-learn

CMD [ "python", "./main.py" ]