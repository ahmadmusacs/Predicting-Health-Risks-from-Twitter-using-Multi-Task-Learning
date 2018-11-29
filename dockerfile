FROM python:3

ADD my_script.py /
ADD TestSet.jsonl /
ADD TestSet3.jsonl /
ADD TestSet4.jsonl /
ADD TestSet5.jsonl /
ADD TrainSetNew2.jsonl /
ADD TrainSetNew3.jsonl /
ADD TrainSetNew4.jsonl /
ADD TrainSetNew5.jsonl /
ADD TrainedModelOnBothDataBCE1.pt /
ADD TrainedModelOnBothDataBCE2.pt /
ADD TrainedModelOnBothDataBCE3.pt /
ADD TrainedModelOnBothDataBCE4.pt /
ADD TrainedModelOnFood1W.pt /
ADD TrainedModelOnFood2W.pt /
ADD TrainedModelOnFood3W.pt /
ADD TrainedModelOnFood4W.pt /
ADD TrainSetFiltered.jsonl /
ADD TrainSetFiltered2.jsonl /
ADD TrainSetFiltered3.jsonl /
ADD TrainSetFiltered4.jsonl /
ADD TestSetFiltered.jsonl /
ADD TestSetFiltered2.jsonl /
ADD TestSetFiltered3.jsonl /
ADD TestSetFiltered4.jsonl /


RUN pip3 install torch torchvision

RUN pip install spacy

RUN pip install nltk

RUN pip install torchtext

RUN pip install jsonlines

RUN python -m spacy download en

ENTRYPOINT [ "python", "./my_script.py" ]