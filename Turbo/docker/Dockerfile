FROM python3

RUN pip install -r requirements.txt

COPY ICTR.py /opt/ml/code/ICTR.py

ENV SAGEMAKER_PROGRAM ICTR.py