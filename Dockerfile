FROM python:3.6

#RUN mkdir /code
#WORKDIR /code

# ARG on_local

RUN git clone https://da93a600a1764d0ab1d9f08a69a62edc174b9aa6@github.com/blancaag/ss_image_class.git
RUN cd ss_image_class

ADD . .

RUN pip install -r requirements.txt
WORKDIR src/scripts/
RUN python3 test.py

EXPOSE 9090
CMD ["python3", "src/scripts/test.py"]
