FROM python:3.6

#RUN mkdir /code
#WORKDIR /code

ADD . .

# ARG on_local

RUN pip install -r requirements.txt
WORKDIR src/scripts/
RUN python3 test.py

EXPOSE 9090
CMD ["python", "/code/app.py"]
