# Docker file for a slim Ubuntu-based Python3 image

FROM ubuntu:latest
MAINTAINER fnndsc "dev@babymri.org"

RUN apt-get update \
  && apt-get install -y apt-utils \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip \
  && apt-get install -y git

RUN echo 'test0'

# RUN git init
# RUN git clone https://github.com/blancaag/ss_image_class.git
# RUN git clone https://da93a600a1764d0ab1d9f08a69a62edc174b9aa6@github.com/blancaag/ss_image_class.git
# RUN cd ss_image_class
RUN pip3 install -r requirements.txt

# WORKDIR
# ADD . /ss_image_class

RUN echo 'test1'

RUN python3 /src/scripts/validation.py

ENTRYPOINT ["python3"]
