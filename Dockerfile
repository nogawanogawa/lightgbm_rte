FROM ubuntu:20.04

RUN apt update
RUN apt install -y python3 python3-pip

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

RUN apt -y install mecab libmecab-dev mecab-ipadic-utf8
RUN cp /etc/mecabrc /usr/local/etc/
ENV USERNAME user

WORKDIR /home
