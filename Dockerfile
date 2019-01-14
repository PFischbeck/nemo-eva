FROM python:3.6-alpine3.7 as nemo-eva

COPY requirements.txt /requirements.txt

RUN apk --no-cache add --virtual build-deps \
        bash g++ freetype-dev gfortran lapack-dev \
        libpng-dev libexecinfo-dev make ncurses-dev readline-dev cmake \
 && cat requirements.txt | xargs -n 1 pip3 install \
 && apk del build-deps

RUN apk --no-cache add make libgomp libstdc++ lapack xvfb

RUN mkdir /data
RUN chmod -R 777 /data
COPY data-paper /data-paper
RUN chmod -R 777 /data-paper
COPY src /src

WORKDIR /src
CMD main.py


FROM nemo-eva as nemo-eva-dev

RUN pip3 install pycodestyle pytest pytest-ordering
