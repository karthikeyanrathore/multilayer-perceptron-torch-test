FROM python:3.9-alpine3.13 

RUN apk --update  add \
    musl-dev python3-dev cargo libressl-dev \
    && pip3 install --upgrade pip 

WORKDIR /home/mlp-test

COPY mlp /home/mlp-test/mlp

# CMD ls mlp
CMD python3 -u mlp/mlp_test.py


