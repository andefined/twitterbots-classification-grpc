FROM python:3.6-onbuild

RUN mkdir -p /usr/src/app/
COPY . /usr/src/app/

EXPOSE 50051
WORKDIR /usr/src/app/

CMD ["python", "server.py"]