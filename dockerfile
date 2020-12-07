FROM ubuntu:20.04

RUN apt-get -y update
RUN apt-get -y upgrade

RUN DEBIAN_FRONTEND=noninteractive apt-get -yq install build-essential gcc clang-tidy cmake openmpi-bin libopenmpi-dev

CMD [ "/bin/bash" ]
