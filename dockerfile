FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
MAINTAINER Ilya Trofimov
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -yqq update
RUN apt-get -yqq install git cmake vim wget

#
# Conda
#
RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
RUN chmod 777 ./Anaconda3-2020.11-Linux-x86_64.sh
RUN bash ./Anaconda3-2020.11-Linux-x86_64.sh -b -p /home/anaconda
ENV PATH="/home/anaconda/bin:$PATH"
RUN conda create -y -n py37 python=3.7 anaconda
RUN conda init bash
RUN conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

RUN pip install git+https://github.com/simonzhang00/ripser-plusplus.git
RUN pip install git+https://github.com/IlyaTrofimov/RTD.git

EXPOSE 8896
RUN echo "(jupyter notebook --ip=0.0.0.0 --port 8896 --allow-root &)" > start_jupyter.sh
CMD ["/bin/bash"]
