# Dockerfile to create bda_ca2 container
FROM centos:centos7

# install required pre-reqs
RUN yum install -y epel-release && \
    yum install -y python36 && \
    yum install -y python3-pip

# change to working directory
WORKDIR /opt/model

# copy required files
COPY app.py  create_diabetes_prediction_model.py  diabetes_prediction_dataset.csv  requirements.txt  ./
COPY templates templates

# pip install required libraries
RUN pip3 install joblib numpy pandas scikit-learn flask

# create model
RUN python3 create_diabetes_prediction_model.py

# set env varaibles
ENV LC_ALL=en_US.utf-8
ENV LANG=en_US.utf-8

# Expose port 5000 for Flask
EXPOSE 5000

# Start webserver
CMD ["flask", "run", "-h", "0.0.0.0"]
