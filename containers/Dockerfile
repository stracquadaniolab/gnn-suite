# TODO: fix pytorch version and torch_geometric version - consider pip install[?]

FROM pytorch/pytorch:latest

LABEL org.opencontainers.image.title="pytorch-geometric"
LABEL org.opencontainers.image.description="A Docker image with a full pytorch-geometric installation and support for CUDA (cu101)"
LABEL org.opencontainers.image.url="https://github.com/stracquadaniolab/docker-pytorch-geometric"
LABEL org.opencontainers.image.documentation="https://github.com/stracquadaniolab/docker-pytorch-geometric"
LABEL org.opencontainers.image.source="https://github.com/stracquadaniolab/docker-pytorch-geometric"
LABEL org.opencontainers.image.revision="v0.2.21"
LABEL org.opencontainers.image.vendor="stracquadaniolab"
LABEL org.opencontainers.image.authors="Giovanni Stracquadanio"

USER root
WORKDIR /
ARG CONDA_FILE=containers/environment.yml
ARG PACKAGE_FILE=containers/pkg.list

# copy package list
COPY ${PACKAGE_FILE} pkg.list

# Install basic tools
RUN apt-get update \
    && apt-get install --yes $(cat pkg.list) \
    && apt-get autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


COPY ${CONDA_FILE} environment.yml
RUN conda env update -n base --file environment.yml && conda clean --all --yes 


# Add Tini
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini

# setup CUDA variables and linking libraries
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
ENV CPATH=/usr/local/cuda/include:/usr/local/cuda/compat
ENV CUDA_HOME=/usr/local/cuda

ENV CUDA=cu101
RUN mkdir -p /usr/local/cuda/
RUN cd /usr/local/cuda/ && ln -s /usr/local/cuda/compat/libnvidia-* .

# installing pytorch geometric and its requirements
RUN conda install pyg -c pyg


#RUN pip install torch-scatter \
#    && pip install torch-sparse \
#    && pip install torch-cluster \
#    && pip install torch-spline-conv \
#    && pip install torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html \
#    && pip install ogb

ENTRYPOINT ["/tini", "--"]
CMD ["python3"]

