# Dockerfile for for R base image with INLA installed
# after installation, can be used interactively with:
# sudo docker run --rm -it docker_r_base bash
FROM rstudio/r-base:4.4-jammy
RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker
RUN echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker
RUN DEBIAN_FRONTEND=noninteractive \
  apt-get update && \
  apt-get install -y libudunits2-dev libgdal-dev libssl-dev libfontconfig1-dev libgsl-dev
#RUN R -e 'r = getOption("repos"); r["CRAN"] = "http://cran.us.r-project.org"; options(repos = r); install.packages("INLA", repos=c(getOption("repos")), dep=TRUE)'
RUN R -e 'r = getOption("repos"); r["CRAN"] = "http://cran.us.r-project.org"; options(repos = r); install.packages("remotes", repos=c(getOption("repos")), dep=TRUE)'
RUN R -e 'r = getOption("repos"); r["CRAN"] = "http://cran.us.r-project.org"; options(repos = r); library(remotes); remotes::install_version("INLA", version="24.05.10",repos=c(getOption("repos"), INLA="https://inla.r-inla-download.org/R/testing"), dep=TRUE)'
#RUN R -e 'r = getOption("repos"); r["CRAN"] = "http://cran.us.r-project.org"; options(repos = r); install.packages("https://inla.r-inla-download.org/R/stable/src/contrib/INLA_24.12.11.tar.gz", repos=c(getOption("repos")), dep=TRUE)'
RUN R -e 'r = getOption("repos"); r["CRAN"] = "http://cran.us.r-project.org"; options(repos = r); install.packages("tidyverse", repos=c(getOption("repos"), dep=TRUE))'
RUN R -e 'r = getOption("repos"); r["CRAN"] = "http://cran.us.r-project.org"; options(repos = r); install.packages(c("tsModel", "dlnm"), repos=c(getOption("repos"), dep=TRUE))'
RUN R -e 'r = getOption("repos"); r["CRAN"] = "http://cran.us.r-project.org"; options(repos = r); install.packages(c("spdep"), repos=c(getOption("repos"), dep=TRUE))'
RUN rm -rf /var/lib/apt/lists/*
RUN useradd -ms /bin/bash apprunner
RUN echo "apprunner:apprunner" | chpasswd
#RUN chmod -R 777  /opt/R/*/lib/R/library/INLA/
RUN chmod -R 777  /opt/R/*
#RUN chmod -R 777 /opt/R/*/lib/R/library
#RUN chmod -R 777 /opt/R/*/lib/R/
#USER apprunner

#docker run --rm -it rstudio/r-base:4.3-jammy

# sudo docker run -ti --rm -v "./:/home/run/" -w /home/run/ docker_r_base Rscript setup.R

# to build:
#  - docker build -t ivargr/r_inla .