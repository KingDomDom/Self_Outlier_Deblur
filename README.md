# Project Name

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Description

This project aims to finish  《Pattern Recognition》  Curriculum-Design in AIA， HUST 

The report has been included whose name is '模式识别课程设计报告'

The review of literature has been included whose name is '模式识别综述'

The folder 'code' covers all of our models and experiments during this Curriculum-Design

In '/code':

	Branch OID-MATLAB:Outlier Identification Discarding Method(using MATLAB code)

	Branch MATLAB-experiments:All experiments related with OID(using MATLAB code)

	Branch OID-PyTorch:Outlier Identification Discarding Method(using Python code)

	Branch SelfDeblur:Neural Blind Deconvolution Using Deep Priors(using Python code)

	Branch SOD:Self Outlier Deblur(using Python Code)

- [Installation](#installation)
- [Usage](#usage)
- [Required](#contributing)
- [License](#license)

## Installation
Push the green button named 'Code',then Download ZIP,you can run all of the code locally

## Usage
Unzip what you have downloaded, and then extract all of them where you want.
Use platform MATLAB to run 'OID-MATLAB' and 'MATLAB-experiments' files
Use platform VSCOde or Pycharm to run 'OID-PyTorch' , 'SelfDeblur' and 'SOD' files

## Required
system:Ubuntu_22.04 & Windows 10/11
if code = Python ： #you need all moudles followed
  python --version = 3.12.4
  pytorch --version = 2.3.1 ;cuda --version = 12.1; cudnn -version = 8.9.2_0
  scikit-image --version = 0.23.2
  scipy --version = 1.13.1
  numpy --version = 1.26.4

And you need GPU with at least 1GB memory to process the image.

!!!Attention:the results you get are relied on your device and your moudle version(eg.processing time) 

## License

This project is licensed under the [MIT License](LICENSE).
