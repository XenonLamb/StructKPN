# Low-Dose CT Denoising with Kernel Prediction Networks

## Requirements

- numpy
- scipy
- Pillow
- matplotlib
- Pytorch (tested on 1.1.0)
- pydicom
- PyYAML
- astra-toolbox (if one needs to synthesize noisy inputs himself)

## Run the code

We use a .yml file to specify various experiment settings. By default, all .yml file are stored in `options` folder. To run the code:

```shell
python main.py -opt AAPMKPN.yml
```
A sample_option_with_doc.yml is provided to illustrate various settings.

### Prepare the data

Download the AAPM dataset into `data` folder.

The data used by the training and test script should be normalized and stored in .npz files.

The publicly available datasets include:

AAPM-NIH LDCT dataset 

DOSE dataset (Notice: the low-dose and full-dose images are not perfectly aligned, only for qualitative comparison): store in `bnm0xx-x-x.npz`

CDE dataset (patient lung), store in `CDE` folder.


Some scripts (`process_CDE.ipynb`,`process_phantom.ipynb`) are stored in the data directories for preprocessing the data.


## Train and Test

Config files like `xxxx_EVAL.yml` are used for evaluation.




