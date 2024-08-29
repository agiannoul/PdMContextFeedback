## Install Enviroment

If you don't have Conda installed, you can install it by downloading and installing Anaconda or Miniconda:
- [Anaconda](https://www.anaconda.com/products/individual)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)


### Create Enviroment
```sh
conda env create --file environment.yml

conda activate ContextFeedbackPdm
```
# Example of usage:

```sh
conda activate ContextFeedbackPdm
# run jupyter lab
jupyter-lab
```
Navigate to : Example.ipynb

## Reproducibility:

Using main_reproducibility.py we can fine tune methods and context to generate the results shown in paper.


### Code for Azure Dataset: AzureApplication

AzureMain.py : Used to  fine-tune hyperparameters for the different methods using Mango framework.
AzureFeedback: functions for running Application of Feedback in fine-tuned solutions for Metro Dataset.
Azure_plots: functions plotting results and testing the sensitivity of the regarding different threshold values.

Where the best parameters for azure are shown in best_parameters_for_Azure.txt

### Code for Philips Dataset: PhilipsApplocation

philips_application_main.py: functions for running Feedback application to Philips dataset
philips_main.py : Used to  fine-tune hyperparameters for the different methods using Mango framework in Philips data.
philips_robustness: functions plotting results and testing the sensitivity of the regarding different threshold values.
