

# Example of usage:

Example.ipynb

## Reproducibility:

Using main_reproducibility.py we can fine tune methods and context to generate the results shown in paper.


### Code for Metro Dataset: MetroApplication

AzureMain.py : Used to  fine-tune hyperparameters for the different methods using Mango framework.
AzureFeedback: functions for running Application of Feedback in fine-tuned solutions for Metro Dataset.
Azure_plots: functions plotting results and testing the sensitivity of the regarding different threshold values.

Where the best parameters for azure are shown in best_parameters_for_Azure.txt

### Code for Philips Dataset: PhilipsApplocation

philips_application_main.py: functions for running Feedback application to Philips dataset
philips_main.py : Used to  fine-tune hyperparameters for the different methods using Mango framework in Philips data.
philips_robustness: functions plotting results and testing the sensitivity of the regarding different threshold values.
