
import Philips_Application.philips_robustness as phlips_plots

from metroapplication.test_new_dataset import similarity_tech


def Azure_plots():
    from AzureApplication.Azure_plots import sensetivity
    from AzureApplication import Azure_plots
    Azure_plots.plot_nice()

    #experiments
    #sensetivity()

def Philips_plots():
    phlips_plots.plot_nice()


    # experiments in thresholding
    #phlips_plots.similarity_ocsvm()
    #phlips_plots.similarity_IF()
    #phlips_plots.similarity_PB()
    #phlips_plots.similarity_lof()
    #phlips_plots.plot_nice()

    # measuring the difference
    # phlips_plots.get_recall_precission_FPR_PB()
    # phlips_plots.get_recall_precission_FPR_LOF()
    # phlips_plots.get_recall_precission_FPR_OCSVM()
    # phlips_plots.get_recall_precission_FPR_IF()

def metro_plots():
    similarity_tech()



if __name__ == '__main__':
    Philips_plots()
    # Azure_plots()
    # metro_plots()

