import os
import logging

from scipy import io


def post_process_data(pipelineOutput, output_path):
    # create output dir
    pipelineOutput_dir = os.path.join(output_path,'final_pipeline_output')
    if not os.path.isdir(pipelineOutput_dir):
        os.makedirs(pipelineOutput_dir)
    logging.info("Saving output results")

    # save specific results from pipelineOutput
    io.savemat(pipelineOutput_dir, pipelineOutput)