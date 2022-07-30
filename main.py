# Nati Daniel General Framework
# Technion IIT, Savir Lab

import sys
sys.path.append('../')

from utils import logutils
from algo.common.pipeline import Pipeline
from algo.postprocessing.post_processing import post_process_data
 
import argparse
import logging

# Pipeline main
if __name__ == '__main__':
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("pipeline", help="path to pipeline json")
    arg_parse.add_argument("output_path", help="path to pipeline output path")
    args = arg_parse.parse_args()

    logutils.init_logger()
    logging.info("Pipeline Logger is ready for use")

    logging.info("Pipeline Initialized")
    pipeline: object = Pipeline()
    pipeline.deserialize(args.pipeline)
    logging.info("Pipeline Deserialized")

    logging.info("Running Pipeline")
    pipelineOutput = pipeline.run()
    logging.info("Pipeline Completed")

    logging.info("Post processing")
    post_process_data(pipelineOutput, args.output_path)
    logging.info("Pipeline Finished")