from clearml.automation import PipelineController


def pre_execute_callback_example(a_pipeline, a_node, current_param_override):
    # type (PipelineController, PipelineController.Node, dict) -> bool
    print(
        "Cloning Task id={} with parameters: {}".format(
            a_node.base_task_id, current_param_override
        )
    )
    # if we want to skip this node (and subtree of this node) we return False
    # return True to continue DAG execution
    return True


def post_execute_callback_example(a_pipeline, a_node):
    # type (PipelineController, PipelineController.Node) -> None
    print("Completed Task id={}".format(a_node.executed))
    # if we need the actual executed Task: Task.get_task(task_id=a_node.executed)
    return


# Connecting ClearML with the current pipeline,
# from here on everything is logged automatically
pipe = PipelineController(
    name="Evaluate Labels", project="MTQE", version="0.0.1", add_pipeline_tags=False
)

pipe.add_parameter(
    "src",
    "/home/ahuang/Downloads/en_nl_sample_outputs.jsonl",
    "local source path",
)

pipe.add_parameter(
    "dst",
    "s3://lw-data/ahuang/en_nl_sample_outputs.jsonl",
    "remote destination path",
)

pipe.set_default_execution_queue("mdr-cpu")

pipe.add_step(
    name="uploadToS3",
    base_task_project="MTQE",
    base_task_name="Upload to S3",
    parameter_override={
        "General/src": "${pipeline.src}",
        "General/dst": "${pipeline.dst}"
    },
)

pipe.add_step(
    name="evaluateLabels",
    parents=["uploadToS3"],
    base_task_project="MTQE",
    base_task_name="Evaluate Labels",
    parameter_override={
        "General/s3_upload_task_id": "${uploadToS3.id}"
    },
    pre_execute_callback=pre_execute_callback_example,
    post_execute_callback=post_execute_callback_example,
)

# for debugging purposes use local jobs
# pipe.start_locally()

# Starting the pipeline (in the background)
pipe.start('mdr-cpu')

print("done")