from clearml import Task, StorageManager

args = {
    'src' : 'C:\\Users\\Andrew\\Downloads\\en_nl_sample_outputs.jsonl',
    'dst' : 's3://lw-data/ahuang/en_nl_sample_outputs.jsonl'
}
# create an dataset experiment
task = Task.init(project_name="examples", task_name="Upload to S3")
task.connect(args)

# only create the task, we will actually execute it later
task.running_locally()

# simulate local dataset, download one, so we have something local
sample_outputs = StorageManager.upload_file(args['src'], args['dst'])

# add and upload local file containing our toy dataset
task.upload_artifact('sample outputs', artifact_object=sample_outputs)

print('uploading artifacts in the background')

# we are done
print('Done')
task.close()