import numpy
import json
import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from clearml import Task, Logger

task = Task.init(project_name="MTQE", task_name="Evaluate Labels")

args = {
    'items' : 's3://lw-data/ahuang/en_nl_sample_outputs.jsonl',
    'out_file' : None,
    'gold_label_key' : "label",
    'pred_label_key' : "output",
    'pred_label_score_key' : "output_score",
    'label_set' : ["Good", "Adequate", "Poor"],
    'default_group_name' : "ALL",
    'aggregated_labels_name' : "ALL_LABELS",
    'group_by_key' : None
}

task.connect(args)
task.execute_remotely('mdr-cpu')


items_file = args['items']
with open(items_file) as f:
    items = [json.loads(item) for item in f]

group_names = []
if args['group_by_key']:
    if type(args['group_by_key']) == str:
        group_names.extend(list(set([item[args['group_by_key']] for item in items
                                     if args['group_by_key'] in item])))
    else:
        group_names.extend(list(set(["-".join([item[grp_by] for grp_by in args['group_by_key'] if grp_by in item])
                                     for item in items])))
if not group_names:
    group_names.append(args['default_group_name'])

results = {}
for group_name in group_names:
    if group_name != args['default_group_name']:
        if type(args['group_by_key']) == str:
            group_items = [item for item in items
                           if item[args['group_by_key']] == group_name]
        else:
            group_items = [item for item in items
                           if "-".join([item[grp_by] for grp_by in args['group_by_key'] if grp_by in item]) == group_name]
    else:
        group_items = items

    gold_labels = numpy.array([item[args['gold_label_key']]
                               for item in group_items])
    pred_labels = numpy.array([item[args['pred_label_key']]
                               for item in group_items])
    if not label_set:
        label_set = list(set(gold_labels))

    precs, recs, f1s, n_items = precision_recall_fscore_support(y_true=gold_labels,
                                                                y_pred=pred_labels,
                                                                labels=label_set)
    mean_prec, mean_rec, mean_f1, _ = precision_recall_fscore_support(y_true=gold_labels,
                                                                      y_pred=pred_labels,
                                                                      average="weighted")
    results[group_name] = {args['aggregated_labels_name']: {"precision": mean_prec, "recall": mean_rec, "f1": mean_f1, "n_items": len(group_items)},
                           **{label: {"precision": prec, "recall": rec, "f1": f1, "n_items": int(n)}
                              for label, prec, rec, f1, n in zip(label_set, precs, recs, f1s, n_items)}}

print(json.dumps(results, indent=4))

if not args['out_file']:
    out_file = "{}.results.json".format(os.path.splitext(items_file)[0])
with open(out_file, "w") as f:
    json.dump(results, f, indent=4)
print("Saved results to {}".format(out_file))

Logger.current_logger().report_table(pd.DataFrame.from_dict(results[args['default_group_name']]))


task.close()